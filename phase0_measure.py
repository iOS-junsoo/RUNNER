from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from celeba_dataset import build_celeba_datasets
from celeba_main import resolve_device
from celeba_model import CelebARunnerModel
from celeba_utils import equalized_odds_gap_sum, make_eval_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 0 baseline metric measurement")
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to a saved checkpoint")
    parser.add_argument("--data_root", default="/workspace/RUNNER/Data", type=str, help="CelebA root path")
    parser.add_argument("--task", default="wavy_hair", choices=["wavy_hair", "attractive"], help="Task name")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "alexnet"], help="Model architecture")
    parser.add_argument("--image_size", default=224, type=int, help="Center crop size")
    parser.add_argument("--batch_size", default=256, type=int, help="Evaluation batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Evaluation dataloader workers")
    parser.add_argument("--device", default="auto", type=str, help="auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", help="Enable AMP during inference on CUDA")
    parser.add_argument("--seed", default=0, type=int, help="Random seed used when measuring from a fresh pretrained initialization")
    parser.add_argument(
        "--use_pretrained_init",
        action="store_true",
        help="Measure Phase 0 from a fresh pretrained backbone with a newly initialized binary head instead of loading a checkpoint",
    )
    parser.add_argument("--output", default=None, type=str, help="Optional JSON output path")
    args = parser.parse_args()
    if not args.use_pretrained_init and not args.checkpoint:
        parser.error("Either --checkpoint must be provided or --use_pretrained_init must be set.")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = None

    if args.use_pretrained_init:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        task = args.task
        model_name = args.model
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        task = checkpoint.get("task", args.task)
        model_name = checkpoint.get("model_name", args.model)

    _, _, test_dataset = build_celeba_datasets(
        root=args.data_root,
        task=task,
        image_size=args.image_size,
        download=False,
    )

    model = CelebARunnerModel(model_name, pretrained=args.use_pretrained_init).to(device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.BCELoss(reduction="sum")
    loader = make_eval_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    y_true = []
    y_score = []
    y_pred = []
    sensitive = []
    loss_sum = 0.0

    autocast_enabled = args.amp and device.type == "cuda"
    with torch.no_grad():
        for images, labels, groups in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1).float()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                outputs, _ = model(images)
            loss_sum += float(criterion(outputs.float(), labels).item())
            y_true.append(labels.squeeze(1).cpu())
            y_score.append(outputs.squeeze(1).cpu())
            y_pred.append((outputs >= 0.5).long().squeeze(1).cpu())
            sensitive.append(groups.cpu())

    y_true_tensor = torch.cat(y_true)
    y_score_tensor = torch.cat(y_score)
    y_pred_tensor = torch.cat(y_pred)
    sensitive_tensor = torch.cat(sensitive)

    mean_bce_loss = loss_sum / len(test_dataset)
    average_precision = average_precision_score(
        y_true_tensor.numpy().astype("int64"),
        y_score_tensor.numpy(),
    )
    eo_gap = equalized_odds_gap_sum(
        y_true_tensor.numpy().astype("int64"),
        y_pred_tensor.numpy().astype("int64"),
        sensitive_tensor.numpy().astype("int64"),
    )

    result = {
        "measurement_source": "pretrained_init" if args.use_pretrained_init else "checkpoint",
        "checkpoint": args.checkpoint,
        "task": task,
        "model_name": model_name,
        "seed": args.seed if args.use_pretrained_init else checkpoint.get("seed"),
        "epoch": None if args.use_pretrained_init else checkpoint.get("epoch"),
        "mean_bce_loss": mean_bce_loss,
        "average_precision": float(average_precision),
        "eo_gap": eo_gap,
        "saved_metrics": {} if checkpoint is None else checkpoint.get("metrics", {}),
    }

    print(json.dumps(result, indent=2))

    if args.output is not None:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()