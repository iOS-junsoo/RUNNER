from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from celeba_dataset import build_celeba_datasets
from celeba_main import resolve_device
from celeba_model import CelebARunnerModel
from celeba_utils import equalized_odds_gap_sum, make_eval_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 masking analysis for CelebA checkpoints")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to a saved checkpoint")
    parser.add_argument("--data_root", default="/workspace/RUNNER/Data", type=str, help="CelebA root path")
    parser.add_argument("--image_size", default=224, type=int, help="Center crop size")
    parser.add_argument("--batch_size", default=256, type=int, help="Evaluation batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Evaluation dataloader workers")
    parser.add_argument(
        "--calibration_batch_size",
        default=256,
        type=int,
        help="Per-group batch size used to estimate fairness-importance rankings",
    )
    parser.add_argument(
        "--calibration_steps",
        default=20,
        type=int,
        help="Number of grouped calibration steps used to estimate rankings",
    )
    parser.add_argument("--device", default="auto", type=str, help="auto/cpu/cuda")
    parser.add_argument("--amp", action="store_true", help="Enable AMP during inference on CUDA")
    parser.add_argument("--mask_splits", default=5, type=int, help="Number of equal mask buckets to evaluate")
    parser.add_argument("--output", default=None, type=str, help="Optional JSON output path")
    return parser.parse_args()


def _group_loader(dataset, indices: np.ndarray, batch_size: int, num_workers: int, pin_memory: bool):
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    return make_eval_loader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


def _estimate_importance_rankings(
    model: CelebARunnerModel,
    dataset,
    batch_size: int,
    steps: int,
    num_workers: int,
    device: torch.device,
) -> List[torch.Tensor]:
    model.eval()
    weights = model.get_runner_weights()
    criteria_sums = [torch.zeros(weight.shape[0], device=device) for weight in weights]

    group_loaders = {
        key: _group_loader(
            dataset,
            dataset.group_indices[key],
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        for key in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    group_iterators = {key: iter(loader) for key, loader in group_loaders.items()}

    effective_steps = 0
    for _ in range(steps):
        batches = {}
        exhausted = False
        for key, iterator in group_iterators.items():
            try:
                batches[key] = next(iterator)
            except StopIteration:
                exhausted = True
                break
        if exhausted:
            break

        batch_x_00 = batches[(0, 0)][0].to(device, non_blocking=True)
        batch_x_01 = batches[(0, 1)][0].to(device, non_blocking=True)
        batch_x_10 = batches[(1, 0)][0].to(device, non_blocking=True)
        batch_x_11 = batches[(1, 1)][0].to(device, non_blocking=True)

        loss_reg = torch.zeros((), device=device)
        for batch_x_0, batch_x_1 in ((batch_x_00, batch_x_10), (batch_x_01, batch_x_11)):
            output_0, _ = model(batch_x_0)
            output_1, _ = model(batch_x_1)
            loss_reg = loss_reg + torch.abs(output_0.mean() - output_1.mean())

        grads = torch.autograd.grad(loss_reg, weights, retain_graph=False, allow_unused=True)
        for index, (weight, grad) in enumerate(zip(weights, grads)):
            if grad is None:
                continue
            criteria_sums[index] += (weight * grad).detach().pow(2).view(weight.shape[0], -1).sum(dim=1)
        model.zero_grad(set_to_none=True)
        effective_steps += 1

    if effective_steps == 0:
        raise RuntimeError("Could not estimate importance rankings because one or more EO groups had no calibration batches.")

    model.eval()
    return [criteria.argsort(descending=True) for criteria in criteria_sums]


def _run_masked_predictions(
    model: CelebARunnerModel,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
    mask_indices: Sequence[torch.Tensor | None] | None,
):
    loader = make_eval_loader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    criterion = nn.BCELoss(reduction="sum")
    y_true = []
    y_score = []
    y_pred = []
    sensitive = []
    loss_sum = 0.0

    autocast_enabled = use_amp and device.type == "cuda"
    model.eval()
    with torch.no_grad():
        for images, labels, groups in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1).float()
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=autocast_enabled):
                if mask_indices is None:
                    outputs, _ = model(images)
                else:
                    outputs = model.mask_forward(images, mask_indices)
            loss_sum += float(criterion(outputs.float(), labels).item())
            y_true.append(labels.squeeze(1).cpu().numpy().astype(np.int64))
            scores = outputs.squeeze(1).cpu().numpy()
            y_score.append(scores)
            y_pred.append((scores >= 0.5).astype(np.int64))
            sensitive.append(groups.numpy().astype(np.int64))

    y_true_np = np.concatenate(y_true, axis=0)
    y_score_np = np.concatenate(y_score, axis=0)
    y_pred_np = np.concatenate(y_pred, axis=0)
    sensitive_np = np.concatenate(sensitive, axis=0)

    return {
        "mean_bce_loss": loss_sum / len(dataset),
        "average_precision": float(average_precision_score(y_true_np, y_score_np)),
        "eo_gap": equalized_odds_gap_sum(y_true_np, y_pred_np, sensitive_np),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    task = checkpoint.get("task", "wavy_hair")
    model_name = checkpoint.get("model_name", "resnet18")

    _, _, test_dataset = build_celeba_datasets(
        root=args.data_root,
        task=task,
        image_size=args.image_size,
        download=False,
    )

    model = CelebARunnerModel(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rankings = _estimate_importance_rankings(
        model,
        test_dataset,
        batch_size=args.calibration_batch_size,
        steps=args.calibration_steps,
        num_workers=args.num_workers,
        device=device,
    )

    baseline_metrics = _run_masked_predictions(
        model,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        use_amp=args.amp,
        mask_indices=None,
    )

    mask_results = []
    for split_index in range(args.mask_splits):
        mask_indices = []
        for ranking in rankings:
            chunk = int(math.ceil(len(ranking) / args.mask_splits))
            start = split_index * chunk
            end = min(len(ranking), start + chunk)
            mask_indices.append(ranking[start:end].to(device))

        masked_metrics = _run_masked_predictions(
            model,
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_amp=args.amp,
            mask_indices=mask_indices,
        )
        mask_results.append(
            {
                "bucket_index": split_index,
                "bucket_percent_start": (100 * split_index) / args.mask_splits,
                "bucket_percent_end": (100 * (split_index + 1)) / args.mask_splits,
                "masked_units_per_layer": [int(mask.numel()) for mask in mask_indices],
                "metrics": masked_metrics,
                "delta": {
                    key: masked_metrics[key] - baseline_metrics[key]
                    for key in baseline_metrics
                },
            }
        )

    result = {
        "phase": 1,
        "analysis": "mask_top_fairness_importance_buckets",
        "checkpoint": args.checkpoint,
        "task": task,
        "model_name": model_name,
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "calibration_batch_size": args.calibration_batch_size,
        "calibration_steps": args.calibration_steps,
        "mask_splits": args.mask_splits,
        "baseline": baseline_metrics,
        "mask_results": mask_results,
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