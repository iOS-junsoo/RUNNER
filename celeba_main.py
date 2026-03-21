from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from pytorch_lightning import seed_everything
except ModuleNotFoundError:
    def seed_everything(seed: int) -> int:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

from celeba_dataset import build_celeba_datasets
from celeba_model import CelebARunnerModel
from celeba_utils import evaluate_dp, evaluate_eo, train_dp, train_eo


TASK_DEFAULT_MODEL = {
    "attractive": "alexnet",
    "wavy_hair": "resnet18",
}

TASK_FILE_STEM = {
    "attractive": "attr",
    "wavy_hair": "wavy",
}


class TeeLogger:
    def __init__(self, log_path: str) -> None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.stream = open(log_path, "w", encoding="utf-8")

    def log(self, *parts) -> None:
        message = " ".join(str(part) for part in parts)
        print(message)
        self.stream.write(message + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_name(task: str, model_name: str) -> str:
    if model_name == "auto":
        return TASK_DEFAULT_MODEL[task]
    return model_name


def checkpoint_path(results_dir: str, task: str, method: str, mode: str, seed: int) -> str:
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(
        checkpoint_dir,
        f"celeba_{TASK_FILE_STEM[task]}_{method}_{mode}_seed{seed}_best.pt",
    )


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    *,
    seed: int,
    epoch: int,
    task: str,
    method: str,
    mode: str,
    model_name: str,
    ap_val: float,
    gap_val: float,
    ap_test: float,
    gap_test: float,
) -> None:
    torch.save(
        {
            "seed": seed,
            "epoch": epoch,
            "task": task,
            "method": method,
            "mode": mode,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {
                "ap_val": ap_val,
                "gap_val": gap_val,
                "ap_test": ap_test,
                "gap_test": gap_test,
            },
        },
        path,
    )


def run_experiments(args) -> None:
    device = resolve_device(args.device)
    model_name = resolve_model_name(args.task, args.model)
    results_path = os.path.join(args.results_dir, f"celeba_{TASK_FILE_STEM[args.task]}_{args.mode}.txt")
    logger = TeeLogger(results_path)

    logger.log(f"Task: {args.task}")
    logger.log(f"Model: {model_name}")
    logger.log(f"Method: {args.method}")
    logger.log(f"Mode: {args.mode}")
    logger.log(f"Device: {device}")
    logger.log(f"Pretrained: {args.pretrained}")
    logger.log(f"AMP: {args.amp and device.type == 'cuda'}")
    logger.log(f"Train workers: {args.train_num_workers}")
    logger.log(f"Eval workers: {args.num_workers}")

    train_dataset, val_dataset, test_dataset = build_celeba_datasets(
        root=args.data_root,
        task=args.task,
        image_size=args.image_size,
        download=args.download,
    )

    ap_results = []
    gap_results = []

    try:
        for seed in range(args.ex_num):
            seed_everything(seed)
            logger.log(f"Seed set to {seed}")
            logger.log(f"On experiment {seed}")

            model = CelebARunnerModel(model_name, pretrained=args.pretrained).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.BCELoss()
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

            ap_val_epoch = []
            gap_val_epoch = []
            ap_test_epoch = []
            gap_test_epoch = []
            start_time = time.time()
            best_gap_val = float("inf")
            best_checkpoint = checkpoint_path(args.results_dir, args.task, args.method, args.mode, seed)

            for epoch in range(args.epochs):
                logger.log("")
                logger.log(f"Epoch: {epoch}")
                log_prefix = f"[seed {seed}][epoch {epoch}] "
                if args.mode == "dp":
                    train_stats = train_dp(
                        model,
                        criterion,
                        optimizer,
                        train_dataset,
                        args.method,
                        args.lam,
                        args.neuron_ratio,
                        batch_size=args.dp_batch_size,
                        steps_per_epoch=args.steps_per_epoch,
                        device=device,
                        num_workers=args.train_num_workers,
                        use_amp=args.amp,
                        scaler=scaler,
                        logger=logger.log,
                        log_prefix=log_prefix,
                        log_interval=args.log_interval,
                    )
                    ap_val, gap_val = evaluate_dp(
                        model,
                        val_dataset,
                        batch_size=args.eval_batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        use_amp=args.amp,
                    )
                    ap_test, gap_test = evaluate_dp(
                        model,
                        test_dataset,
                        batch_size=args.eval_batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        use_amp=args.amp,
                    )
                else:
                    train_stats = train_eo(
                        model,
                        criterion,
                        optimizer,
                        train_dataset,
                        args.method,
                        args.lam,
                        args.neuron_ratio,
                        batch_size=args.eo_batch_size,
                        steps_per_epoch=args.steps_per_epoch,
                        device=device,
                        num_workers=args.train_num_workers,
                        use_amp=args.amp,
                        scaler=scaler,
                        logger=logger.log,
                        log_prefix=log_prefix,
                        log_interval=args.log_interval,
                    )
                    ap_val, gap_val = evaluate_eo(
                        model,
                        val_dataset,
                        batch_size=args.eval_batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        use_amp=args.amp,
                    )
                    ap_test, gap_test = evaluate_eo(
                        model,
                        test_dataset,
                        batch_size=args.eval_batch_size,
                        num_workers=args.num_workers,
                        device=device,
                        use_amp=args.amp,
                    )

                ap_val_epoch.append(ap_val)
                gap_val_epoch.append(gap_val)
                ap_test_epoch.append(ap_test)
                gap_test_epoch.append(gap_test)

                logger.log(f"loss_reg: {train_stats['loss_reg']:.6f}")
                logger.log(f"loss_sup: {train_stats['loss_sup']:.6f}")
                logger.log(f"ap_test: {ap_test:.6f}")
                logger.log(f"gap_test: {gap_test:.6f}")

                if gap_val < best_gap_val:
                    best_gap_val = gap_val
                    save_checkpoint(
                        best_checkpoint,
                        model,
                        optimizer,
                        seed=seed,
                        epoch=epoch,
                        task=args.task,
                        method=args.method,
                        mode=args.mode,
                        model_name=model_name,
                        ap_val=ap_val,
                        gap_val=gap_val,
                        ap_test=ap_test,
                        gap_test=gap_test,
                    )
                    logger.log(f"checkpoint_saved: {best_checkpoint}")

            best_idx = int(np.argmin(gap_val_epoch))
            ap_results.append(ap_test_epoch[best_idx])
            gap_results.append(gap_test_epoch[best_idx])
            logger.log("--------INDEX---------")
            logger.log(f"idx: {best_idx + 1}")
            logger.log(f"ap_test: {ap_test_epoch[best_idx]:.6f}")
            logger.log(f"gap_test: {gap_test_epoch[best_idx]:.6f}")
            logger.log(f"time costs:{time.time() - start_time:.6f} s")

        logger.log("--------AVG---------")
        logger.log(f"Average Precision: {np.mean(ap_results):.6f} ± {np.std(ap_results):.6f}")
        logger.log(f"{args.mode} gap: {np.mean(gap_results):.6f} ± {np.std(gap_results):.6f}")
    finally:
        logger.close()


def build_parser():
    parser = argparse.ArgumentParser(description="CelebA Experiment for RUNNER")
    parser.add_argument("--method", default="van", type=str, help="van/GapReg/mixup/NeuronImportance_GapReg")
    parser.add_argument("--mode", default="eo", choices=["dp", "eo"], help="Fairness metric")
    parser.add_argument("--lam", default=1.0, type=float, help="Lambda for regularization")
    parser.add_argument("--neuron_ratio", default=0.05, type=float, help="Ratio of top-k important neurons/channels")
    parser.add_argument("--ex_num", default=10, type=int, help="Number of repeated experiments")
    parser.add_argument("--task", default="wavy_hair", choices=["wavy_hair", "attractive"], help="CelebA task")
    parser.add_argument("--model", default="auto", choices=["auto", "resnet18", "alexnet"], help="Backbone model")
    parser.add_argument("--epochs", default=5, type=int, help="Training epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--dp_batch_size", default=64, type=int, help="Per-group batch size for DP training")
    parser.add_argument("--eo_batch_size", default=128, type=int, help="Per-subgroup batch size for EO training")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Evaluation batch size")
    parser.add_argument("--steps_per_epoch", default=None, type=int, help="Optional override for training steps per epoch")
    parser.add_argument("--data_root", default="/workspace/RUNNER/Data", type=str, help="CelebA root path")
    parser.add_argument("--results_dir", default="paper_reproduction_results", type=str, help="Results directory")
    parser.add_argument("--image_size", default=224, type=int, help="Center crop size")
    parser.add_argument("--train_num_workers", default=8, type=int, help="Training dataloader workers")
    parser.add_argument("--num_workers", default=4, type=int, help="Evaluation dataloader workers")
    parser.add_argument("--log_interval", default=10, type=int, help="Print training progress every N steps")
    parser.add_argument("--device", default="auto", type=str, help="auto/cpu/cuda")
    parser.add_argument("--download", action="store_true", help="Download CelebA if missing")
    parser.add_argument("--scratch", action="store_true", help="Disable pretrained ImageNet weights")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision")
    parser.set_defaults(amp=True)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.pretrained = not args.scratch
    run_experiments(args)