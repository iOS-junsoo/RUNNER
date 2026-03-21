from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from celeba_dataset import build_celeba_datasets
from celeba_main import resolve_device
from celeba_model import CelebARunnerModel
from celeba_utils import equalized_odds_gap_sum, make_eval_loader


@dataclass
class Metrics:
    mean_bce_loss: float
    average_precision: float
    eo_gap: float


def parse_args():
    parser = argparse.ArgumentParser(description="Weight-level RUNNER UI vs FairGRAPE correlation analysis")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to a saved checkpoint")
    parser.add_argument("--data_root", default="/workspace/RUNNER/Data", type=str, help="CelebA root path")
    parser.add_argument("--image_size", default=224, type=int, help="Center crop size")
    parser.add_argument("--batch_size", default=256, type=int, help="Evaluation batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Evaluation dataloader workers")
    parser.add_argument("--device", default="auto", type=str, help="auto/cpu/cuda")
    parser.add_argument("--layer", default="layer1", choices=["layer1", "layer4"], help="Analysis layer")
    parser.add_argument("--subset_size", default=2000, type=int, help="Test subset size for actual delta EO measurements")
    parser.add_argument("--top_k", default=500, type=int, help="Number of top UI weights to sample")
    parser.add_argument("--bottom_k", default=500, type=int, help="Number of bottom UI weights to sample")
    parser.add_argument("--random_k", default=500, type=int, help="Number of random weights to sample")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for sampling and subset selection")
    parser.add_argument(
        "--results_dir",
        default="correlation_results",
        type=str,
        help="Output directory for json and plot files",
    )
    parser.add_argument("--plot_prefix", default=None, type=str, help="Optional output file prefix")
    return parser.parse_args()


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_target_weight(model: CelebARunnerModel, layer: str) -> tuple[str, torch.nn.Parameter]:
    if model.architecture != "resnet18":
        raise ValueError("This analysis currently supports resnet18 checkpoints only.")
    if layer == "layer1":
        return "backbone.layer1[0].conv1.weight", model.backbone.layer1[0].conv1.weight
    return "backbone.layer4[1].conv2.weight", model.backbone.layer4[1].conv2.weight


def materialize_subset(dataset: Dataset, indices: np.ndarray) -> TensorDataset:
    images = []
    labels = []
    groups = []
    for index in indices.tolist():
        image, label, group = dataset[int(index)]
        images.append(image)
        labels.append(label)
        groups.append(group)
    return TensorDataset(
        torch.stack(images, dim=0),
        torch.stack(labels, dim=0),
        torch.stack(groups, dim=0),
    )


def evaluate_metrics(model, dataset, batch_size: int, num_workers: int, device: torch.device) -> Metrics:
    loader = make_eval_loader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    criterion = nn.BCELoss(reduction="sum")
    y_true = []
    y_score = []
    y_pred = []
    sensitive = []
    loss_sum = 0.0

    model.eval()
    with torch.no_grad():
        for images, labels, groups in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1).float()
            outputs, _ = model(images)
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
    positive_mask = y_true_np == 1
    if positive_mask.any():
        average_precision = float(np.sum(y_score_np[positive_mask]) / max(1, positive_mask.sum()))
    else:
        average_precision = 0.0
    try:
        from sklearn.metrics import average_precision_score

        average_precision = float(average_precision_score(y_true_np, y_score_np))
    except Exception:
        pass

    return Metrics(
        mean_bce_loss=loss_sum / len(dataset),
        average_precision=average_precision,
        eo_gap=equalized_odds_gap_sum(y_true_np, y_pred_np, sensitive_np),
    )


def build_group_loaders(dataset, batch_size: int, num_workers: int, device: torch.device):
    return {
        key: make_eval_loader(
            Subset(dataset, dataset.group_indices[key].tolist()),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        for key in ((0, 0), (0, 1), (1, 0), (1, 1))
    }


def mean_output_gradient(model, loader: DataLoader, target_weight: torch.nn.Parameter, device: torch.device):
    grad_sum = torch.zeros_like(target_weight, device=device)
    output_sum = 0.0
    count = 0
    model.eval()
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        outputs, _ = model(images)
        batch_sum = outputs.sum()
        grad_batch = torch.autograd.grad(batch_sum, target_weight, retain_graph=False, allow_unused=False)[0]
        grad_sum += grad_batch.detach()
        output_sum += float(batch_sum.detach().item())
        count += images.shape[0]
        model.zero_grad(set_to_none=True)
    if count == 0:
        raise RuntimeError("Encountered an empty EO subgroup while computing RUNNER UI.")
    return grad_sum / count, output_sum / count


def mean_loss_gradient(model, loader: DataLoader, target_weight: torch.nn.Parameter, device: torch.device):
    criterion = nn.BCELoss(reduction="sum")
    grad_sum = torch.zeros_like(target_weight, device=device)
    loss_sum = 0.0
    count = 0
    model.eval()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1).float()
        outputs, _ = model(images)
        batch_loss = criterion(outputs.float(), labels)
        grad_batch = torch.autograd.grad(batch_loss, target_weight, retain_graph=False, allow_unused=False)[0]
        grad_sum += grad_batch.detach()
        loss_sum += float(batch_loss.detach().item())
        count += images.shape[0]
        model.zero_grad(set_to_none=True)
    return grad_sum / count, loss_sum / count


def compute_runner_ui(model, dataset, target_weight: torch.nn.Parameter, batch_size: int, num_workers: int, device: torch.device):
    group_loaders = build_group_loaders(dataset, batch_size=batch_size, num_workers=num_workers, device=device)
    grad_00, mean_00 = mean_output_gradient(model, group_loaders[(0, 0)], target_weight, device)
    grad_01, mean_01 = mean_output_gradient(model, group_loaders[(0, 1)], target_weight, device)
    grad_10, mean_10 = mean_output_gradient(model, group_loaders[(1, 0)], target_weight, device)
    grad_11, mean_11 = mean_output_gradient(model, group_loaders[(1, 1)], target_weight, device)

    sign_0 = 0.0 if mean_00 == mean_10 else math.copysign(1.0, mean_00 - mean_10)
    sign_1 = 0.0 if mean_01 == mean_11 else math.copysign(1.0, mean_01 - mean_11)
    fairness_grad = sign_0 * (grad_00 - grad_10) + sign_1 * (grad_01 - grad_11)
    return (fairness_grad * target_weight.detach()).pow(2)


def compute_performance_importance(model, dataset, target_weight: torch.nn.Parameter, batch_size: int, num_workers: int, device: torch.device):
    loader = make_eval_loader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=device.type == "cuda")
    perf_grad, mean_loss = mean_loss_gradient(model, loader, target_weight, device)
    return (perf_grad * target_weight.detach()).pow(2), mean_loss


def sample_weight_indices(ui_flat: np.ndarray, top_k: int, bottom_k: int, random_k: int, seed: int) -> np.ndarray:
    total = ui_flat.size
    top_k = min(top_k, total)
    bottom_k = min(bottom_k, max(0, total - top_k))
    order = np.argsort(ui_flat)
    bottom = order[:bottom_k]
    top = order[-top_k:] if top_k > 0 else np.array([], dtype=np.int64)
    used = np.zeros(total, dtype=bool)
    used[top] = True
    used[bottom] = True
    candidates = np.flatnonzero(~used)
    rng = np.random.default_rng(seed)
    random_k = min(random_k, candidates.size)
    random_part = rng.choice(candidates, size=random_k, replace=False) if random_k > 0 else np.array([], dtype=np.int64)
    sampled = np.concatenate([top, bottom, random_part]).astype(np.int64)
    return sampled


def iter_subset_batches(dataset, batch_size: int):
    for start in range(0, len(dataset), batch_size):
        end = min(len(dataset), start + batch_size)
        images = dataset.tensors[0][start:end]
        labels = dataset.tensors[1][start:end]
        groups = dataset.tensors[2][start:end]
        yield images, labels, groups


def evaluate_subset_eo(model, subset_dataset: TensorDataset, batch_size: int, device: torch.device) -> Metrics:
    criterion = nn.BCELoss(reduction="sum")
    y_true = []
    y_score = []
    y_pred = []
    sensitive = []
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels, groups in iter_subset_batches(subset_dataset, batch_size):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1, 1).float()
            outputs, _ = model(images)
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
    from sklearn.metrics import average_precision_score

    return Metrics(
        mean_bce_loss=loss_sum / len(subset_dataset),
        average_precision=float(average_precision_score(y_true_np, y_score_np)),
        eo_gap=equalized_odds_gap_sum(y_true_np, y_pred_np, sensitive_np),
    )


def compute_masked_deltas(
    model,
    target_weight: torch.nn.Parameter,
    subset_dataset: TensorDataset,
    sampled_indices: np.ndarray,
    ui_flat: np.ndarray,
    perf_flat: np.ndarray,
    baseline_subset: Metrics,
    batch_size: int,
    device: torch.device,
):
    results = []
    weight_data = target_weight.data
    for order_index, flat_index in enumerate(sampled_indices.tolist(), start=1):
        weight_index = np.unravel_index(int(flat_index), tuple(weight_data.shape))
        original_value = weight_data[weight_index].item()
        weight_data[weight_index] = 0.0
        masked_metrics = evaluate_subset_eo(model, subset_dataset, batch_size=batch_size, device=device)
        weight_data[weight_index] = original_value

        results.append(
            {
                "sample_order": order_index,
                "flat_index": int(flat_index),
                "weight_index": [int(value) for value in weight_index],
                "runner_ui": float(ui_flat[flat_index]),
                "performance_importance": float(perf_flat[flat_index]),
                "masked_eo_gap": masked_metrics.eo_gap,
                "masked_mean_bce_loss": masked_metrics.mean_bce_loss,
                "masked_average_precision": masked_metrics.average_precision,
                "delta_eo_improvement": baseline_subset.eo_gap - masked_metrics.eo_gap,
                "delta_eo_signed": masked_metrics.eo_gap - baseline_subset.eo_gap,
                "delta_eo_abs": abs(masked_metrics.eo_gap - baseline_subset.eo_gap),
            }
        )

        if order_index % 100 == 0 or order_index == len(sampled_indices):
            print(f"masked weights evaluated: {order_index}/{len(sampled_indices)}")
    return results


def correlation_summary(x: np.ndarray, y: np.ndarray):
    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def save_scatter(x: np.ndarray, y: np.ndarray, x_label: str, y_label: str, title: str, path: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(np.log10(x + 1e-12), y, s=12, alpha=0.45)
    plt.xlabel(f"log10({x_label})")
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_decile_plot(ui_values: np.ndarray, delta_values: np.ndarray, path: str):
    order = np.argsort(ui_values)
    splits = np.array_split(order, 10)
    means = [float(delta_values[index_group].mean()) if len(index_group) else 0.0 for index_group in splits]
    labels = [f"D{i}" for i in range(1, 11)]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, means)
    plt.xlabel("UI decile")
    plt.ylabel("Mean absolute delta EO")
    plt.title("UI decile vs mean absolute delta EO")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)

    checkpoint = safe_torch_load(args.checkpoint, device)
    task = checkpoint.get("task", "wavy_hair")
    model_name = checkpoint.get("model_name", "resnet18")
    if model_name != "resnet18":
        raise ValueError("The current design file targets ResNet-18 only.")

    _, _, test_dataset = build_celeba_datasets(
        root=args.data_root,
        task=task,
        image_size=args.image_size,
        download=False,
    )

    model = CelebARunnerModel(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_name, target_weight = resolve_target_weight(model, args.layer)
    print(f"target layer: {target_name} shape={tuple(target_weight.shape)}")

    full_baseline = evaluate_metrics(model, test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    print(json.dumps({
        "phase0_baseline": {
            "mean_bce_loss": full_baseline.mean_bce_loss,
            "average_precision": full_baseline.average_precision,
            "eo_gap": full_baseline.eo_gap,
        }
    }, indent=2))

    runner_ui = compute_runner_ui(
        model,
        test_dataset,
        target_weight=target_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    performance_importance, mean_loss = compute_performance_importance(
        model,
        test_dataset,
        target_weight=target_weight,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    ui_flat = runner_ui.detach().cpu().numpy().reshape(-1)
    perf_flat = performance_importance.detach().cpu().numpy().reshape(-1)
    sampled_indices = sample_weight_indices(
        ui_flat,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        random_k=args.random_k,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    subset_size = min(args.subset_size, len(test_dataset))
    subset_indices = rng.choice(len(test_dataset), size=subset_size, replace=False).astype(np.int64)
    subset_dataset = materialize_subset(test_dataset, subset_indices)
    subset_baseline = evaluate_subset_eo(model, subset_dataset, batch_size=args.batch_size, device=device)

    sampled_results = compute_masked_deltas(
        model,
        target_weight,
        subset_dataset,
        sampled_indices,
        ui_flat,
        perf_flat,
        subset_baseline,
        batch_size=args.batch_size,
        device=device,
    )

    sampled_ui = np.array([row["runner_ui"] for row in sampled_results], dtype=np.float64)
    sampled_perf = np.array([row["performance_importance"] for row in sampled_results], dtype=np.float64)
    sampled_delta = np.array([row["delta_eo_abs"] for row in sampled_results], dtype=np.float64)
    sampled_delta_signed = np.array([row["delta_eo_signed"] for row in sampled_results], dtype=np.float64)

    h1 = correlation_summary(sampled_ui, sampled_delta)
    h1_signed = correlation_summary(sampled_ui, sampled_delta_signed)
    h2_all = correlation_summary(ui_flat.astype(np.float64), perf_flat.astype(np.float64))
    h2_sampled = correlation_summary(sampled_ui, sampled_perf)

    os.makedirs(args.results_dir, exist_ok=True)
    prefix = args.plot_prefix or f"{task}_{args.layer}"
    eo_plot_path = os.path.join(args.results_dir, f"runner_ui_vs_eo_{args.layer}.png")
    perf_plot_path = os.path.join(args.results_dir, f"runner_ui_vs_perf_{args.layer}.png")
    decile_plot_path = os.path.join(args.results_dir, f"runner_ui_decile_deltaeo_{args.layer}.png")
    save_scatter(sampled_ui, sampled_delta, "RUNNER UI", "Absolute delta EO", f"RUNNER UI vs absolute delta EO ({args.layer})", eo_plot_path)
    save_scatter(ui_flat.astype(np.float64), perf_flat.astype(np.float64), "RUNNER UI", "Performance importance", f"RUNNER UI vs I_w ({args.layer})", perf_plot_path)
    save_decile_plot(sampled_ui, sampled_delta, decile_plot_path)

    result = {
        "design_phase": 1,
        "task": task,
        "model_name": model_name,
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "epoch": checkpoint.get("epoch"),
        "layer": args.layer,
        "target_weight_name": target_name,
        "target_weight_shape": list(target_weight.shape),
        "full_test_baseline": {
            "mean_bce_loss": full_baseline.mean_bce_loss,
            "average_precision": full_baseline.average_precision,
            "eo_gap": full_baseline.eo_gap,
        },
        "subset_baseline_for_delta_eo": {
            "subset_size": subset_size,
            "mean_bce_loss": subset_baseline.mean_bce_loss,
            "average_precision": subset_baseline.average_precision,
            "eo_gap": subset_baseline.eo_gap,
        },
        "phase1_step12": {
            "runner_ui_num_weights": int(ui_flat.size),
            "performance_importance_num_weights": int(perf_flat.size),
            "mean_test_bce_loss_used_for_I_w": mean_loss,
        },
        "sampling": {
            "top_k": int(min(args.top_k, ui_flat.size)),
            "bottom_k": int(min(args.bottom_k, max(0, ui_flat.size - min(args.top_k, ui_flat.size)))),
            "random_k": int(min(args.random_k, ui_flat.size)),
            "sampled_count": int(len(sampled_results)),
        },
        "correlation_ui_vs_actual_delta_eo": h1,
        "correlation_ui_vs_signed_delta_eo": h1_signed,
        "correlation_ui_vs_performance_importance_all_weights": h2_all,
        "correlation_ui_vs_performance_importance_sampled_weights": h2_sampled,
        "output_files": {
            "runner_ui_vs_eo_plot": eo_plot_path,
            "runner_ui_vs_perf_plot": perf_plot_path,
            "runner_ui_decile_plot": decile_plot_path,
        },
        "sampled_weights": sampled_results,
    }

    json_path = os.path.join(args.results_dir, "correlation_results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps({
        "json": json_path,
        "h1": h1,
        "h1_signed": h1_signed,
        "h2_all": h2_all,
        "h2_sampled": h2_sampled,
    }, indent=2))


if __name__ == "__main__":
    main()