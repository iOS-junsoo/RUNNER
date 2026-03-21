from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from numpy.random import beta
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, RandomSampler, Subset


def sample_indices(indices: np.ndarray, batch_size: int) -> np.ndarray:
    replace = len(indices) < batch_size
    return np.random.choice(indices, size=batch_size, replace=replace)


def sample_random_indices(dataset_size: int, batch_size: int) -> np.ndarray:
    replace = dataset_size < batch_size
    return np.random.choice(np.arange(dataset_size), size=batch_size, replace=replace)


def fetch_batch(dataset, indices: Iterable[int], device: torch.device):
    images = []
    labels = []
    sensitive = []
    for index in indices:
        image, label, group = dataset[int(index)]
        images.append(image)
        labels.append(float(label))
        sensitive.append(int(group))

    batch_x = torch.stack(images, dim=0).to(device, non_blocking=True)
    batch_y = torch.tensor(labels, dtype=torch.float32, device=device).view(-1, 1)
    batch_a = torch.tensor(sensitive, dtype=torch.long, device=device)
    return batch_x, batch_y, batch_a


def sample_sensitive_batch(dataset, sensitive_value: int, batch_size: int, device: torch.device):
    return fetch_batch(dataset, sample_indices(dataset.sensitive_indices[sensitive_value], batch_size), device)


def sample_sensitive_label_batch(
    dataset,
    sensitive_value: int,
    label_value: int,
    batch_size: int,
    device: torch.device,
):
    return fetch_batch(dataset, sample_indices(dataset.group_indices[(sensitive_value, label_value)], batch_size), device)


def sample_random_batch(dataset, batch_size: int, device: torch.device):
    return fetch_batch(dataset, sample_random_indices(len(dataset), batch_size), device)


def _loader_kwargs(num_workers: int, pin_memory: bool):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def make_train_loader(
    dataset,
    batch_size: int,
    steps_per_epoch: int,
    num_workers: int,
    pin_memory: bool,
    indices: np.ndarray | None = None,
):
    source_dataset = dataset if indices is None else Subset(dataset, indices.tolist())
    sampler = RandomSampler(source_dataset, replacement=True, num_samples=batch_size * steps_per_epoch)
    return DataLoader(
        source_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        **_loader_kwargs(num_workers, pin_memory),
    )


def _move_batch(batch, device: torch.device):
    images, labels, sensitive = batch
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).view(-1, 1).float()
    sensitive = sensitive.to(device, non_blocking=True)
    return images, labels, sensitive


def _amp_context(device: torch.device, use_amp: bool):
    enabled = use_amp and device.type == "cuda"
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.float16)


def _topk_ratio(size: int, neuron_ratio: float) -> int:
    return max(1, int(math.ceil(size * neuron_ratio)))


def _activation_distance(
    activations_a: Sequence[torch.Tensor],
    activations_b: Sequence[torch.Tensor],
    important_indices: Sequence[torch.Tensor],
) -> torch.Tensor:
    loss = torch.zeros((), device=activations_a[0].device)
    for act_a, act_b, index in zip(activations_a, activations_b, important_indices):
        loss = loss + torch.abs(act_a[index] - act_b[index]).mean()
    return loss


def cal_importance(model, loss: torch.Tensor, neuron_ratio: float):
    weights = model.get_runner_weights()
    grads = torch.autograd.grad(loss, weights, retain_graph=False, allow_unused=True)

    important_indices = []
    important_values = []
    for weight, grad in zip(weights, grads):
        if grad is None:
            criteria = torch.zeros(weight.shape[0], device=weight.device)
        else:
            criteria = (weight * grad).detach().pow(2).view(weight.shape[0], -1).sum(dim=1)
        topk = _topk_ratio(criteria.numel(), neuron_ratio)
        values, indices = criteria.topk(topk, dim=0, largest=True)
        important_indices.append(indices)
        important_values.append(values)

    return important_indices, important_values


def cal_importance_gapreg_dp(model, batch_x_0: torch.Tensor, batch_x_1: torch.Tensor, neuron_ratio: float):
    output_0, _ = model(batch_x_0)
    output_1, _ = model(batch_x_1)
    loss_reg = torch.abs(output_0.mean() - output_1.mean())
    return cal_importance(model, loss_reg, neuron_ratio)


def cal_importance_gapreg_eo(
    model,
    batch_pairs_0: Sequence[torch.Tensor],
    batch_pairs_1: Sequence[torch.Tensor],
    neuron_ratio: float,
):
    loss_reg = torch.zeros((), device=batch_pairs_0[0].device)
    for batch_x_0, batch_x_1 in zip(batch_pairs_0, batch_pairs_1):
        output_0, _ = model(batch_x_0)
        output_1, _ = model(batch_x_1)
        loss_reg = loss_reg + torch.abs(output_0.mean() - output_1.mean())
    return cal_importance(model, loss_reg, neuron_ratio)


def train_dp(
    model,
    criterion,
    optimizer,
    train_dataset,
    method: str,
    lam: float,
    neuron_ratio: float,
    batch_size: int = 64,
    steps_per_epoch: int | None = None,
    device: torch.device | None = None,
    num_workers: int = 0,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    logger=None,
    log_prefix: str = "",
    log_interval: int = 10,
):
    model.train()
    device = device or next(model.parameters()).device
    steps = steps_per_epoch or max(1, len(train_dataset) // (2 * batch_size))
    pin_memory = device.type == "cuda"
    loader_sensitive_0 = make_train_loader(
        train_dataset,
        batch_size=batch_size,
        steps_per_epoch=steps,
        num_workers=num_workers,
        pin_memory=pin_memory,
        indices=train_dataset.sensitive_indices[0],
    )
    loader_sensitive_1 = make_train_loader(
        train_dataset,
        batch_size=batch_size,
        steps_per_epoch=steps,
        num_workers=num_workers,
        pin_memory=pin_memory,
        indices=train_dataset.sensitive_indices[1],
    )
    loader_random = None
    if method == "van":
        loader_random = make_train_loader(
            train_dataset,
            batch_size=2 * batch_size,
            steps_per_epoch=steps,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    loss_reg_total = 0.0
    loss_sup_total = 0.0
    iterator_sensitive_0 = iter(loader_sensitive_0)
    iterator_sensitive_1 = iter(loader_sensitive_1)
    iterator_random = iter(loader_random) if loader_random is not None else None
    for step_idx in range(steps):
        batch_x_0, batch_y_0, _ = _move_batch(next(iterator_sensitive_0), device)
        batch_x_1, batch_y_1, _ = _move_batch(next(iterator_sensitive_1), device)

        with _amp_context(device, use_amp):
            if method == "mixup":
                gamma = beta(1, 1)
                batch_x_mix = (batch_x_0 * gamma + batch_x_1 * (1 - gamma)).requires_grad_(True)
                output_mix, _ = model(batch_x_mix)
                gradx = torch.autograd.grad(output_mix.sum(), batch_x_mix, create_graph=True)[0]
                batch_x_delta = batch_x_1 - batch_x_0
                loss_reg = torch.abs((gradx * batch_x_delta).flatten(1).sum(1).mean())
                batch_x = torch.cat((batch_x_0, batch_x_1), dim=0)
                batch_y = torch.cat((batch_y_0, batch_y_1), dim=0)
            elif method == "GapReg":
                output_0, _ = model(batch_x_0)
                output_1, _ = model(batch_x_1)
                loss_reg = torch.abs(output_0.mean() - output_1.mean())
                batch_x = torch.cat((batch_x_0, batch_x_1), dim=0)
                batch_y = torch.cat((batch_y_0, batch_y_1), dim=0)
            elif method == "NeuronImportance_GapReg":
                important_indices, _ = cal_importance_gapreg_dp(model, batch_x_0, batch_x_1, neuron_ratio)
                output_0, activations_0 = model(batch_x_0)
                output_1, activations_1 = model(batch_x_1)
                loss_reg = _activation_distance(activations_0, activations_1, important_indices)
                batch_x = torch.cat((batch_x_0, batch_x_1), dim=0)
                batch_y = torch.cat((batch_y_0, batch_y_1), dim=0)
            else:
                loss_reg = torch.zeros((), device=device)
                if method == "van":
                    batch_x, batch_y, _ = _move_batch(next(iterator_random), device)
                else:
                    batch_x = torch.cat((batch_x_0, batch_x_1), dim=0)
                    batch_y = torch.cat((batch_y_0, batch_y_1), dim=0)

            output, _ = model(batch_x)
        loss_sup = criterion(output.float(), batch_y)
        loss = loss_sup + lam * loss_reg

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_reg_total += float(loss_reg.detach().item())
        loss_sup_total += float(loss_sup.detach().item())

        if logger is not None and ((step_idx + 1) % max(1, log_interval) == 0 or step_idx + 1 == steps):
            logger(
                f"{log_prefix}step {step_idx + 1}/{steps} "
                f"loss_sup={loss_sup.detach().item():.6f} "
                f"loss_reg={loss_reg.detach().item():.6f}"
            )

    return {
        "loss_reg": loss_reg_total / steps,
        "loss_sup": loss_sup_total / steps,
    }


def train_eo(
    model,
    criterion,
    optimizer,
    train_dataset,
    method: str,
    lam: float,
    neuron_ratio: float,
    batch_size: int = 128,
    steps_per_epoch: int | None = None,
    device: torch.device | None = None,
    num_workers: int = 0,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    logger=None,
    log_prefix: str = "",
    log_interval: int = 10,
):
    model.train()
    device = device or next(model.parameters()).device
    steps = steps_per_epoch or max(1, len(train_dataset) // (4 * batch_size))
    pin_memory = device.type == "cuda"
    group_loaders = {
        key: make_train_loader(
            train_dataset,
            batch_size=batch_size,
            steps_per_epoch=steps,
            num_workers=num_workers,
            pin_memory=pin_memory,
            indices=train_dataset.group_indices[key],
        )
        for key in ((0, 0), (0, 1), (1, 0), (1, 1))
    }
    loader_random = None
    if method == "van":
        loader_random = make_train_loader(
            train_dataset,
            batch_size=4 * batch_size,
            steps_per_epoch=steps,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    scaler = scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    loss_reg_total = 0.0
    loss_sup_total = 0.0
    group_iterators = {key: iter(loader) for key, loader in group_loaders.items()}
    iterator_random = iter(loader_random) if loader_random is not None else None
    for step_idx in range(steps):
        batch_x_00, batch_y_00, _ = _move_batch(next(group_iterators[(0, 0)]), device)
        batch_x_01, batch_y_01, _ = _move_batch(next(group_iterators[(0, 1)]), device)
        batch_x_10, batch_y_10, _ = _move_batch(next(group_iterators[(1, 0)]), device)
        batch_x_11, batch_y_11, _ = _move_batch(next(group_iterators[(1, 1)]), device)

        batch_pairs_0 = [batch_x_00, batch_x_01]
        batch_pairs_1 = [batch_x_10, batch_x_11]

        with _amp_context(device, use_amp):
            if method == "mixup":
                loss_reg = torch.zeros((), device=device)
                for batch_x_0, batch_x_1 in zip(batch_pairs_0, batch_pairs_1):
                    gamma = beta(1, 1)
                    batch_x_mix = (batch_x_0 * gamma + batch_x_1 * (1 - gamma)).requires_grad_(True)
                    output_mix, _ = model(batch_x_mix)
                    gradx = torch.autograd.grad(output_mix.sum(), batch_x_mix, create_graph=True)[0]
                    batch_x_delta = batch_x_1 - batch_x_0
                    loss_reg = loss_reg + torch.abs((gradx * batch_x_delta).flatten(1).sum(1).mean())
            elif method == "GapReg":
                loss_reg = torch.zeros((), device=device)
                for batch_x_0, batch_x_1 in zip(batch_pairs_0, batch_pairs_1):
                    output_0, _ = model(batch_x_0)
                    output_1, _ = model(batch_x_1)
                    loss_reg = loss_reg + torch.abs(output_0.mean() - output_1.mean())
            elif method == "NeuronImportance_GapReg":
                important_indices, _ = cal_importance_gapreg_eo(model, batch_pairs_0, batch_pairs_1, neuron_ratio)
                loss_reg = torch.zeros((), device=device)
                for batch_x_0, batch_x_1 in zip(batch_pairs_0, batch_pairs_1):
                    _, activations_0 = model(batch_x_0)
                    _, activations_1 = model(batch_x_1)
                    loss_reg = loss_reg + _activation_distance(activations_0, activations_1, important_indices)
            else:
                loss_reg = torch.zeros((), device=device)

            if method == "van":
                batch_x, batch_y, _ = _move_batch(next(iterator_random), device)
            else:
                batch_x = torch.cat((batch_x_00, batch_x_01, batch_x_10, batch_x_11), dim=0)
                batch_y = torch.cat((batch_y_00, batch_y_01, batch_y_10, batch_y_11), dim=0)

            output, _ = model(batch_x)
        loss_sup = criterion(output.float(), batch_y)
        loss = loss_sup + lam * loss_reg

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_reg_total += float(loss_reg.detach().item())
        loss_sup_total += float(loss_sup.detach().item())

        if logger is not None and ((step_idx + 1) % max(1, log_interval) == 0 or step_idx + 1 == steps):
            logger(
                f"{log_prefix}step {step_idx + 1}/{steps} "
                f"loss_sup={loss_sup.detach().item():.6f} "
                f"loss_reg={loss_reg.detach().item():.6f}"
            )

    return {
        "loss_reg": loss_reg_total / steps,
        "loss_sup": loss_sup_total / steps,
    }


def make_eval_loader(dataset, batch_size: int, num_workers: int, pin_memory: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(num_workers, pin_memory),
    )


def collect_predictions(
    model,
    dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    device: torch.device | None = None,
    use_amp: bool = False,
):
    model.eval()
    device = device or next(model.parameters()).device
    loader = make_eval_loader(dataset, batch_size, num_workers, pin_memory=device.type == "cuda")

    y_true = []
    y_score = []
    sensitive = []
    with torch.no_grad():
        for images, labels, groups in loader:
            images = images.to(device, non_blocking=True)
            with _amp_context(device, use_amp):
                outputs, _ = model(images)
            y_true.append(labels.numpy())
            y_score.append(outputs.squeeze(1).cpu().numpy())
            sensitive.append(groups.numpy())

    y_true = np.concatenate(y_true, axis=0).astype(np.int64)
    y_score = np.concatenate(y_score, axis=0)
    sensitive = np.concatenate(sensitive, axis=0).astype(np.int64)
    return y_true, y_score, sensitive


def demographic_parity_gap(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    mask0 = sensitive == 0
    mask1 = sensitive == 1
    rate0 = y_pred[mask0].mean() if mask0.any() else 0.0
    rate1 = y_pred[mask1].mean() if mask1.any() else 0.0
    return float(abs(rate0 - rate1))


def _safe_rate(numerator_mask: np.ndarray, denominator_mask: np.ndarray) -> float:
    denominator = denominator_mask.sum()
    if denominator == 0:
        return 0.0
    return float(numerator_mask.sum() / denominator)


def equalized_odds_gap_sum(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    mask0 = sensitive == 0
    mask1 = sensitive == 1

    tpr0 = _safe_rate((y_pred[mask0] == 1) & (y_true[mask0] == 1), y_true[mask0] == 1)
    tpr1 = _safe_rate((y_pred[mask1] == 1) & (y_true[mask1] == 1), y_true[mask1] == 1)
    fpr0 = _safe_rate((y_pred[mask0] == 1) & (y_true[mask0] == 0), y_true[mask0] == 0)
    fpr1 = _safe_rate((y_pred[mask1] == 1) & (y_true[mask1] == 0), y_true[mask1] == 0)
    return float(abs(tpr0 - tpr1) + abs(fpr0 - fpr1))


def evaluate_dp(
    model,
    dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    device: torch.device | None = None,
    use_amp: bool = False,
):
    y_true, y_score, sensitive = collect_predictions(
        model,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_amp=use_amp,
    )
    y_pred = (y_score >= 0.5).astype(np.int64)
    ap = average_precision_score(y_true, y_score)
    gap = demographic_parity_gap(y_pred, sensitive)
    return float(ap), gap


def evaluate_eo(
    model,
    dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    device: torch.device | None = None,
    use_amp: bool = False,
):
    y_true, y_score, sensitive = collect_predictions(
        model,
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        use_amp=use_amp,
    )
    y_pred = (y_score >= 0.5).astype(np.int64)
    ap = average_precision_score(y_true, y_score)
    gap = equalized_odds_gap_sum(y_true, y_pred, sensitive)
    return float(ap), gap