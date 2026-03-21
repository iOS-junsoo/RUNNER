from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


CELEBA_SENSITIVE_INDEX = 20
CELEBA_TASK_INDICES = {
    "attractive": 2,
    "wavy_hair": 33,
}


@dataclass(frozen=True)
class CelebAMetadata:
    task: str
    task_index: int
    sensitive_index: int = CELEBA_SENSITIVE_INDEX


def build_celeba_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class CelebAFairnessDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        task: str,
        transform: transforms.Compose | None = None,
        download: bool = False,
    ) -> None:
        if task not in CELEBA_TASK_INDICES:
            raise ValueError(f"Unsupported CelebA task: {task}")

        self.metadata = CelebAMetadata(task=task, task_index=CELEBA_TASK_INDICES[task])
        self.root = Path(root)
        self.base_dir = self.root / "celeba"
        self.image_dir = self.base_dir / "img_align_celeba"
        self.transform = transform

        if download:
            print("download=True was requested, but the local CelebA loader uses existing files under Data/celeba.")

        self._validate_layout()

        attr_names, attr_map = self._load_attr_map()
        split_filenames = self._load_split_filenames(split)

        self.filenames: List[str] = [name for name in split_filenames if name in attr_map]
        if not self.filenames:
            raise RuntimeError(f"No CelebA samples found for split '{split}' in {self.base_dir}")

        attrs = torch.tensor([attr_map[name] for name in self.filenames], dtype=torch.int64)
        self.attr_names = attr_names
        self.labels = attrs[:, self.metadata.task_index].clone()
        self.sensitive = attrs[:, self.metadata.sensitive_index].clone()

        self.indices = np.arange(len(self.filenames), dtype=np.int64)
        self.sensitive_indices = {
            0: self.indices[self.sensitive.numpy() == 0],
            1: self.indices[self.sensitive.numpy() == 1],
        }
        self.group_indices: Dict[Tuple[int, int], np.ndarray] = {}
        for sensitive_value in (0, 1):
            for label_value in (0, 1):
                mask = (self.sensitive.numpy() == sensitive_value) & (self.labels.numpy() == label_value)
                self.group_indices[(sensitive_value, label_value)] = self.indices[mask]

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        image_path = self.image_dir / self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index].float()
        sensitive = self.sensitive[index].long()
        return image, label, sensitive

    def _validate_layout(self) -> None:
        required_files = [
            self.base_dir / "list_attr_celeba.txt",
            self.base_dir / "list_eval_partition.txt",
        ]
        missing = [str(path) for path in required_files if not path.exists()]
        if not self.image_dir.is_dir():
            missing.append(str(self.image_dir))
        if missing:
            raise RuntimeError("Missing required CelebA files: " + ", ".join(missing))

    def _load_attr_map(self) -> Tuple[List[str], Dict[str, List[int]]]:
        attr_path = self.base_dir / "list_attr_celeba.txt"
        with attr_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]

        attr_names = lines[1].split()
        attr_map: Dict[str, List[int]] = {}
        for line in lines[2:]:
            parts = line.split()
            filename = parts[0]
            values = [(int(value) + 1) // 2 for value in parts[1:]]
            attr_map[filename] = values
        return attr_names, attr_map

    def _load_split_filenames(self, split: str) -> List[str]:
        split_map = {
            "train": "0",
            "valid": "1",
            "test": "2",
            "all": None,
        }
        if split not in split_map:
            raise ValueError(f"Unsupported CelebA split: {split}")

        target_partition = split_map[split]
        filenames: List[str] = []
        partition_path = self.base_dir / "list_eval_partition.txt"
        with partition_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                filename, partition = parts
                if target_partition is None or partition == target_partition:
                    filenames.append(filename)
        return filenames


def build_celeba_datasets(
    root: str,
    task: str,
    image_size: int = 224,
    download: bool = False,
):
    transform = build_celeba_transform(image_size=image_size)
    train_dataset = CelebAFairnessDataset(root, "train", task, transform=transform, download=download)
    val_dataset = CelebAFairnessDataset(root, "valid", task, transform=transform, download=False)
    test_dataset = CelebAFairnessDataset(root, "test", task, transform=transform, download=False)
    return train_dataset, val_dataset, test_dataset