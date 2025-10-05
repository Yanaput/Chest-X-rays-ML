from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import ChestXray14Dataset


def build_transforms(img_size: int, mean: float = 0.5, std: float = 0.25) -> Tuple[A.Compose, A.Compose]:
    train_tf = A.Compose([
        # A.LongestMaxSize(max_size=img_size),
        # A.PadIfNeeded(img_size, img_size, border_mode=0),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1), 
            translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)}, 
            rotate=(-10, 10),
            shear=(-5, 5),p=0.7
        ),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.05,
                                   contrast_limit=0.05, p=0.3),
        A.GaussNoise(std_range=(0.02, 0.06), per_channel=False, p=0.15),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.05, 0.12),
            hole_width_range=(0.05, 0.12),
            fill=0,
            p=0.15
        ),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=0),
        A.Normalize(mean=(mean,), std=(std,)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


class ChestXray14DataModule(pl.LightningDataModule):
    """
    Lightning DataModule wrapping NIH14 classification datasets.
    """
    def __init__(
        self,
        train_csv: str | Path,
        val_csv: str | Path,
        test_csv: Optional[str | Path] = None,
        img_size: int = 512,
        batch_size: int = 32,
        num_workers: int = 6,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cached_dir: Optional[str | Path] = None,
        mean: float = 0.5,
        std: float = 0.25,
    ) -> None:
        super().__init__()
        self.train_csv = Path(train_csv)
        self.val_csv   = Path(val_csv)
        self.test_csv  = Path(test_csv) if test_csv else None
        self.img_size  = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.cached_dir = Path(cached_dir) if cached_dir else None
        self.mean, self.std = mean, std

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self) -> None:
        # rank-0 only ops (no GPU). Keep it light: file existence checks, etc.
        assert self.train_csv.exists(), f"Missing train CSV: {self.train_csv}"
        assert self.val_csv.exists(),   f"Missing val CSV: {self.val_csv}"
        if self.test_csv is not None:
            assert self.test_csv.exists(), f"Missing test CSV: {self.test_csv}"

    def setup(self, stage: Optional[str] = None) -> None:
        train_tf, val_tf = build_transforms(self.img_size, self.mean, self.std)

        if stage in (None, "fit"):
            self.train_ds = ChestXray14Dataset(self.train_csv, transform=train_tf, use_cached_dir=self.cached_dir)
            self.val_ds   = ChestXray14Dataset(self.val_csv,   transform=val_tf,   use_cached_dir=self.cached_dir)

        if stage in (None, "test") and self.test_csv is not None:
            self.test_ds  = ChestXray14Dataset(self.test_csv,  transform=val_tf,   use_cached_dir=self.cached_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
