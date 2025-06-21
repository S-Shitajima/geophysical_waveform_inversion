import os
from pathlib import Path
import random
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from util.log_transform import log_transform_torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def explode_paths(df: pd.DataFrame):
    df.loc[:, "path"] = df["path"].apply(lambda x: [f"{x}_{i}" for i in range(500)])
    df = df.explode("path").reset_index(drop=True)
    df.loc[:, "path"] = df["path"] + ".npz"
    return df


class MyDataset(Dataset):
    def __init__(
            self,
            paths: pd.DataFrame,
            height: int,
            width: int,
            mean_x: torch.Tensor,
            std_x: torch.Tensor,
            do_transform: bool = False
        ) -> None:
        self.paths = paths
        self.height = height
        self.width = width
        self.mean_x = mean_x
        self.std_x = std_x
        self.do_transform = do_transform
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        _, label, path = tuple(self.paths.iloc[index])
        images = np.load(path)
        x = images["x"] # (5, 1000, 70)
        x = torch.from_numpy(x)
        # x = F.pad(x, pad=(1, 1, 76, 76), mode="constant")
        x = log_transform_torch(x)
        x = (x - self.mean_x) / self.std_x
        x = x.unsqueeze(dim=0) # (1, 5, 1000, 70)
        x = F.interpolate(x, size=(self.height, self.width), mode="bicubic")
        x = x.squeeze(dim=0) # (5, new_h, new_w)
        x = x.float()

        y = images["y"] # (1, 70, 70)
        y = torch.from_numpy(y)
        y = y.float()

        if self.do_transform:
            x, y = self._random_hflip(x, y)

        return x, y, label, path.as_posix()
    
    def _random_hflip(
            self,
            image1: torch.Tensor,
            image2: torch.Tensor,
            p: float = 0.5
        ) -> tuple[torch.Tensor, torch.Tensor]:

        if torch.rand(()) < p:
            return image1.flip(dims=[0, -1]), image2.flip(dims=[-1])
        else:
            return image1, image2
        

class MyDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_paths: pd.DataFrame,
            valid_paths: pd.DataFrame,
            test_paths: pd.DataFrame,
            seed: int,
            batch_size: int,
            height: int,
            width: int,
            mean_x: torch.Tensor,
            std_x: torch.Tensor,
            do_transform: bool = False,
        ) -> None:

        super().__init__()
        self.train_paths = train_paths
        self.valid_paths = valid_paths
        self.test_paths = test_paths
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.mean_x = mean_x
        self.std_x = std_x
        self.do_transform = do_transform
        self.generator = torch.Generator().manual_seed(seed)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = MyDataset(
                paths=self.train_paths,
                height=self.height,
                width=self.width,
                mean_x=self.mean_x,
                std_x=self.std_x,
                do_transform=self.do_transform,
            )
            self.valid_dataset = MyDataset(
                paths=self.valid_paths,
                height=self.height,
                width=self.width,
                mean_x=self.mean_x,
                std_x=self.std_x,
                do_transform=False,
            )
        elif stage == "test":
            self.test_dataset = MyDataset(
                paths=self.test_paths,
                height=self.height,
                width=self.width,
                mean_x=self.mean_x,
                std_x=self.std_x,
                do_transform=False,
            )
        else:
            pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=int(os.cpu_count()*2/3),
            worker_init_fn=seed_worker,
            generator=self.generator,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()//6,
            generator=self.generator,
            pin_memory=False,
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()//6,
            generator=self.generator,
            pin_memory=False,
        )