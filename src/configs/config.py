from pathlib import Path
import random

import numpy as np
import torch


class CFG:
    def __init__(
            self,
            output_dir: Path,
            model_name: str,
            pretrained: bool,
            debag: bool,
            train_ratio: float,
            seed: int,
            height: int,
            width: int,
            batch_size: int,
            epochs: int,
            learning_rate: float,
            patience: int,
            accumulation_steps: int,
            do_transform: bool,
        ) -> None:

        self.output_dir = output_dir
        self.model_name = model_name
        self.pretrained = pretrained
        self.debag = debag
        self.train_ratio = train_ratio
        self.seed = seed
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.accumulation_steps = accumulation_steps
        self.do_transform = do_transform
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True, parents=True)
        assert 0.0 < train_ratio < 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def seed_everything(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False