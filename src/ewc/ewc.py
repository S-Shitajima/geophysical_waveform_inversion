import copy

import pytorch_lightning as L
import torch
import torch.nn as nn


class EWC:
    def __init__(
            self,
            model: nn.Module,
            datamodule: L.LightningDataModule,
            loss_fn1: nn.Module,
            loss_fn2: nn.Module,
        ) -> None:

        self.model = model
        datamodule.setup()
        dataloader = datamodule.train_dataloader()
        self.device = model.device
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {n: p.clone().detach() for n, p in self.params.items()}
        self._fisher = self._compute_fisher(dataloader, loss_fn1, loss_fn2)

    def _computr_fisher(self, dataloader, loss_fn1, loss_fn2):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.params.items()}
        self.model.eval()
        for x, y, label in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            label = label.to(self.device)

            self.model.zero_grad()
            reg_logit, clf_logit = self.model(x)
            loss = loss_fn1(reg_logit, y) + loss_fn2(clf_logit, label)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach()**2 / len(dataloader)
        
        self.model.train()
        return fisher

    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self._fisher:
                _loss = self._fisher[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss