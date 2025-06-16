from copy import deepcopy
import torch
import pytorch_lightning as L


class EMACallback(L.Callback):
    def __init__(self, decay=0.9999, ema_device=None):
        super().__init__()
        self.decay = decay
        self.ema_state = {}
        self.device = ema_device

    def state_key(self):
        key = self.__class__.__qualname__
        print(f"Using state_key: {key}")
        return key
    
    def on_train_start(self, trainer, module):
        for k, v in module.state_dict().items():
            self.ema_state[k] = v.clone().to(self.device) if self.device else v.clone()

    def on_train_batch_end(self, trainer, module, *args):
        for k, v in module.state_dict().items():
            self.ema_state[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def on_validation_start(self, trainer, module):
        self.backup = deepcopy(module.state_dict())
        module.load_state_dict(self.ema_state, strict=False)

    def on_validation_end(self, trainer, module):
        module.load_state_dict(self.backup, strict=False)

    def on_test_start(self, trainer, module):
        self.backup = deepcopy(module.state_dict())
        module.load_state_dict(self.ema_state, strict=False)

    def on_test_end(self, trainer, module):
        module.load_state_dict(self.backup, strict=False)

    def on_save_checkpoint(self, trainer, module, checkpoint):
        return {'ema_state': self.ema_state}

    def on_load_checkpoint(self, trainer, module, callback_state):
        self.ema_state = callback_state['ema_state']