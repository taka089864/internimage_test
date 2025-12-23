import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ClearCUDACacheHook(Hook):
    """Validation 後に CUDA キャッシュを解放するフック。"""

    def after_val_epoch(self, runner, metrics=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
