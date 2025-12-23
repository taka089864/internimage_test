from collections import defaultdict

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class TensorboardMultiLrHook(Hook):
    """Log mean LR for backbone/head parameter groups to TensorBoard."""

    def __init__(self, interval: int = 1) -> None:
        self.interval = interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter % self.interval != 0:
            return

        optim = runner.optim_wrapper.optimizer if hasattr(runner, 'optim_wrapper') else runner.optimizer
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        p2n = {p: n for n, p in model.named_parameters()}

        lr_buf = defaultdict(list)
        for group in optim.param_groups:
            if not group['params']:
                continue
            any_param = group['params'][0]
            name = p2n.get(any_param, '')
            tag = 'backbone' if name.startswith('backbone') else 'head'
            lr_buf[tag].append(group['lr'])

        if not lr_buf:
            return

        for tag, lrs in lr_buf.items():
            mean_lr = sum(lrs) / len(lrs)
            runner.visualizer.add_scalar(f'lr/{tag}', mean_lr, step=runner.iter)
