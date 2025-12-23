from mmengine.hooks import LoggerHook
from mmengine.registry import HOOKS
from mmengine.runner.base_loop import BaseLoop


@HOOKS.register_module()
class IterLoggerHook(LoggerHook):
    """Logger hook that always uses runner.iter for validation scalars."""

    def after_val_epoch(self, runner, metrics=None):
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)

        if self.log_metric_by_epoch:
            epoch = getattr(runner, 'epoch', 0)
            runner.visualizer.add_scalars(
                tag, step=epoch, file_path=self.json_log_path)
        else:
            iter_step = None
            train_loop = getattr(runner, 'train_loop', None)
            if isinstance(train_loop, BaseLoop):
                iter_step = train_loop.iter
            if iter_step is None and hasattr(runner, 'iter'):
                iter_step = runner.iter
            if iter_step is None and hasattr(runner, 'message_hub'):
                iter_step = runner.message_hub.get_info('iter', 0)
            if iter_step is None:
                iter_step = 0
            runner.logger.info(
                f'IterLoggerHook: logging validation metrics at iter {iter_step}')
            runner.visualizer.add_scalars(
                tag, step=iter_step, file_path=self.json_log_path)
