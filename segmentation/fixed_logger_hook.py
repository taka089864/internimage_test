# Fixed LoggerHook for mmengine 0.10.7 TensorBoard x-axis bug
from mmengine.registry import HOOKS
from mmengine.hooks import LoggerHook
from mmengine.runner.loops import BaseLoop
from typing import Dict, Optional


@HOOKS.register_module(force=True)
class FixedLoggerHook(LoggerHook):
    """Fixed LoggerHook that properly handles validation metric logging for iteration-based training.

    This fixes a bug in mmengine 0.10.7 where validation metrics are logged with x-axis=0
    when log_metric_by_epoch=False in iteration-based training loops.

    Root cause: mmengine's LoggerHook.after_val_epoch() checks if runner._train_loop
    is dict/None and defaults to iter=0, causing all validation points to stack at x=0.

    Fix: Safely retrieve iteration number from message_hub or constructed train_loop,
    avoiding the problematic _train_loop type check while preventing unwanted train_loop
    construction in validation-only workflows.
    """

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Log validation metrics with correct iteration number on x-axis.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)

        if self.log_metric_by_epoch:
            # Epoch-based logging (original behavior)
            if (isinstance(runner._train_loop, dict)
                    or runner._train_loop is None):
                epoch = 0
            else:
                epoch = runner.epoch
            runner.visualizer.add_scalars(
                tag, step=epoch, file_path=self.json_log_path)
        else:
            # Iteration-based logging (FIXED)
            # Strategy 1: Try to get iter from message_hub (doesn't trigger train_loop construction)
            iter_num = 0
            if hasattr(runner, 'message_hub') and runner.message_hub is not None:
                # message_hub stores current iter without constructing train_loop
                iter_num = runner.message_hub.get_info('iter', 0)

            # Strategy 2: If message_hub doesn't have it and train_loop is already constructed
            elif isinstance(runner._train_loop, BaseLoop):
                # Only access runner.iter if train_loop is already a BaseLoop instance
                # This avoids triggering expensive train_loop construction
                iter_num = runner.iter

            # Strategy 3: Fallback to 0 if neither option works (validation-only mode)
            # This is safe and won't cause issues

            runner.visualizer.add_scalars(
                tag, step=iter_num, file_path=self.json_log_path)
