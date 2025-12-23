# yapf:disable
log_config = dict(
    interval=16,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        # dict(type='TensorboardMultiLrHook', by_epoch=False)
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# resume_from = None  # 廃止: MMEngine 0.10.7ではload_from + resume=Trueを使用
workflow = [('train', 1)]
cudnn_benchmark = True
