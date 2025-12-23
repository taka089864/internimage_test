import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import mmengine

# from mmcv.utils import Config, DictAction, get_git_hash
from mmengine.config import Config, DictAction
from mmengine.utils import get_git_hash

from mmseg import __version__
from mmengine.runner import Runner
from mmengine.logging import MMLogger
import mmcv_custom  # noqa: F401
import mmseg_custom  # noqa: F401

# --- 追加: カスタムフックをレジストリへ登録 -------------------------
# この import により tb_multi_lr_hook.py が実行され，
# TensorboardMultiLrHook が HOOKS レジストリに登録される
import tb_multi_lr_hook  # noqa: F401
from mmseg_custom.hooks import clear_cuda_cache_hook  # noqa: F401
from mmseg_custom.hooks import iter_logger_hook  # noqa: F401
# ---------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description='Train a single head segmentor for teeth caries')
    parser.add_argument('config', help='train config file')
    parser.add_argument('--work-dir', help='where to save logs/models')
    parser.add_argument('--load-from', help='init checkpoint')
    parser.add_argument('--resume-from', help='resume checkpoint')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--no-validate', action='store_true')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    # merge options from command line
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        # MMEngine 0.10.7の新しいAPI: resume_from → load_from + resume=True
        cfg.load_from = args.resume_from
        cfg.resume = True

    cfg.gpu_ids = [args.gpu_id]

    # ------------------- 追加: device を自動設定 ------------------- #
    # mmsegmentation >=0.30 では cfg.device が必須
    if not hasattr(cfg, 'device'):
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ------------------------------------------------------------- #

    mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # NVidia Ampere 以降用（高速化）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ---------------- Training (mmengine Runner) ---------------- #
    # Build runner from config and start training
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
