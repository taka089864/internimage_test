# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
from mmseg_custom.datasets import BldgSSDataset
import cv2
import os.path as osp
import os,sys
import numpy as np
import torch

from mmdeploy.apis import torch2onnx

import time


def test_single_image(model, img_name, out_dir):
    result = inference_segmentor(model, img_name)[0]

    fnelem = os.path.splitext(os.path.basename(img_name))
    oname = ".".join(fnelem[:-1]) + ".png"
    
    # save the results
    mmcv.mkdir_or_exist(out_dir)
    out_path = osp.join(out_dir, oname)
    cv2.imwrite(out_path, result.astype(np.uint8))
    print(f"Result is save at {out_path}")

    # rmask = (result == 0)
    # smask = (result == 1)
    # bmask = (result == 2)
    # oimg = np.zeros_like(result)
    # oimg[rmask] = 64
    # oimg[smask] = 128
    # oimg[bmask] = 255

    # oname = ".".join(fnelem[:-1]) + "_check.jpg"
    # cv2.imwrite(os.path.join(out_dir,oname),oimg)



def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--img_dir',type=str,required=True)
    parser.add_argument('--out_dir', type=str, default="demo", help='out dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    model.CLASSES = BldgSSDataset.CLASSES
    model.PALETTE = BldgSSDataset.PALETTE

    img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    # test_pipeline = [dict(type='LoadImageFromFile'),
    #                  dict(type='LoadBldgSSAnnotations'),
    #                  dict(type='MultiScaleFlipAug',img_scale=(512,512),flip=False,
    #                         transforms=[dict(type='Resize', keep_ratio=True), dict(type='RandomFlip',prob=0.0),
    #                                     dict(type='Normalize', **img_norm_cfg),dict(type='ImageToTensor', keys=['img']),
    #                                     dict(type='Collect', keys=['img']),
    #                 ])]
    test_pipeline = [dict(type='LoadImageFromFile'),
                     dict(type='LoadBldgSSAnnotations',only_img=True)]

    dataset = BldgSSDataset(test_pipeline, args.img_dir,no_annot=True,test_mode=True)

    os.makedirs(args.out_dir,exist_ok=True)

    model.eval()

    time_length = 0
    img_cnt = 0
    for i in range(len(dataset)):
        datas = dataset.__getitem__(i)

        stime = time.perf_counter()

        ipath = datas['img_info']['filename']
        img = datas['img']
        result = inference_segmentor(model, img)[0]

        etime = time.perf_counter()
        time_length += etime -stime
        img_cnt += 1

        rect = datas['crop_rect']
        img = img[rect[1]:rect[3],rect[0]:rect[2]]
        result = result[rect[1]:rect[3],rect[0]:rect[2]]

        oimg = np.zeros_like(img)
        for cid in range(len(model.PALETTE)):
            cmask = (result == cid)
            oimg[cmask] = model.PALETTE[cid]
        
        bname, ext = os.path.splitext(os.path.basename(ipath))
        ipath = os.path.join(args.out_dir,"{0}_img.jpg".format(bname))
        ppath = os.path.join(args.out_dir,"{0}.png".format(bname))
        cv2.imwrite(ipath,img)
        cv2.imwrite(ppath,oimg.astype(np.uint8))


    print("Calc Time:{0:.03f} (ave:{1:.03f})".format(time_length,time_length/float(img_cnt)))


if __name__ == '__main__':
    main()