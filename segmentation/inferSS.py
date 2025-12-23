# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import mmcv

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmcv.runner import load_checkpoint
from mmseg.core import get_classes
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
    parser.add_argument('img', help='Image file or a directory contains images')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', type=str, default="demo", help='out dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='ade20k',
        choices=['ade20k', 'cityscapes', 'cocostuff'],
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, checkpoint=None, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = get_classes(args.palette)

    # print(model.CLASSES)
    # print(get_palette(args.palette))

    # checkpoint['meta'] = {}
    # checkpoint['meta']['CLASSES'] = model.CLASSES
    # with open(os.path.join(args.out,"test.pth"), 'wb') as f:
    #     torch.save(checkpoint, f)
    #     f.flush()
    # sys.exit()
        
        
    # check arg.img is directory of a single image.
    time_length = 0
    img_cnt = 0
    if osp.isdir(args.img):
        for img in os.listdir(args.img):
            ext = os.path.splitext(os.path.basename(img))[-1]
            if not (ext in [".jpg",".png"]):
                continue
            
            stime = time.perf_counter()

            test_single_image(model, osp.join(args.img, img), args.out)
            
            etime = time.perf_counter()
            time_length += etime -stime
            img_cnt += 1
    else:
        stime = time.perf_counter()

        test_single_image(model, args.img, args.out)

        etime = time.perf_counter()
        time_length = etime - stime
        imt_cnt = 1

    print("Calc Time:{0} (ave:{1})".format(time_length,time_length/float(img_cnt)))

if __name__ == '__main__':
    main()