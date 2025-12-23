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
    parser.add_argument('--data_list',type=str,required=True)
    parser.add_argument('--base_path',type=str,required=True)
    parser.add_argument('--out_dir', type=str, default="demo", help='out dir')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--do_eval',action='store_true',default=False)
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
                     dict(type='LoadBldgSSAnnotations')]

    dataset = BldgSSDataset(test_pipeline, args.data_list,
                                data_root=args.base_path)

    os.makedirs(args.out_dir,exist_ok=True)

    model.eval()


    catN = len(model.PALETTE)
    cMat = np.zeros((catN,catN))
    encode_value = catN + 1

    time_length = 0
    img_cnt = 0
    for i in range(len(dataset)):
        datas = dataset.__getitem__(i)

        stime = time.perf_counter()

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
        
        ipath = os.path.join(args.out_dir,"{0:03d}_img.jpg".format(i))
        ppath = os.path.join(args.out_dir,"{0:03d}_pred.png".format(i))
        cv2.imwrite(ipath,img)
        cv2.imwrite(ppath,oimg.astype(np.uint8))

        if not args.do_eval:
            continue
        
        gt = datas['gt_semantic_seg']
        gt = gt[rect[1]:rect[3],rect[0]:rect[2]]
        pd_idx = result
        gt_idx = np.argmax(gt,axis=-1)

        encoded =gt_idx*encode_value + pd_idx
        values, cnt = np.unique(encoded,return_counts=True)

        for v,c in zip(values,cnt):
            pid = v%encode_value
            gid = int((v-pid)/encode_value)
            cMat[pid][gid] += c


    print("Calc Time:{0:.03f} (ave:{1:.03f})".format(time_length,time_length/float(img_cnt)))


    if args.do_eval:
        iou_list = []
        p_list = []
        r_list = []
        d_list = []
        for i in range(catN):
            tp = np.longlong(cMat[i][i])
            fn = np.longlong(cMat[:,i].sum()) - tp
            fp = np.longlong(cMat[i,:].sum()) - tp

            # iou
            denom = tp + fn + fp
            if denom == 0:
                iou_list.append(float('nan'))
            else:
                iou_list.append(float(tp)/denom)

            # precision
            denom = tp + fp
            if denom == 0:
                p_list.append(float('nan'))
            else:
                p_list.append(float(tp)/denom)

            # recall
            denom = tp + fn 
            if denom == 0:
                r_list.append(float('nan'))
            else:
                r_list.append(float(tp)/denom)
            
            # Dice
            denom = 2*tp + fn + fp
            if denom == 0:
                d_list.append(float('nan'))
            else:
                d_list.append(2.0*float(tp)/denom)


        efp = open(os.path.join(args.out_dir,"eval.txt"),'w')
        cfp = open(os.path.join(args.out_dir,"conf.csv"),'w')
        
        efp.write("category,iou,precision,recall,dice\n")
        cfp.write("p/g")
        for i in range(catN):
            efp.write("{0},".format(model.CLASSES[i]))
            cfp.write(",{0}".format(model.CLASSES[i]))

            efp.write("{0:.3f},{1:.3f},{2:.3f},{3:.3f}\n".format(iou_list[i],p_list[i],r_list[i],d_list[i]))
        cfp.write("\n")
        
        efp.close()

        for i in range(catN):
            cfp.write("{0}".format(model.CLASSES[i]))
            
            for j in range(catN):
                cfp.write(",{0:.3f}".format(cMat[i][j]))
            cfp.write("\n")
        cfp.close()

if __name__ == '__main__':
    main()