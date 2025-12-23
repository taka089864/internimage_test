# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
import cv2

from mmseg.datasets.builder import PIPELINES


class BldgAnnot:

    def __init__(self, facePoly, winPolys, doorPolys):
        self.facePoly = facePoly
        self.winPolys = winPolys
        self.doorPolys = doorPolys

        self.rect = np.array([[facePoly[0][0],facePoly[0][0]],[facePoly[0][1],facePoly[0][1]]])
        for i in range(4):
            if self.rect[0][0] > facePoly[i][0]:
                self.rect[0][0] = facePoly[i][0]
            if self.rect[0][1] < facePoly[i][0]:
                self.rect[0][1] = facePoly[i][0]
            if self.rect[1][0] > facePoly[i][1]:
                self.rect[1][0] = facePoly[i][1]
            if self.rect[1][1] < facePoly[i][1]:
                self.rect[1][1] = facePoly[i][1]
    
    def get_rect(self,img_size):
        sx = max(0,self.rect[0][0])
        ex = min(img_size[0],self.rect[0][1])
        sy = max(0,self.rect[1][0])
        ey = min(img_size[1],self.rect[1][1])
        return [sx,ex,sy,ey]


def convert_color_map_into_one_hot_for_bldg(gimg):
    cls_color = [[64,64,192],[255,0,0]]

    otensor = np.zeros((gimg.shape[0],gimg.shape[1],3))
    for i in range(len(cls_color)):
        cmask = np.all(gimg == cls_color[i],axis=-1)
        otensor[cmask,i+1] = 255
    
    bg_mask = np.all(otensor==0,axis=2)
    otensor[bg_mask,0] = 255

    return otensor



@PIPELINES.register_module()
class LoadBldgSSAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 only_gt=False,
                 only_img=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.only_gt = only_gt
        self.only_img = only_img
        if only_img:
            self.only_gt = False


    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if not self.only_img:
            apath = results['ann_info']['seg_map']
            bldg_ann = results['ann_info']['bldg_ann']
            aimg = cv2.imread(apath)

        if not self.only_gt:
            img = results['img']
            ishape = results['img_shape']
            ori_shape = results['ori_shape']
        else:
            ori_shape = aimg.shape

        out_shape = ori_shape
        if not self.only_img:
            #crop
            aimg = cv2.resize(aimg,(ori_shape[1],ori_shape[0]),interpolation=cv2.INTER_NEAREST)
            rect = bldg_ann.get_rect((ori_shape[1],ori_shape[0]))
            if not self.only_gt:
                img = img[rect[2]:rect[3],rect[0]:rect[1]]
            aimg = aimg[rect[2]:rect[3],rect[0]:rect[1]]

            #convert
            otensor = convert_color_map_into_one_hot_for_bldg(aimg)

            out_shape = aimg.shape

        #padding
        sx = sy = 0
        ex = out_shape[1]
        ey = out_shape[0]
        if out_shape[0] < out_shape[1]:
            sy = (out_shape[1]-out_shape[0])//2
            ey = sy + out_shape[0]

            if not self.only_img:
                pad_gimg = np.zeros((out_shape[1],out_shape[1],otensor.shape[2])).astype(np.uint8)
                pad_gimg[sy:ey,:,:] = otensor.astype(np.uint8)
            if not self.only_gt:
                pad_img = np.zeros((out_shape[1],out_shape[1],3)).astype(np.uint8)
                pad_img[sy:ey,:,:] = img
        else:
            sx = (out_shape[0]-out_shape[1])//2
            ex = sx + out_shape[1]

            if not self.only_img:
                pad_gimg = np.zeros((out_shape[0],out_shape[0],otensor.shape[2])).astype(np.uint8)        
                pad_gimg[:,sx:ex,:] = otensor.astype(np.uint8)
            if not self.only_gt:
                pad_img = np.zeros((out_shape[0],out_shape[0],3)).astype(np.uint8)
                pad_img[:,sx:ex,:] = img

        if not self.only_gt:
            results['img'] = pad_img 
            results['img_shape'] = pad_img.shape
            results['ori_shape'] = pad_img.shape

        results['crop_rect'] = [sx,sy,ex,ey]

        # results['gt_semantic_seg'] = otensor.astype(np.uint8)
        if not self.only_img:
            results['gt_semantic_seg'] = pad_gimg.astype(np.uint8)
        if 'seg_fields' not in results.keys():
            results['seg_fields'] = []
        results['seg_fields'].append('gt_semantic_seg')

        # if self.file_client is None:
        #     self.file_client = mmcv.FileClient(**self.file_client_args)

        # if results.get('seg_prefix', None) is not None:
        #     filename = osp.join(results['seg_prefix'],
        #                         results['ann_info']['seg_map'])
        # else:
        #     filename = results['ann_info']['seg_map']
        # img_bytes = self.file_client.get(filename)
        # gt_semantic_seg = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged',
        #     backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # # reduce zero_label
        # if self.reduce_zero_label:
        #     # avoid using underflow conversion
        #     gt_semantic_seg[gt_semantic_seg == 0] = 255
        #     gt_semantic_seg = gt_semantic_seg - 1
        #     gt_semantic_seg[gt_semantic_seg == 254] = 255
        # # modify if custom classes
        # if results.get('label_map', None) is not None:
        #     # Add deep copy to solve bug of repeatedly
        #     # replace `gt_semantic_seg`, which is reported in
        #     # https://github.com/open-mmlab/mmsegmentation/pull/1445/
        #     gt_semantic_seg_copy = gt_semantic_seg.copy()
        #     for old_id, new_id in results['label_map'].items():
        #         gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # results['gt_semantic_seg'] = gt_semantic_seg
        # results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
