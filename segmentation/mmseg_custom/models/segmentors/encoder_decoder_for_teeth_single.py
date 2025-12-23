# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor


def add_prefix(inputs, prefix):
    """Add prefix to keys of a dict.

    Args:
        inputs (dict): The input dictionary.
        prefix (str): The prefix to add.

    Returns:
        dict: The dict with prefixed keys.
    """
    outputs = dict()
    for key, value in inputs.items():
        outputs[f'{prefix}.{key}'] = value
    return outputs


def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    """Wrapper for F.interpolate to match old mmseg.ops.resize signature."""
    return F.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)



@SEGMENTORS.register_module()
class EncoderDecoderForTeethSingle(BaseSegmentor):
    """Encoder Decoder segmentor for single-head teeth caries segmentation.

    This is a specialized version for direct caries segmentation without
    conditioning on teeth segmentation. It consists of backbone, decode_head, 
    and optionally auxiliary_head.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 isLane=False):

        super(EncoderDecoderForTeethSingle, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.isLane = isLane

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses


    # def forward_test(self, imgs, img_metas):
    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0])
            # return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas)
            # return self.aug_test(imgs, img_metas, **kwargs)

        # return self.simple_test(imgs, img_metas, **kwargs)

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        # output = F.softmax(seg_logit, dim=1)
        output = seg_logit
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        # seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            # seg_pred = seg_pred.unsqueeze(0)
            # return seg_pred
            return seg_logit

        if not self.isLane:
            seg_pred = self.convert_logits_to_label(seg_logit)
        else:
            seg_pred = self.convert_lane_logits_to_label(seg_logit)

        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit_list = []
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        seg_logit_list.append(seg_logit)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            # seg_logit += cur_seg_logit
            seg_logit_list.append(cur_seg_logit)
        
        # seg_logit /= len(imgs)
        # seg_pred = seg_logit.argmax(dim=1)
        if not self.isLane:
            seg_pred = self.convert_logits_to_label_for_aug(seg_logit_list)
        else:
            seg_pred = self.convert_lane_logits_to_label_for_aug(seg_logit_list)

        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


    def convert_logits_to_label(self, seg_logit):
        seg_logit = seg_logit.reshape(-1, seg_logit.size(1), seg_logit.size(2), seg_logit.size(3))

        # う蝕のセグメンテーション: 0=背景, 1=疑い, 2=確定
        caries_logit = F.softmax(seg_logit, dim=1)
        caries_lab = caries_logit.argmax(dim=1)
        
        return caries_lab

    def convert_logits_to_label_back(self, seg_logit):
        TEETH_CLASS_NUM = 2  # 背景、歯

        seg_logit = seg_logit.reshape(-1, seg_logit.size(1), seg_logit.size(2), seg_logit.size(3))

        # 歯のセグメンテーション: 0=背景, 1=歯
        teeth_logit = seg_logit[:, :TEETH_CLASS_NUM,:,:]
        teeth_logit = F.softmax(teeth_logit, dim=1)

        # う蝕のセグメンテーション: 0=背景, 1=疑い, 2=確定
        caries_logit = seg_logit[:, TEETH_CLASS_NUM:,:,:]
        caries_logit = F.softmax(caries_logit, dim=1)

        teeth_lab = teeth_logit.argmax(dim=1)
        caries_lab = caries_logit.argmax(dim=1)

        # 歯がある場所（teeth_lab == 1）かつ、う蝕がある場所（caries_lab > 0）
        # の場合は、う蝕のラベルを表示（クラス番号を調整: 2=疑い, 3=確定）
        caries_mask = (teeth_lab == 1) & (caries_lab > 0)

        pred = teeth_lab
        pred[caries_mask] = caries_lab[caries_mask] + TEETH_CLASS_NUM - 1

        return pred


    def convert_lane_logits_to_label(self, seg_logit):
        seg_logit = seg_logit.reshape(-1, seg_logit.size(1), seg_logit.size(2), seg_logit.size(3))
        seg_logit = F.softmax(seg_logit,dim=1)

        lab = seg_logit.argmax(dim=1)

        return lab


    def convert_logits_to_label_for_aug(self, seg_logit_list):
        TEETH_CLASS_NUM = 2  # 背景、歯

        for i in range(len(seg_logit_list)):
            seg_logit = seg_logit_list[i]
            seg_logit = seg_logit.reshape(-1, seg_logit.size(1), seg_logit.size(2), seg_logit.size(3))

            tlogit = seg_logit[:, :TEETH_CLASS_NUM,:,:]
            tlogit = F.softmax(tlogit, dim=1)
            clogit = seg_logit[:, TEETH_CLASS_NUM:,:,:]
            clogit = F.softmax(clogit, dim=1)

            if i == 0:
                teeth_logit = tlogit
                caries_logit = clogit
            else:
                teeth_logit += tlogit
                caries_logit += clogit

        teeth_logit /= len(seg_logit_list)
        caries_logit /= len(seg_logit_list)
            
        teeth_lab = teeth_logit.argmax(dim=1)
        caries_lab = caries_logit.argmax(dim=1)
        
        # 歯がある場所（teeth_lab == 1）かつ、う蝕がある場所（caries_lab > 0）
        # の場合は、う蝕のラベルを表示（クラス番号を調整: 2=疑い, 3=確定）
        caries_mask = (teeth_lab == 1) & (caries_lab > 0)

        pred = teeth_lab 
        pred[caries_mask] = caries_lab[caries_mask] + TEETH_CLASS_NUM - 1

        return pred

    
    def convert_lane_logits_to_label_for_aug(self, seg_logit_list):

        for i in range(len(seg_logit_list)):
            seg_logit = seg_logit_list[i]
            seg_logit = seg_logit.reshape(-1, seg_logit.size(1), seg_logit.size(2), seg_logit.size(3))

            lab = F.softmax(seg_logit,dim=1)

            if i == 0:
                label = lab
            else:
                label += lab

        label /= len(seg_logit_list)

        pred = label.argmax(dim=1)

        return pred

    # ========================================================================
    # New API methods for mmengine compatibility (minimal changes approach)
    # ========================================================================

    def _parse_data_samples(self, data_samples):
        """Parse data_samples to extract gt_semantic_seg and img_metas.

        Args:
            data_samples (list[SegDataSample]): The seg data samples.

        Returns:
            tuple: (gt_semantic_seg, img_metas)
        """
        gt_semantic_seg = torch.stack([
            data_sample.gt_sem_seg.data for data_sample in data_samples
        ])
        img_metas = [data_sample.metainfo for data_sample in data_samples]
        return gt_semantic_seg, img_metas

    def _pack_seg_results(self, seg_pred_list, data_samples):
        """Pack segmentation results into SegDataSample format.

        Args:
            seg_pred_list (list[ndarray]): Segmentation predictions.
            data_samples (list[SegDataSample]): The seg data samples.

        Returns:
            list[SegDataSample]: Packed segmentation results.
        """
        from mmseg.structures import SegDataSample
        from mmengine.structures import PixelData

        results = []
        for seg_pred, data_sample in zip(seg_pred_list, data_samples):
            # clone to keep gt_sem_seg and other annotations for evaluator
            if data_sample is not None:
                result = data_sample.clone()
            else:
                result = SegDataSample()

            pred_tensor = torch.from_numpy(seg_pred)
            if pred_tensor.ndim == 2:
                pred_tensor = pred_tensor.unsqueeze(0)
            result.pred_sem_seg = PixelData(data=pred_tensor.long())
            results.append(result)
        return results

    def loss(self, inputs, data_samples):
        """Calculate losses from a batch of inputs and data samples.

        This is the new API method that wraps the old forward_train().

        Args:
            inputs (Tensor or list[Tensor]): Input images.
            data_samples (list[SegDataSample]): The seg data samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # Handle case where inputs is a list of tensors (from new data pipeline)
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)

        gt_semantic_seg, img_metas = self._parse_data_samples(data_samples)
        return self.forward_train(inputs, img_metas, gt_semantic_seg)

    def predict(self, inputs, data_samples=None):
        """Predict results from a batch of inputs and data samples.

        This is the new API method that wraps the old forward_test().

        Args:
            inputs (Tensor): Input images.
            data_samples (list[SegDataSample], optional): The seg data samples.

        Returns:
            list[SegDataSample]: Segmentation results.
        """
        # Handle case where inputs is a list of tensors (from new data pipeline)
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)

        if data_samples is not None:
            # Extract metainfo and ensure all required keys are present
            import numpy as np
            img_metas = []
            for data_sample in data_samples:
                metainfo = data_sample.metainfo.copy()

                # Get default 3-element shape tuple (H, W, C) from inputs
                # inputs shape is (N, C, H, W), so default shape is (H, W, C)
                default_shape = (inputs.shape[2], inputs.shape[3], inputs.shape[1])

                # Ensure required keys are present with default values if missing
                # All shape keys should be 3-element tuples (H, W, C)
                if 'img_shape' not in metainfo:
                    metainfo['img_shape'] = default_shape
                elif len(metainfo['img_shape']) == 2:
                    # If only (H, W), add C from inputs
                    metainfo['img_shape'] = (*metainfo['img_shape'], inputs.shape[1])

                if 'pad_shape' not in metainfo:
                    metainfo['pad_shape'] = metainfo.get('img_shape', default_shape)
                elif len(metainfo['pad_shape']) == 2:
                    metainfo['pad_shape'] = (*metainfo['pad_shape'], inputs.shape[1])

                if 'ori_shape' not in metainfo:
                    metainfo['ori_shape'] = metainfo.get('img_shape', default_shape)
                elif len(metainfo['ori_shape']) == 2:
                    metainfo['ori_shape'] = (*metainfo['ori_shape'], inputs.shape[1])

                # scale_factor should be 4-element ndarray [scale_h, scale_w, scale_h, scale_w]
                if 'scale_factor' not in metainfo:
                    metainfo['scale_factor'] = np.ones(4, dtype=np.float32)
                else:
                    # Convert to numpy array if it's a list or tuple
                    if isinstance(metainfo['scale_factor'], (list, tuple)):
                        metainfo['scale_factor'] = np.array(metainfo['scale_factor'], dtype=np.float32)

                    # Now handle based on array length
                    if isinstance(metainfo['scale_factor'], np.ndarray):
                        if len(metainfo['scale_factor']) == 2:
                            # If it's a 2-element array [scale_h, scale_w], expand to 4-element
                            metainfo['scale_factor'] = np.array([
                                metainfo['scale_factor'][0], metainfo['scale_factor'][1],
                                metainfo['scale_factor'][0], metainfo['scale_factor'][1]
                            ], dtype=np.float32)
                        # If already 4 elements, keep as is
                    else:
                        # If it's a scalar, convert to 4-element array
                        metainfo['scale_factor'] = np.ones(4, dtype=np.float32) * metainfo['scale_factor']

                if 'flip' not in metainfo:
                    metainfo['flip'] = False
                if 'flip_direction' not in metainfo:
                    metainfo['flip_direction'] = None

                # Validate shape tuples
                assert len(metainfo['img_shape']) == 3, f"img_shape must be 3-element tuple, got {metainfo['img_shape']}"
                assert len(metainfo['pad_shape']) == 3, f"pad_shape must be 3-element tuple, got {metainfo['pad_shape']}"
                assert len(metainfo['ori_shape']) == 3, f"ori_shape must be 3-element tuple, got {metainfo['ori_shape']}"

                img_metas.append([metainfo])
        else:
            # Create default img_metas if not provided
            # All shapes should be 3-element tuples (H, W, C)
            import numpy as np
            default_shape = (inputs.shape[2], inputs.shape[3], inputs.shape[1])
            img_metas = [[dict(
                ori_shape=default_shape,
                img_shape=default_shape,
                pad_shape=default_shape,
                scale_factor=np.ones(4, dtype=np.float32),
                flip=False,
                flip_direction=None
            )] for _ in range(inputs.shape[0])]

        seg_pred_list = self.forward_test([inputs], img_metas)

        if data_samples is None:
            # Create dummy data_samples for packing
            from mmseg.structures import SegDataSample
            data_samples = [SegDataSample() for _ in range(inputs.shape[0])]
            for ds, meta in zip(data_samples, img_metas):
                ds.set_metainfo(meta[0])

        return self._pack_seg_results(seg_pred_list, data_samples)

    def _forward(self, inputs, data_samples=None):
        """Network forward process (tensor mode).

        This is the new API method for tensor mode forward.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[SegDataSample], optional): The seg data samples.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head(x)
