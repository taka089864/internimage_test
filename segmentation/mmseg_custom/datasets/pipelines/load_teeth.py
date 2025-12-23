import os.path as osp

import mmcv
import numpy as np
from PIL import Image
from mmengine.fileio import FileClient

from mmseg.registry import TRANSFORMS as PIPELINES


@PIPELINES.register_module()
class LoadTeethAnnotations:
    """Load annotations for single head teeth caries segmentation.
    
    Classes: 0 (background), 1 (suspect), 2 (confirmed)
    
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
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):

        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def color_to_caries_one_hot_old(self, seg_data, caries_colors):
        """Convert color image to one-hot encoded format for caries segmentation.

        Args:
            seg_data: Input color image array
            caries_colors: List of colors for caries classes

        Returns:
            One-hot encoded array with shape (H, W, 3)
            Channel 0: background
            Channel 1: suspect
            Channel 2: confirmed
        """
        img_shape = seg_data.shape[:2]
        one_hot = np.zeros((*img_shape, 3), dtype=np.uint8)

        # Default colors if not provided (suspect: green, confirmed: red)
        if not caries_colors:
            caries_colors = [[123, 236, 0], [255, 0, 143]]  # green: suspect, red: confirmed

        # Initialize all pixels as background
        one_hot[:, :, 0] = 255

        # Check for caries pixels
        # caries_colors[0] is suspect, caries_colors[1] is confirmed
        if len(caries_colors) > 0:
            # Suspect (green) -> channel 1 (matches CLASSES index)
            mask = np.all(seg_data == caries_colors[0], axis=-1)
            one_hot[mask, 0] = 0  # Remove background
            one_hot[mask, 1] = 255  # Set suspect

        if len(caries_colors) > 1:
            # Confirmed (red) -> channel 2 (matches CLASSES index)
            mask = np.all(seg_data == caries_colors[1], axis=-1)
            one_hot[mask, 0] = 0  # Remove background
            one_hot[mask, 2] = 255  # Set confirmed

        return one_hot

    def color_to_caries_one_hot(self, seg_data, caries_colors):
        """Convert color image to one-hot encoded format for caries segmentation.

        動的なクラス数に対応：2クラスでも4クラスでもN個のクラスでも処理可能。

        Args:
            seg_data: Input color image array (BGR format)
            caries_colors: List of colors for caries classes (RGB format)
                          [[r1, g1, b1], [r2, g2, b2], ...]

        Returns:
            One-hot encoded array with shape (H, W, num_classes)
            Channel 0: background
            Channel 1~N: caries classes (in order of caries_colors)
        """
        img_shape = seg_data.shape[:2]

        # Calculate number of classes dynamically (background + caries classes)
        num_classes = len(caries_colors) + 1

        # Initialize one-hot array with dynamic number of channels
        one_hot = np.zeros((*img_shape, num_classes), dtype=np.uint8)

        # Initialize all pixels as background
        one_hot[:, :, 0] = 255

        # Process each caries class with loop
        for class_idx, color in enumerate(caries_colors):
            # Create mask for pixels matching this color
            mask = np.all(seg_data == color, axis=-1)

            # Update one-hot encoding
            # Channel index: class_idx + 1 (because channel 0 is background)
            one_hot[mask, 0] = 0  # Remove background
            one_hot[mask, class_idx + 1] = 255  # Set class

        return one_hot

    def color_to_class_indices(self, seg_data, caries_colors):
        """Convert color image to class indices for mmseg IoUMetric compatibility.

        Args:
            seg_data: Input color image array (BGR format)
            caries_colors: List of colors for caries classes (RGB format)

        Returns:
            2D array with shape (H, W) containing class indices
            0: background, 1+: caries classes
        """
        img_shape = seg_data.shape[:2]

        # Initialize with background class (0)
        class_indices = np.zeros(img_shape, dtype=np.uint8)

        # Assign class index for each caries color
        for class_idx, color in enumerate(caries_colors):
            # Create mask for pixels matching this color
            mask = np.all(seg_data == color, axis=-1)
            # Class index: class_idx + 1 (because 0 is background)
            class_indices[mask] = class_idx + 1

        return class_indices

    def __call__(self, results):
        """Call function to load annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if results.get('seg_map', None) is not None:
            filename = results['seg_map']
        else:
            filename = results['ann_info']['seg_map']

        img_bytes = self.file_client.get(filename)

        # Load image
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='color', backend=self.imdecode_backend).squeeze()

        # Get caries colors from results
        caries_colors = results.get('caries_colors', [])

        # Convert color image to class indices (H, W) for mmseg compatibility
        gt_semantic_seg = self.color_to_class_indices(gt_semantic_seg, caries_colors)

        # Set modified results with correct key name for PackSegInputs
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


