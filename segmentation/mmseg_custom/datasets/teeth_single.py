import os.path as osp
import warnings
from collections import OrderedDict
import json

import mmcv
import numpy as np
from prettytable import PrettyTable
from torch.utils.data import Dataset

from .evaluation_utils import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmengine.logging import MMLogger
from mmengine.fileio import FileClient
from mmseg.registry import DATASETS
from mmengine.dataset import Compose

from .pipelines import LoadTeethAnnotations


@DATASETS.register_module()
class TeethSingleDataset(Dataset):
    """Custom dataset for single head teeth caries segmentation.

    This dataset directly segments caries without conditioning on teeth.
    Classes: 0 (background), 1 (confirmed), 2 (suspect)
    Note: These map to original classes 0, 2, 3 respectively

    Args:
        pipeline (list[dict]): Processing pipeline
        data_list (str): Data list file.
        data_root (str, optional): Data root for img_dir/ann_dir. Default: None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        caries_color_json (str): Path to caries color JSON file.
    """

    CLASSES = ('bg', 'A1_tai_caries_suspect', 'A1_tai_caries_confirmed')
    PALETTE = [[0, 0, 0], [123, 236, 0], [255, 0, 143]]  # black, green, pink

    # METAINFO for mmseg 1.2.2+ compatibility
    METAINFO = dict(
        classes=('bg', 'A1_tai_caries_suspect', 'A1_tai_caries_confirmed'),
        palette=[[0, 0, 0], [123, 236, 0], [255, 0, 143]]
    )

    def __init__(self,
                 pipeline,
                 data_list,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 gt_seg_map_loader_cfg=dict(),
                 file_client_args=dict(backend='disk'),
                 class_color_json=None,  # 互換性のため残す
                 caries_color_json=None):

        self.pipeline = Compose(pipeline)
        self.data_list = data_list
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        
        # Initialize color lists
        self.caries_colors = []

        # Load caries classes from JSON file if provided
        if caries_color_json:
            self._load_classes_from_json(caries_color_json)

        self.gt_seg_map_loader = LoadTeethAnnotations(
            reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)

        self.file_client_args = file_client_args
        self.file_client = FileClient.infer_client(self.file_client_args)

        # Initialize metainfo for IoUMetric compatibility (mmseg 1.2.2+)
        # This allows ValLoop to pass dataset_meta to evaluators
        self.metainfo = dict(classes=self.CLASSES, palette=self.PALETTE)

        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        # load annotations
        self.img_infos = self.load_annotations()

    def _load_classes_from_json(self, caries_color_json):
        """Load caries classes and palette from JSON file."""
        # Load caries segmentation classes
        with open(caries_color_json, 'r', encoding='utf-8') as f:
            caries_data = json.load(f)
        
        # Initialize with background class
        classes = ['bg']
        palette = [[0, 0, 0]]
        
        # Add caries classes (confirmed, suspect)
        for item in caries_data:
            class_name = item[0]
            color = item[1]
            classes.append(class_name)
            palette.append(color)
            self.caries_colors.append(color)
        
        # Update class attributes
        self.__class__.CLASSES = tuple(classes)
        self.__class__.PALETTE = palette

        # Update METAINFO for mmseg 1.2.2+ compatibility
        self.__class__.METAINFO = dict(
            classes=tuple(classes),
            palette=palette
        )

        # Update instance metainfo to ensure IoUMetric receives it
        self.metainfo = dict(
            classes=self.__class__.CLASSES,
            palette=self.__class__.PALETTE
        )

        logger = MMLogger.get_current_instance()
        logger.info(f'Loaded {len(classes)} classes for single head')
        logger.info(f'Classes: {classes}')
        logger.info(f'Caries colors: {self.caries_colors}')

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self):
        fp = open(self.data_list, "r")

        img_infos = []
        for line in fp:
            line = line.strip().replace('\t', ' ')
            lelem = line.strip().split(" ")
            
            img_path = lelem[0]
            if not self.test_mode:
                dmg_path = lelem[1]  # う蝕のアノテーション

            if self.data_root is not None:
                if not osp.isabs(img_path):
                    img_path = osp.join(self.data_root, img_path)

                if not self.test_mode:
                    if not osp.isabs(dmg_path):
                        dmg_path = osp.join(self.data_root, dmg_path)

            if not self.test_mode:
                img_info = dict(filename=img_path, ann=dict(seg_map=dmg_path))
            else:
                img_info = dict(filename=img_path)

            img_infos.append(img_info)

        fp.close()

        return img_infos

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        if self.custom_classes:
            results['label_map'] = self.label_map
        # Add color information to results
        results['caries_colors'] = self.caries_colors

        # Set img_path from img_info for new mmseg API
        if 'img_info' in results and 'filename' in results['img_info']:
            results['img_path'] = results['img_info']['filename']

        # Set seg_map_path from ann_info for new mmseg API
        if 'ann_info' in results and 'seg_map' in results['ann_info']:
            results['seg_map_path'] = results['ann_info']['seg_map']

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new type.
        """

        img_info = self.img_infos[idx]
        ann_info = img_info['ann']
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new type.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_seg_map_by_idx(self, index):
        """Get one ground truth segmentation map for evaluation."""
        ann_info = self.img_infos[index]['ann']
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        # LoadTeethAnnotations sets 'gt_seg_map', not 'gt_semantic_seg'
        return results['gt_seg_map']

    def get_gt_seg_maps(self, efficient_test=None):
        """Get ground truth segmentation maps for evaluation."""
        if efficient_test is not None:
            warnings.warn(
                'DeprecationWarning: ``efficient_test`` has been deprecated '
                'since MMSeg v0.23, the ``get_gt_seg_maps()`` is CPU memory '
                'friendly by default. ')

        for idx in range(len(self)):
            ann_info = self.img_infos[idx]['ann']
            results = dict(ann_info=ann_info)
            self.pre_pipeline(results)
            self.gt_seg_map_loader(results)
            # LoadTeethAnnotations sets 'gt_seg_map', not 'gt_semantic_seg'
            yield results['gt_seg_map']

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                results or predict segmentation map for computing evaluation
                metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            if gt_seg_maps is None:
                gt_seg_maps = self.get_gt_seg_maps()
            num_classes = len(self.CLASSES)
            ret_metrics = eval_metrics(
                results,
                gt_seg_maps,
                num_classes,
                self.ignore_index,
                metric,
                label_map=dict(),
                reduce_zero_label=self.reduce_zero_label)
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        if logger is None:
            logger = MMLogger.get_current_instance()
        logger.info('per class results:')
        logger.info('\n' + class_table_data.get_string())
        logger.info('Summary:')
        logger.info('\n' + summary_table_data.get_string())

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results

    @property
    def custom_classes(self):
        return False