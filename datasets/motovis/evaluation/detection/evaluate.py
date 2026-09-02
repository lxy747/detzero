from nuscenes.eval.detection.evaluate import NuScenesEval

import argparse
import json
import os
import random
import time
from typing import Tuple, Dict, Any

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
# from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from . import MotovisDetectionConfig
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
import pickle
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation as R
from .data_class import MotovisDetectionBox, DETECTION_NAMES, OD_CATEGROY_MAPPING
from tqdm import tqdm
import pdb

def _get_box_class_field(eval_boxes: EvalBoxes) -> str:
    """
    Retrieve the name of the class field in the boxes.
    This parses through all boxes until it finds a valid box.
    If there are no valid boxes, this function throws an exception.
    :param eval_boxes: The EvalBoxes used for evaluation.
    :return: The name of the class field in the boxes, e.g. detection_name or tracking_name.
    """
    assert len(eval_boxes.boxes) > 0
    box = None
    for val in eval_boxes.boxes.values():
        if len(val) > 0:
            box = val[0]
            break
    if isinstance(box, DetectionBox) or isinstance(box, MotovisDetectionBox):
        class_field = 'detection_name'
    elif isinstance(box, TrackingBox):
        class_field = 'tracking_name'
    else:
        raise Exception('Error: Invalid box type: %s' % box)

    return class_field

def load_gt(gt_path: str, box_cls, pred_sample_tokens, verbose: bool = False, ref_cs_is_lidar=True) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    
    def load_extra_info(info, extra_dir):
        if 'filepath' not in info.keys():
            return dict()
        pkl_file = os.path.join(extra_dir,  info['filepath'])
        if not os.path.exists(pkl_file):
            return dict()
        info = pickle.load(open(pkl_file, 'rb'))
        return info

    extra_dir = os.path.split(gt_path)[0]

    all_annotations = EvalBoxes()
    

    with open(gt_path,'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) else data
    ego_pose_dict= dict()
    for info in tqdm(infos, desc='load gt'):
        sample_boxes = [] 
        # with open('data/bench2drive/pkls/'+info['filepath'],'rb') as f:
        #     info = pickle.load(f)
        info.update(load_extra_info(info, extra_dir))
        sample_token = info['token']
        if sample_token not in  pred_sample_tokens:
            continue
        # 保存ego pose
        if sample_token not in ego_pose_dict:
            ego_pose_dict[sample_token] = info['ego2global_translation']
        
        if box_cls == MotovisDetectionBox:
            if 'gt_names' in info.keys():
                gt_names, gt_boxes, gt_velocity = info['gt_names'],info['gt_boxes'], info['gt_velocity']
            else:
                gt_names, gt_boxes, gt_velocity = info['anns']['gt_names'],info['anns']['gt_boxes'], info['anns']['gt_velocity_3d']
     
            # for name, boxes, velocity in zip(info['gt_names'],info['gt_boxes'], info['gt_velocity']):
            for name, boxes, velocity in zip(gt_names, gt_boxes, gt_velocity):
                if name in OD_CATEGROY_MAPPING:
                    motovis_name = OD_CATEGROY_MAPPING[name]
                    if motovis_name not in DETECTION_NAMES:
                        continue
                else:
                    continue
                g_box = Box(center=boxes[:3],size=boxes[[4,3,5]],orientation=Quaternion(axis=[0, 0, 1], angle=boxes[6]),velocity = (velocity[0],velocity[1], 0 ))
                if ref_cs_is_lidar:
                    # lidar2ego_rotation = Quaternion(matrix=info['lidar2ego_rotation'])
                    lidar2ego_rotation = R.from_matrix(info['lidar2ego_rotation']).as_quat()
                    lidar2ego_rotation = Quaternion(np.array([lidar2ego_rotation[3], lidar2ego_rotation[0], lidar2ego_rotation[1], lidar2ego_rotation[2]]))
                    g_box.rotate(lidar2ego_rotation)
                    g_box.translate(np.array(info["lidar2ego_translation"]))
                # ego2global_rotation = Quaternion(matrix=info['ego2global_rotation'])
                ego2global_rotation = R.from_matrix(info['ego2global_rotation']).as_quat()
                ego2global_rotation = Quaternion(np.array([ego2global_rotation[3], ego2global_rotation[0], ego2global_rotation[1], ego2global_rotation[2]]))
                g_box.rotate(ego2global_rotation)
                g_box.translate(np.array(info["ego2global_translation"]))
                sample_boxes.append(
                        MotovisDetectionBox(
                            sample_token=sample_token,
                            translation=g_box.center.tolist(),
                            size=g_box.wlh.tolist(),
                            rotation=g_box.orientation.elements.tolist(),
                            velocity=g_box.velocity[:2].tolist(),
                            num_pts=1,
                            detection_name=motovis_name,
                            detection_score=-1.0,  # GT samples do not have a score.
                            attribute_name=''
                        )
                    )
        all_annotations.add_boxes(sample_token, sample_boxes)
    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(all_annotations.sample_tokens)))

    return all_annotations, ego_pose_dict
    
def add_center_dist(ego_pose_dict,
                    eval_boxes: EvalBoxes):
    """
    Adds the cylindrical (xy) center distance from ego vehicle to each box.
    :param nusc: The NuScenes instance.
    :param eval_boxes: A set of boxes, either GT or predictions.
    :return: eval_boxes augmented with center distances.
    """
    
    for sample_token in eval_boxes.sample_tokens:
        pose_record = ego_pose_dict[sample_token]

        for box in eval_boxes[sample_token]:
            # Both boxes and ego pose are given in global coord system, so distance can be calculated directly.
            # Note that the z component of the ego pose is 0.
            ego_translation = (box.translation[0] - pose_record[0],
                               box.translation[1] - pose_record[1],
                               box.translation[2] - pose_record[2])
            if isinstance(box, DetectionBox) or isinstance(box, TrackingBox) or isinstance(box, MotovisDetectionBox) :
                box.ego_translation = ego_translation
            else:
                raise NotImplementedError
            
    return eval_boxes

def filter_eval_boxes(eval_boxes: EvalBoxes,
                      max_dist: Dict[str, float],
                      verbose: bool = False) -> EvalBoxes:
    """
    Applies filtering to boxes. Distance, bike-racks and points per box.
    :param nusc: An instance of the NuScenes class.
    :param eval_boxes: An instance of the EvalBoxes class.
    :param max_dist: Maps the detection name to the eval distance threshold for that class.
    :param verbose: Whether to print to stdout.
    """
    # Retrieve box type for detectipn/tracking boxes.
    class_field = _get_box_class_field(eval_boxes)
    
    # pdb.set_trace()
    # Accumulators for number of filtered boxes.
    total, dist_filter, point_filter, bike_rack_filter = 0, 0, 0, 0
    for ind, sample_token in enumerate(eval_boxes.sample_tokens):

        # Filter on distance first.
        total += len(eval_boxes[sample_token])

        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if
                                          (box.__getattribute__(class_field) in max_dist.keys()) and (box.ego_dist < max_dist[box.__getattribute__(class_field)])]
        dist_filter += len(eval_boxes[sample_token])

        # Then remove boxes with zero points in them. Eval boxes have -1 points by default.
        eval_boxes.boxes[sample_token] = [box for box in eval_boxes[sample_token] if not box.num_pts == 0]
        point_filter += len(eval_boxes[sample_token])

        filtered_boxes = []
        for box in eval_boxes[sample_token]:
            filtered_boxes.append(box)

        eval_boxes.boxes[sample_token] = filtered_boxes
        bike_rack_filter += len(eval_boxes.boxes[sample_token])

    if verbose:
        print("=> Original number of boxes: %d" % total)
        print("=> After distance based filtering: %d" % dist_filter)
        print("=> After LIDAR points based filtering: %d" % point_filter)

    return eval_boxes

class MotovisEval(NuScenesEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """
    def __init__(self,
                 gt_path: str,
                 config: MotovisDetectionConfig,
                 result_path: str,
                 output_dir: str = None,
                 verbose: bool = True,
                 ref_cs_is_lidar=True
                 ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.ref_cs_is_lidar=ref_cs_is_lidar
        self.result_path = result_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.gt_path = gt_path
        
        # cfg 
        # self.cfg.max_boxes_per_sample = 300
        # self.cfg.class_range = {'car': 50, 
        #                         'truck': 50, 
        #                         'bus': 50, 
        #                         'van':50,
        #                         'construction_vehicle': 50, 
        #                         'pedestrian': 40, 
        #                         'bicycle' :40,
        #                         'tricycle':40,
        #                         'traffic_cone': 30, 
        #                         'barrier': 30,
        #                         'generic_object': 30}
        
        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, MotovisDetectionBox,
                                                     verbose=verbose)
        # pdb.set_trace()
        self.gt_boxes, ego_pose_dict = load_gt(self.gt_path, MotovisDetectionBox, self.pred_boxes.sample_tokens, verbose=verbose, ref_cs_is_lidar=self.ref_cs_is_lidar)
        
        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens) ==set(ego_pose_dict.keys()), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(ego_pose_dict, self.pred_boxes)
        self.gt_boxes = add_center_dist(ego_pose_dict, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens
        
    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        # pdb.set_trace()
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta # self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err']))

        return metrics_summary
        