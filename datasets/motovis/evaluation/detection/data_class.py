# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2019.

from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np

from nuscenes.eval.common.data_classes import MetricData, EvalBox
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.constants import ATTRIBUTE_NAMES, TP_METRICS

from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
import pdb    

# DETECTION_NAMES = ["car", "van", "truck", "construction_vehicle", "bus",
#                     "bicycle", "tricycle", "pedestrian", "barrier", "traffic_cone",
#                     "generic_object"]
DETECTION_NAMES = ["car", "truck", "construction_vehicle", "bus",
                    "bicycle", "tricycle", "pedestrian", 
                    "barrier", "traffic_cone", "generic_object"]

TRACKING_NAMES = ["car", "van", "truck","construction_vehicle","bus",'bicycle','tricycle','pedestrian']


OD_CATEGROY_MAPPING = {
    # 小车
    "car": "car",
    'suv': 'car',
    'van': 'car',
    'vehicle': 'car',

    # 卡车, 包含卡车、挂车
    "truck": "truck",
    "trailer": "truck",

    # 工程车
    'crane': 'construction_vehicle',         # 吊车
    'excavator': 'construction_vehicle',   # 挖掘机
    'earthmover': 'construction_vehicle',  # 推土机
    'road_roller': 'construction_vehicle',   # 压路机
    'construction_vehicle': 'construction_vehicle',  # 其他工程车辆

    # 公交车
    "bus": "bus",

    # 障碍
    'pole': 'barrier',                           # 隔离柱
    'column': 'barrier',                         # 柱子
    "barrier": "barrier",
    'isolation_barrel': 'barrier',               # 防撞桶
    'isolation_barrier': 'barrier',              # 隔离栏
    'anti_collision': 'barrier',                 # 防撞球
    # 'triangle_warning': 'barrier',               # 三角标

    'construction_sign': 'barrier',              # 道路施工牌
    'anti_collision_bar': 'barrier',             # 防撞杆
    'czone_sign': 'barrier',                     # 道路施工牌

    # 2轮车  包含自行车，摩托车
    'motor': 'bicycle',
    "bicycle": "bicycle",
    "motorcycle": "bicycle",                     
    
    # 3轮车
    'tricycle': 'tricycle',

    # 行人
    'human': 'pedestrian',
    "pedestrian": "pedestrian",
    
    # 锥桶
    'cone': 'traffic_cone',
    "traffic_cone": "traffic_cone",
    'triangle_warning': 'traffic_cone',  # 三角标
    
    # 通用障碍物
    'other': 'generic_object',   # 其他障碍物
    'others': 'generic_object',
    'generic_object': 'generic_object',
    'general_obstacle': 'generic_object', # 通用障碍物
}

OD_DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "van": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
        "generic_object":"",
        "tricycle":"",
        "others":""
    }
OD_ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }

class MotovisDetectionBox(EvalBox):
    """ Data class used during detection evaluation. Can be a prediction or ground truth."""
    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 detection_name: str = 'car',  # The class name used in the detection challenge.
                 detection_score: float = -1.0,  # GT samples do not have a score.
                 attribute_name: str = ''):  # Box attribute. Each box can have at most 1 attribute.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)
        
        assert detection_name is not None, 'Error: detection_name cannot be empty!'
        assert detection_name in DETECTION_NAMES, 'Error: Unknown detection_name %s' % detection_name

        assert attribute_name in ATTRIBUTE_NAMES or attribute_name == '', \
            'Error: Unknown attribute_name %s' % attribute_name

        assert type(detection_score) == float, 'Error: detection_score must be a float!'
        assert not np.any(np.isnan(detection_score)), 'Error: detection_score may not be NaN!'

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.detection_name == other.detection_name and
                self.detection_score == other.detection_score and
                self.attribute_name == other.attribute_name)

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'detection_name': self.detection_name,
            'detection_score': self.detection_score,
            'attribute_name': self.attribute_name
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   detection_name=content['detection_name'],
                   detection_score=-1.0 if 'detection_score' not in content else float(content['detection_score']),
                   attribute_name=content['attribute_name'])


class MotovisTrackingBox(EvalBox):
    """ Data class used during tracking evaluation. Can be a prediction or ground truth."""

    def __init__(self,
                 sample_token: str = "",
                 translation: Tuple[float, float, float] = (0, 0, 0),
                 size: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
                 velocity: Tuple[float, float] = (0, 0),
                 ego_translation: Tuple[float, float, float] = (0, 0, 0),  # Translation to ego vehicle in meters.
                 num_pts: int = -1,  # Nbr. LIDAR or RADAR inside the box. Only for gt boxes.
                 tracking_id: str = '',  # Instance id of this object.
                 tracking_name: str = '',  # The class name used in the tracking challenge.
                 tracking_score: float = -1.0,
                ):  # Does not apply to GT.

        super().__init__(sample_token, translation, size, rotation, velocity, ego_translation, num_pts)

        assert tracking_name is not None, 'Error: tracking_name cannot be empty!'
        assert tracking_name in TRACKING_NAMES, 'Error: Unknown tracking_name %s' % tracking_name

        assert type(tracking_score) == float, 'Error: tracking_score must be a float!'
        assert not np.any(np.isnan(tracking_score)), 'Error: tracking_score may not be NaN!'

        # Assign.
        self.tracking_id = tracking_id
        self.tracking_name = tracking_name
        self.tracking_score = tracking_score
        

    def __eq__(self, other):
        return (self.sample_token == other.sample_token and
                self.translation == other.translation and
                self.size == other.size and
                self.rotation == other.rotation and
                self.velocity == other.velocity and
                self.ego_translation == other.ego_translation and
                self.num_pts == other.num_pts and
                self.tracking_id == other.tracking_id and
                self.tracking_name == other.tracking_name and
                self.tracking_score == other.tracking_score 
                )

    def serialize(self) -> dict:
        """ Serialize instance into json-friendly format. """
        return {
            'sample_token': self.sample_token,
            'translation': self.translation,
            'size': self.size,
            'rotation': self.rotation,
            'velocity': self.velocity,
            'ego_translation': self.ego_translation,
            'num_pts': self.num_pts,
            'tracking_id': self.tracking_id,
            'tracking_name': self.tracking_name,
            'tracking_score': self.tracking_score,
        }

    @classmethod
    def deserialize(cls, content: dict):
        """ Initialize from serialized content. """
        return cls(sample_token=content['sample_token'],
                   translation=tuple(content['translation']),
                   size=tuple(content['size']),
                   rotation=tuple(content['rotation']),
                   velocity=tuple(content['velocity']),
                   ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
                   else tuple(content['ego_translation']),
                   num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
                   tracking_id=content['tracking_id'],
                   tracking_name=content['tracking_name'],
                   tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']),
                   )