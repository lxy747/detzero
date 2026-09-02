import json
import os
from typing import Dict, List
from nuscenes.eval.detection.data_classes import DetectionConfig

class MotovisDetectionConfig(DetectionConfig):
    """ Data class that specifies the detection evaluation settings. """

    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int):

        # assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        # self.class_names = self.class_range.keys() # 多卡训练时会报错: TypeError: cannot pickle 'dict_keys' object
        self.class_names = list(self.class_range.keys())
        
def config_factory(configuration_name: str) -> MotovisDetectionConfig:
    """
    Creates a DetectionConfig instance that can be used to initialize a NuScenesEval instance.
    Note that this only works if the config file is located in the nuscenes/eval/detection/configs folder.
    :param configuration_name: Name of desired configuration in eval_detection_configs.
    :return: DetectionConfig instance.
    """
    # Check if config exists.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(this_dir, 'configs', '%s.json' % configuration_name)
    assert os.path.exists(cfg_path), \
        'Requested unknown configuration {}'.format(configuration_name)

    # Load config file and deserialize it.
    with open(cfg_path, 'r') as f:
        data = json.load(f)
    cfg = MotovisDetectionConfig.deserialize(data)

    return cfg