

from ..nuscenes.nuscenes_dataset import NuScenesDataset
import pickle
import numpy as np
import laspy
import tempfile
from detzero_det.datasets.augmentor.data_augmentor import MotovisDataAugmentor
from detzero_utils import common_utils

from detzero_utils.ops.roiaware_pool3d import roiaware_pool3d_utils
from tqdm import tqdm
import os
import copy
from pathlib import Path
try:
    import open3d
except Exception as e:
    pass
from nuscenes.utils.data_classes import Box as NuScenesBox
from scipy.spatial.transform import Rotation as R
import pyquaternion, json
from .evaluation.detection.data_class import OD_CATEGROY_MAPPING, OD_ErrNameMapping
import random
import pdb
# b2d_lidar2motovis_lidar = np.array([
#                             [0,1,0,0],
#                             [1,0,0,0.39],
#                             [0,0,1,-1.84],
#                             [0,0,0,1],
#             ])

def load_lidar_points(lidar_file: str, dim: int = 3):
    extname = os.path.splitext(lidar_file)[-1]
    if extname == '.pcd':
        pcd = open3d.t.io.read_point_cloud(lidar_file)
        points = pcd.points['positions'].numpy()
    elif extname=='.npz':
        pcd = np.load(lidar_file)
        points = pcd['positions'].astype(np.float32)
    elif extname in ['.las', '.laz']:
        las_data = laspy.read(lidar_file)
        points = las_data.xyz
    else:
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, dim)
        
        # points = np.fromfile(lidar_file)
    points = points.astype(np.float32)[:, :3]
    isnan = np.isnan(points)
    isinf = np.isinf(points)
    valid = np.logical_or(isnan, isinf).sum(axis=1) < 1

    valid = np.logical_and(valid, np.abs(points[:, 0]) > 0.5)
    valid = np.logical_and(valid, np.abs(points[:, 1]) > 0.5)

    points = points[valid, :]
    return points

def remap_class_names(input_dict, other_keys = []):
    '''
        将不在OD_CATEGROY_MAPPING里面的类别删掉
    '''
    keys = ['gt_names', 'gt_boxes', 'gt_ids', 'instance_inds', 'num_lidar_pts', 'gt_velocity', 'gt_velocity_3d'] + other_keys
    if 'gt_names' not in input_dict.keys():
        return input_dict
    
    gt_names = input_dict['gt_names']
    mask = np.zeros((len(gt_names),), dtype=np.bool_)
    for i in range(len(gt_names)):
        name = OD_CATEGROY_MAPPING.get(gt_names[i], None)
        if name is None:
            continue
        mask[i] = True
        gt_names[i] = name
    for k in keys:
        if k not in input_dict.keys():
            continue
        # pdb.set_trace()
        if isinstance(input_dict[k], list):
            input_dict[k] = np.array(input_dict[k])
  
        input_dict[k] = input_dict[k][mask]
  
    return input_dict


class MotovisDetectionDataset(NuScenesDataset):
    timestamp_scale=1.0e-6
    timestamp_interval=0.5
    max_frame_shift = 5
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        self.ONLY_KEY_FRAME = dataset_cfg.get('ONLY_KEY_FRAME', True)
        self.use_next_frame = dataset_cfg.get('use_next_frame', False)
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        
        if self.training:
            return
        from .evaluation.detection import config_factory
        det3d_eval_version= self.dataset_cfg.get('det3d_eval_version', "motovis_detection_cvpr_2019")  
        self.det3d_eval_configs = config_factory(det3d_eval_version)

    @property
    def pt_dim(self):
        return self.dataset_cfg.get('PT_DIM', 3)

    def _remap_class_names(self, input_dict):
        if 'anns' in input_dict.keys():
            input_dict['anns'] = remap_class_names(input_dict['anns'])
            return input_dict
        return remap_class_names(input_dict)

    def init_data_augmentor(self):
        if self.dataset_cfg.DATA_AUGMENTOR is None:
            return None
        data_augmentor = MotovisDataAugmentor(
            self.root_path,
            self.dataset_cfg.DATA_AUGMENTOR,
            self.class_names,
            logger=self.logger
        ) if self.training else None
        return data_augmentor

    def include_nuscenes_data(self, mode):
        # pdb.set_trace()
        self.logger.info('Loading Motovis dataset')
        nuscenes_infos = []
        extra_path = f'extra_path_{mode}'
        self.extra_path = self.dataset_cfg.INFO_PATH.get(extra_path, '.')
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                self.logger.info('not exist info_path: %s' % str(info_path))
                continue
            self.ann_file=info_path
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                # pdb.set_trace()
                if isinstance(infos, dict):
                    infos = infos['infos']
                infos = [self._remap_class_names(info) for info in infos]
                nuscenes_infos.extend(infos)
        # pdb.set_trace()
        for i, info in enumerate(nuscenes_infos):
            info['raw_index'] = i
        self.raw_infos = nuscenes_infos
        
        if self.ONLY_KEY_FRAME:
            key_infos = []
            for i, info in enumerate(nuscenes_infos):
                if info['key_frame']:
                    key_infos.append(info)
            self.infos.extend(key_infos)
        else:
            self.infos.extend(nuscenes_infos)
        # pdb.set_trace()
        self.logger.info('Total samples for Motovis dataset: %d' % (len(self.infos)))

    def get_neighbor_sweep(self, cur_index_or_info, curr_global2lidar=None, curr_timestamp=0.0, flag=1):
        '''
            flag:1, get next frame
            flag: -1, get prev frame
        '''
        # assert flag in [-1, 1]
        if isinstance(cur_index_or_info, dict):
            raw_index = cur_index_or_info['raw_index']
            scene_token = cur_index_or_info['scene_token']
        else:
            raw_index = self.infos[cur_index_or_info]['raw_index']
            scene_token = self.infos[cur_index_or_info]['scene_token']
        # pdb.set_trace()
        # if self.training:
        #     neightbor_index = raw_index + np.random.randint(1, self.max_frame_shift)*flag
        # else:
        #     neightbor_index = raw_index + flag
        neightbor_index = raw_index + flag
        if (neightbor_index>=len(self.raw_infos)) or (neightbor_index < 0) or (scene_token != self.raw_infos[neightbor_index]['scene_token']):
            return None, None
        return self.get_sweep(self.raw_infos[neightbor_index], curr_global2lidar, curr_timestamp)

    def get_sweep(self, sweep_info, curr_global2lidar=None, curr_timestamp=0.0):
        
        # pdb.set_trace()
        sweep_time = sweep_info['timestamp'] 
        cur_times = self.timestamp_scale * (sweep_time - curr_timestamp)
        if abs(cur_times)> self.timestamp_interval:
            return None, None
        if 'filepath' in sweep_info.keys():
            sweep_info = self.load_extra_info(sweep_info)

        if 'lidar_path' in sweep_info.keys():
            lidar_path = self.root_path / sweep_info['lidar_path']
        else:    
            lidar_path = self.root_path / sweep_info['data_path']
        # points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = load_lidar_points(lidar_path, self.pt_dim)
        points_sweep = self._remove_ego_pts(points_sweep)
        # pdb.set_trace()
        # points_sweep = remove_ego_points(points_sweep).T
        points_sweep = points_sweep.T
        # if sweep_info['transform_matrix'] is not None:
        if curr_global2lidar is not None:
            num_points = points_sweep.shape[1]
            '''
                lidar2global = ego2global @ sensor2global
            '''
            try:
                if self.REF_CS_IS_LIDAR:
                    sweep_lidar2cur_lidar = curr_global2lidar @ common_utils.get_matrix(sweep_info, 'lidar2global')
                else:
                    sweep_lidar2cur_lidar = curr_global2lidar @ common_utils.get_matrix(sweep_info, 'ego2global')
            
            except Exception as e:
                sweep_ego2global = np.eye(4, dtype=curr_global2lidar.dtype)
                sweep_ego2global[:3,:3] = sweep_info['ego2global_rotation']
                sweep_ego2global[:3,3] = sweep_info['ego2global_translation']
                sweep_lidar2cur_lidar = curr_global2lidar @ sweep_ego2global
                if self.REF_CS_IS_LIDAR:
                    sweep_lidar2ego = np.eye(4, dtype=curr_global2lidar.dtype)
                    sweep_lidar2ego[:3,:3] = sweep_info['sensor2ego_rotation']
                    sweep_lidar2ego[:3,3] = sweep_info['sensor2ego_translation']
                    sweep_lidar2cur_lidar = sweep_lidar2cur_lidar @ sweep_lidar2ego
            t = sweep_lidar2cur_lidar[:2,3]
            dis = np.linalg.norm(t)
            # 0.1s: 3.333m: 120km/h
            if dis > 3.4:
                # print('dis', dis)
                return None, None
            # print(t)
            # pdb.set_trace()
            points_sweep[:3, :] = sweep_lidar2cur_lidar.dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]
            
            # points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
            #     np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        # sweep_time = sweep_info['timestamp'] 
        # cur_times = self.timestamp_scale * (sweep_time - curr_timestamp) * np.ones((1, points_sweep.shape[1]))
        cur_times = cur_times + np.zeros((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index_or_info, max_sweeps=1):
        if isinstance(index_or_info, int):
            info = self.infos[index_or_info]
        elif isinstance(index_or_info,dict):
            info = index_or_info
        else:
            assert False
        # pdb.set_trace()
        # lidar_path = self.root_path / info['lidar_path']
        if 'lidar_path' in info.keys():
            lidar_path = self.root_path / info['lidar_path']
        else:    
            lidar_path = self.root_path / info['data_path']
        # points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        # bench2drive点云只有xyz
        points = load_lidar_points(lidar_path, dim=self.pt_dim)
        # pdb.set_trace()
        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]
        
        # curr_global2lidar = info['global2lidar']
        if self.REF_CS_IS_LIDAR:
            curr_global2lidar = common_utils.get_matrix(info, 'global2lidar')
        else:
            curr_global2lidar = common_utils.get_matrix(info, 'global2ego')
        curr_timestamp = info['timestamp']
        # pdb.set_trace()
        frame_ids = []
        if max_sweeps >0:
            assert max_sweeps%2 == 1
            if self.training:
                if self.use_next_frame:
                    num_frame = max_sweeps//2
                    assert num_frame <= self.max_frame_shift
                    p = random.uniform(0, 1)
                    if p < 0.1:
                        k1, k2 = 0,0
                    elif p<0.55:
                        k1 = random.randint(0, num_frame)
                        k2 = random.randint(0, num_frame)
                    else:
                        k1, k2 = num_frame, num_frame
                    
                    if k1>0:
                        frame_ids.extend(random.sample(list(range(-self.max_frame_shift, 0)), k=k1))
                    if k2>0:
                        frame_ids.extend(random.sample(list(range(1, self.max_frame_shift+1)), k=k2))

                    # frame_ids.extend(random.sample(list(range(-self.max_frame_shift, 0)), k=num_frame))
                    # frame_ids.extend(random.sample(list(range(1, self.max_frame_shift+1)), k=num_frame))
                else:
                    assert self.max_frame_shift >= (max_sweeps-1)
                    p = random.uniform(0, 1)
                    if p<0.1:
                        k=0
                    elif p<0.55:
                        k = random.randint(0, max_sweeps -1)
                    else:
                        k = max_sweeps-1
                    if k>0:
                        frame_ids.extend(random.sample(list(range(-self.max_frame_shift, 0)), k=k))
            else:
                if self.use_next_frame:
                    num_frame = max_sweeps//2
                    assert num_frame <= self.max_frame_shift
                    frame_ids.extend(list(range(-num_frame, 0)))
                    frame_ids.extend(list(range(1, num_frame+1)))
                    # frame_mode = {3: [-2, 2]}
                    # frame_ids = frame_mode[max_sweeps]

                else:
                    assert self.max_frame_shift >= (max_sweeps-1)
                    frame_ids.extend(list(range(-(max_sweeps-1), 0)))
        
        for flag in frame_ids:
            # pdb.set_trace()
            prev_points, prev_times = self.get_neighbor_sweep(index_or_info, curr_global2lidar=curr_global2lidar, curr_timestamp=curr_timestamp, flag=flag)
            if prev_times is not None:
                sweep_points_list.append(prev_points)
                sweep_times_list.append(prev_times)
            # pdb.set_trace()
            # if self.use_next_frame:
            #     # pdb.set_trace()
            #     next_points, next_times = self.get_neighbor_sweep(index_or_info, curr_global2lidar=curr_global2lidar, curr_timestamp=curr_timestamp, flag=flag)
            #     if next_times is not None:
            #         sweep_points_list.append(next_points)
            #         sweep_times_list.append(next_times)
        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        # pdb.set_trace()
        points = self._remove_ego_pts(points)
        return points

    def _remove_ego_pts(self, points):
         # # 可以一定程度上抑制自车除的误检
         # 车宽1.7， 车长4.8
        width = 1.7
        long = 5.0
        if self.dataset_cfg.REF_CS_IS_LIDAR: # rfu
            ego_x = width*0.5
            ego_y = long*0.5
        else:
            ego_x = long*0.5
            ego_y = width*0.5
        
        mask = ~((np.abs(points[:, 0]) < ego_x) & (np.abs(points[:, 1]) < ego_y))
        points = points[mask]
        return points

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        output_dir = os.path.join(*os.path.split(result_path)[:-1])
        if not tracking:
            from .evaluation.detection.evaluate import MotovisEval
            nusc_eval = MotovisEval(
                gt_path = self.ann_file,
                config=self.det3d_eval_configs,
                result_path=result_path,
                output_dir=output_dir,
                verbose=True,
                ref_cs_is_lidar=self.REF_CS_IS_LIDAR
            )
            # pdb.set_trace()
            nusc_eval.main(render_curves=False)
            
            # record metrics
            # metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            metrics = json.load(open(os.path.join(output_dir, "metrics_summary.json"), 'r'))
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.class_names: # self.CLASSES:
                # import pdb;pdb.set_trace()
                if name == "generic_object" or name == "tricycle": continue
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        "{}/{}_AP_dist_{}".format(metric_prefix, name, k)
                    ] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail[
                        # "{}/{}".format(metric_prefix, self.ErrNameMapping[k])
                        "{}/{}".format(metric_prefix, OD_ErrNameMapping[k])
                    ] = val

            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]
        else:
            assert False
            from .evaluation.detection.tracking_evaluate import MotovisTrackingEval
            nusc_eval = MotovisTrackingEval(
                gt_path = self.ann_file,
                config=self.track3d_eval_configs,
                result_path=result_path,
                output_dir=output_dir,
                verbose=True
            )
            metrics = nusc_eval.main()

            # record metrics
            metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
            print(metrics)
            detail = dict()
            metric_prefix = f"{result_name}_NuScenes"
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        return detail
    
    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        from .evaluation.detection.data_class import OD_DefaultAttribute
        nusc_annos = {}
        mapped_class_names = self.class_names

        print("Start to convert detection format...")
        num_all = len(results)
        # pdb.set_trace()
        for sample_id in tqdm(range(num_all)):
            annos = []
            boxes = output_to_nusc_box(results[sample_id])
            extra_info = self.load_extra_info(self.infos[sample_id])
            self.infos[sample_id].update(extra_info)
            sample_token = self.infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.infos[sample_id],
                boxes,
                mapped_class_names,
                eval_configs=self.det3d_eval_configs,
                filter_with_cls_range=True,
                is_ref_lidar= self.REF_CS_IS_LIDAR
            )

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                    "generic_object",
                    "other"
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                        "van",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = OD_DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = OD_DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=str(box.token),
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

        nusc_submissions = {
            "meta": 'lidar',
            "results": nusc_annos,
        }
        os.makedirs(jsonfile_prefix, exist_ok=True)

        # mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = os.path.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        json.dump(nusc_submissions, open(res_path, 'w'), indent=2)
        return res_path

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        assert isinstance(results, list), "results must be a list"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = os.path.join(tmp_dir.name, "results")
        else:
            tmp_dir = None
        result_files = self._format_bbox(results, jsonfile_prefix, tracking)
        return result_files, tmp_dir

    def evaluation(self, det_annos, class_names, **kwargs):
        '''
            ref: http://192.168.22.225:8099/gongxinman/motovis_e2e_playground/blob/perception/projects/mmdet3d_plugin/datasets/motovis_e2e_dataset.py
            需要按照motovis的端到端规则, 写相应的测试代码.
        '''

        results_dict = dict()
        tracking = kwargs.get('tracking', False)
        output_path = kwargs['output_path']

        result_files, tmp_dir = self.format_results(
            det_annos, jsonfile_prefix=output_path, tracking=tracking
        )
        # result_files = os.path.join(output_path, "results_nusc.json")
        ret_dict = self._evaluate_single(
            result_files, tracking=tracking
        )

        results_dict.update(ret_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        # # print main metrics for recording
        metric_str = '\n'
        if "img_bbox_NuScenes/NDS" in results_dict:
            metric_str += f'mAP: {results_dict.get("img_bbox_NuScenes/mAP"):.4f}\n'
            metric_str += f'mATE: {results_dict.get("img_bbox_NuScenes/mATE"):.4f}\n'
            metric_str += f'mASE: {results_dict.get("img_bbox_NuScenes/mASE"):.4f}\n'
            metric_str += f'mAOE: {results_dict.get("img_bbox_NuScenes/mAOE"):.4f}\n' 
            metric_str += f'mAVE: {results_dict.get("img_bbox_NuScenes/mAVE"):.4f}\n' 
            metric_str += f'mAAE: {results_dict.get("img_bbox_NuScenes/mAAE"):.4f}\n' 
            metric_str += f'NDS: {results_dict.get("img_bbox_NuScenes/NDS"):.4f}\n\n'
        
        if "img_bbox_NuScenes/amota" in results_dict:
            metric_str += f'AMOTA: {results_dict["img_bbox_NuScenes/amota"]:.4f}\n' 
            metric_str += f'AMOTP: {results_dict["img_bbox_NuScenes/amotp"]:.4f}\n' 
            metric_str += f'RECALL: {results_dict["img_bbox_NuScenes/recall"]:.4f}\n' 
            metric_str += f'MOTAR: {results_dict["img_bbox_NuScenes/motar"]:.4f}\n' 
            metric_str += f'MOTA: {results_dict["img_bbox_NuScenes/mota"]:.4f}\n' 
            metric_str += f'MOTP: {results_dict["img_bbox_NuScenes/motp"]:.4f}\n' 
            metric_str += f'IDS: {results_dict["img_bbox_NuScenes/ids"]}\n\n' 
        
        # print(metric_str)
        return metric_str, results_dict


    def create_groundtruth_database(self, used_classes=None, max_sweeps=10, save_path=None):
        import torch
        import pdb
        save_path = self.root_path if save_path is None else self.root_path / save_path
        database_save_path = save_path / f'gt_database_{max_sweeps}sweeps'
        db_info_save_path = save_path / f'motovis_dbinfos_{max_sweeps}sweeps.pkl'
        
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        bar = tqdm(range(len(self.infos)))
        scene_list = set()
        for idx in bar:
            sample_idx = idx
            info = self.infos[idx]
            
            scene_token = info['scene_token']
            scene_list.add(scene_token)
            bar.set_description(f'{scene_token}:{len(scene_list)}')
            # if len(scene_list)<408:
            #     continue
            database_save_path_scene = database_save_path / scene_token
            database_save_path_scene.mkdir(exist_ok=True)
            # pdb.set_trace()
            info = self.load_extra_info(info)

            points = self.get_lidar_with_sweeps(sample_idx, max_sweeps=max_sweeps)
            
            info = self._remap_class_names(info)
            gt = self.get_gt(info)
            gt_boxes = gt['gt_boxes']
            gt_names = gt['gt_names']
            
            if gt_boxes.shape[0]==0:
                # pdb.set_trace()
                continue

            # pdb.set_trace()            
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
            # pdb.set_trace()
            for i in range(gt_boxes.shape[0]):
                if (used_classes is None) or gt_names[i] in used_classes:
                    filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                    filepath = database_save_path_scene / filename
                    gt_points = points[box_idxs_of_pts == i]
                    
                    num_gt_pts = gt_points.shape[0]
                    if num_gt_pts<4:
                        continue
                    gt_points[:, :3] -= gt_boxes[i, :3]
                    
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    # pdb.set_trace()
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
    
    # @staticmethod
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        This is the Waymo version. Further refactor for custom dataset later.


        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:
            annos: list of detection results
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'label': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 9])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):

            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            num_perds = pred_scores.shape[0]
            if num_perds == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['label'] = pred_labels - 1
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            # pdb.set_trace()
            single_pred_dict = generate_single_sample_dict(box_dict)
            # single_pred_dict['sequence_name'] = batch_dict['sequence_name'][index]
            single_pred_dict['scene_token'] = batch_dict['metadata'][index]['scene_token']
            single_pred_dict['token'] = batch_dict['metadata'][index]['token']
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            # single_pred_dict['pose'] = batch_dict['pose'][index]
            single_pred_dict['pose'] = None
            single_pred_dict['key_frame'] = batch_dict['key_frame'][index]
            # pdb.set_trace()
            annos.append(single_pred_dict)
        return annos
    

def output_to_nusc_box(detection, threshold=None):
    # box3d = detection["boxes_3d"]
    # scores = detection["scores_3d"].numpy()
    # labels = detection["labels_3d"].numpy()
    box3d = detection['boxes_lidar']
    scores = detection['score']
    labels = detection['label']
    # labels = detection['name']
    
    if "instance_ids" in detection:
        ids = detection["instance_ids"]  # .numpy()
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"] >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    box_gravity_center = box3d[..., :3].copy()
    box_dims = box3d[..., 3:6].copy()
    nus_box_dims = box_dims[..., [1, 0, 2]]
    box_yaw = box3d[..., 6].copy()

    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2
    with_velocity = box3d.shape[-1]>=9
    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d[i, 7:9], 0.0)
        else:
            velocity = (0.0, 0.0, 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(
    info,
    boxes,
    classes,
    eval_configs,
    filter_with_cls_range=True,
    is_ref_lidar=True
):
    box_list = []
    for i, box in enumerate(boxes):
        if is_ref_lidar:
            # Move box to ego vehicle coord system
            lidar2ego_rotation = R.from_matrix(info['lidar2ego_rotation']).as_quat()
            lidar2ego_rotation = pyquaternion.Quaternion(np.array([lidar2ego_rotation[3], lidar2ego_rotation[0], lidar2ego_rotation[1], lidar2ego_rotation[2]]))
            # import pdb;pdb.set_trace()
            # lidar2ego_rotation = pyquaternion.Quaternion(matrix=info['lidar2ego_rotation'])
            box.rotate(lidar2ego_rotation)
            box.translate(np.array(info["lidar2ego_translation"]))

        # filter det in ego.
        if filter_with_cls_range:
            cls_range_map = eval_configs.class_range
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                continue
        # Move box to global coord system
        # ego2global_rotation = R.from_matrix(info['ego2global_rotation']).as_quat()
        # print(info['ego2global_rotation'])
        
        # 获取旋转矩阵
        # rotation_matrix = np.array(info['ego2global_rotation'])
        # import pdb;pdb.set_trace()
        # print(rotation_matrix @ rotation_matrix.T)
        ego2global_rotation = R.from_matrix(info['ego2global_rotation']).as_quat()
        ego2global_rotation = pyquaternion.Quaternion(np.array([ego2global_rotation[3], ego2global_rotation[0], ego2global_rotation[1], ego2global_rotation[2]]))
        
        # 检查并正交化矩阵
        # if not np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=1e-8):
        #     import pdb;pdb.set_trace()
        #     print(f"Warning: Rotation matrix is not orthogonal, orthogonalizing...")
        #     print(rotation_matrix)
        #     rotation_matrix = orth(rotation_matrix)  # 使用正交化
        #     print('-----',rotation_matrix)
        
        # ego2global_rotation = pyquaternion.Quaternion(matrix=rotation_matrix)
    
        # ego2global_rotation = pyquaternion.Quaternion(matrix=info['ego2global_rotation'])
        box.rotate(ego2global_rotation)
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list

class OntimeDetectionDataset(MotovisDetectionDataset):
    def load_extra_info(self, info):
        # pdb.set_trace()
        if 'filepath' in info.keys():
            filepath = info['filepath']
            extra_info = pickle.load(
                open(os.path.join(self.root_path, self.extra_path, filepath), 'rb')
            )
            extra_info.update(info)
            return extra_info
        return info


    def __getitem__(self, index):
        # print('index:', index)
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        info = self.load_extra_info(info)
        
        points = self.get_lidar_with_sweeps(info, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        # pdb.set_trace()
        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            # 'key_frame': info['key_frame']
            'key_frame': info.get('key_frame', False)
        }
        if 'scene_token' in info.keys():
            input_dict['metadata']['scene_token'] = info['scene_token']

        input_dict.update(self.get_gt(info))
        
        input_dict = self._remap_class_names(input_dict)
        data_dict = self.prepare_data(data_dict=input_dict)
        # pdb.set_trace()
        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in data_dict:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        # if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
        #     data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        return data_dict
    

    def create_groundtruth_database(self, used_classes=None, max_sweeps=1, save_path=None, num_worker=-1):
    
        import torch
        import threading
        import queue
        import random
        import pdb
        import concurrent.futures

        # print(f"CUDA 是否可用: {torch.cuda.is_available()}")
        # # 查看可用 GPU 数量
        # print(f"可用 GPU 数量: {torch.cuda.device_count()}")
        # # 查看当前默认 CUDA 设备
        # print(f"当前默认设备: {torch.cuda.current_device()}")
        random.seed(42)
        gpu_num = torch.cuda.device_count()

        # pdb.set_trace()
        save_path = self.root_path if save_path is None else self.root_path / save_path
        database_save_path = save_path / f'gt_database_{max_sweeps}sweeps'
        db_info_save_path = save_path / f'motovis_dbinfos_{max_sweeps}sweeps.pkl'
        
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}


        thread_local = threading.local()

        thread_gpu_dict = dict()

        def init_thread_gpu(gpu_id):
            """
            初始化线程的 GPU 设备上下文
            :param gpu_id: 线程要使用的 GPU 编号
            """
            t_n = threading.current_thread().name
            if t_n not in thread_gpu_dict.keys():
                # 设置当前线程的 CUDA 设备
                torch.cuda.set_device(gpu_id)
                # 将 GPU ID 存入线程局部存储，方便后续使用
                thread_local.gpu_id = gpu_id
                # print(f"线程 {threading.current_thread().name} 绑定到 GPU {gpu_id}")
                thread_gpu_dict[t_n] = gpu_id
            else:
                gpu_id = thread_gpu_dict[t_n]
            return gpu_id

        # pdb.set_trace()
        def process_one(idx):
            sample_idx = idx
            info = self.infos[idx]
            
            scene_token = info['scene_token']
            database_save_path_scene = database_save_path / scene_token
            database_save_path_scene.mkdir(exist_ok=True)
         
            info.update(self.load_extra_info(info))

            points = self.get_lidar_with_sweeps(sample_idx, max_sweeps=max_sweeps)
            
            info = self._remap_class_names(info)
            gt = self.get_gt(info)
            gt_boxes = gt['gt_boxes']
            gt_names = gt['gt_names']
      
            if gt_boxes.shape[0]==0:
                # pdb.set_trace()
                return 
            # pdb.set_trace()   
            gpu_id = idx%gpu_num
            gpu_id = init_thread_gpu(gpu_id) 
            cuda_id = f'cuda:{gpu_id}'
   
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().to(cuda_id),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().to(cuda_id)
            ).long().squeeze(dim=0).cpu().numpy()
            # pdb.set_trace()
            idx = list(range(gt_boxes.shape[0]))
            random.shuffle(idx)

            max_num_per_cls = 2
            cls2cnt = dict()

            # pdb.set_trace()
            # for i in range(gt_boxes.shape[0]):
            for i in idx:
                if (used_classes is None) or gt_names[i] in used_classes:
                    cnt =  cls2cnt.get(gt_names[i], 0)
                    if cnt >= max_num_per_cls:
                        continue
                    filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                    filepath = database_save_path_scene / filename
                    gt_points = points[box_idxs_of_pts == i]
                    
                    num_gt_pts = gt_points.shape[0]
                    if num_gt_pts<4:
                        continue
                    # pdb.set_trace()
                    cls2cnt[gt_names[i]] = cnt + 1
                    gt_points[:, :3] -= gt_boxes[i, :3]
                    
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    # pdb.set_trace()
                    if gt_names[i] not in all_db_infos:
                        all_db_infos[gt_names[i]] = queue.Queue()

                    all_db_infos[gt_names[i]].put(db_info)
        
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker) as executor:
            # 提交所有任务，并记录任务对应的索引
            future_to_index = {
                executor.submit(process_one, idx): idx 
                for idx in range(0, len(self.infos), 2)
            }
            
            # 创建进度条，总数为任务数量
            progress_bar = tqdm(total=len(future_to_index), desc="处理进度", unit="个")
            
            # 遍历完成的任务，更新进度条和结果列表
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    # 获取任务结果
                    future.result()
                except Exception as e:
                    # 捕获任务执行中的异常
                    print(f"\n处理索引 {idx} 的元素时出错: {e}")
                # 更新进度条
                progress_bar.update(1)
            
            # 关闭进度条
            progress_bar.close()

        for k in list(all_db_infos.keys()):
            v = all_db_infos.pop(k)
            num_v = v.qsize()
            v = [v.get()  for i in range(num_v)]
            all_db_infos[k] = v
            print('Database %s: %d' % (k, len(v)))
            
        print('saving ', db_info_save_path)
        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
        print('Done')
