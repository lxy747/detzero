
import pickle
import os
from .motovis_dataset import remap_class_names, MotovisDetectionDataset, load_lidar_points
import numpy as np
from pathlib import Path
import copy
import pdb

class MotovisInferDataset(MotovisDetectionDataset):
    '''
        diffubox中用
    '''
    def __init__(self, dataset_cfg, class_names, root_path=None, logger=None, **kwargs):
        super().__init__(dataset_cfg, class_names, False, root_path, logger)

    def include_nuscenes_data(self, mode):
        # pdb.set_trace()
        nuscenes_infos = []
        info_paths = self.dataset_cfg.INFO_PATH
        for info_path in info_paths:
            if not os.path.isabs(info_path):
                info_path = os.path.join(self.root_path, info_path)
            if not os.path.exists(info_path):
                continue
            self.ann_file=info_path
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                if isinstance(infos, dict):
                    infos = infos['infos']
                [remap_class_names(info) for info in infos]
                nuscenes_infos.extend(infos)
        self.infos.extend(nuscenes_infos)

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        info = copy.deepcopy(self.infos[index])
        # info.update(self.load_extra_info(info))
        # points = self.get_lidar_with_sweeps(info, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        # points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        if 'lidar_path' in info.keys():
            lidar_path = self.root_path / info['lidar_path']
        else:    
            lidar_path = self.root_path / info['data_path']
        # points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        # bench2drive点云只有xyz
        points = load_lidar_points(lidar_path, dim=self.pt_dim)
        # pdb.set_trace()
        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']},
            'key_frame': info['key_frame']
        }
        if 'scene_token' in info.keys():
            input_dict['metadata']['scene_token'] = info['scene_token']
        
        input_dict.update(self.get_dt_boxes(info))
        if info['key_frame']>0:
            input_dict.update(self.get_gt_boxes(info))
            input_dict = self._remap_class_names(input_dict)
     
        data_dict = self.prepare_data(data_dict=input_dict)
        # pdb.set_trace()
        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in data_dict:
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def get_gt_boxes(self, info):
        input_dict = dict()
        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            gt_boxes = info['gt_boxes']
            if ('gt_velocity' in info.keys()) and (self.dataset_cfg.PRED_VELOCITY):
                gt_boxes = np.concatenate([gt_boxes, info['gt_velocity']], axis=1).copy()

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': gt_boxes if mask is None else gt_boxes[mask]
            })
            
        return input_dict
    
    def get_dt_boxes(self, info):
        input_dict = dict()
        if 'boxes_lidar' in info:
            input_dict['boxes_lidar'] = info['boxes_lidar']
            input_dict['name'] = info['name']
            input_dict['score'] = info['score']
            input_dict['dt_instance_inds'] = info.get('dt_instance_inds', None)
        return input_dict
    
    def update_dt_boxes(self, index: int, dt_info: dict):
        metadata = dt_info.pop('metadata')

        assert metadata['token'] == self.infos[index]['token']
        self.infos[index].update(dt_info)
    
    def save_infos(self, tag='diffubox', pkl_path=None):
        if pkl_path is None:
            pkl_path = str(self.ann_file)
            if tag is not None:
                pkl_path = pkl_path[:-4] + '_' + tag + '.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.infos, f)


class MotovisInferOntimeDataset(MotovisInferDataset):
    def load_extra_info(self, info):
        filepath = os.path.join(self.root_path, 'pkls',  info['filepath'])
        # pdb.set_trace()
        ext_info = pickle.load(open(filepath, 'rb'))
        for i in range(len(ext_info['sweeps'])):
            lidar_path = ext_info['sweeps'][i]['data_path']
            lidar_path = lidar_path.split('/')
            lidar_path.insert(1, 'samples')
            lidar_path = os.path.join(*lidar_path)
            ext_info['sweeps'][i]['data_path'] = lidar_path
        return ext_info