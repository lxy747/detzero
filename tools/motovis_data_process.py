import argparse
import pickle
import os
import numpy as np
import torch
from tqdm import tqdm
from detzero_det.datasets.motovis.motovis_dataset import load_lidar_points
from detzero_utils.ops.roiaware_pool3d import roiaware_pool3d_utils
import pdb

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_root', type=str, default=None, help='the dir for source data')
    parser.add_argument('--src_pkl', type=str, default=None, help='')
    parser.add_argument('--dst_pkl', type=str, default=None, help='')
    parser.add_argument('--workers', type=int, default=0, help='')
    parser.add_argument('--mode', type=str, default='calc_num_lidar_pts', help='')
    args = parser.parse_args()
    return args

def calc_num_lidar_pts(points: np.ndarray, boxes3d: np.ndarray):
    '''
        points: shape [N, 3]
        boxes3d: shape [N, 7+c]
    '''
    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(boxes3d[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

    num_lidar_pts = np.zeros(boxes3d.shape[0], dtype=np.int32)
    for i in range(boxes3d.shape[0]):
        num = np.sum(box_idxs_of_pts==i)
        num_lidar_pts[i] = num
    return num_lidar_pts

def calc_num_lidar_pts_process(src_pkl, dst_pkl, data_root, workers=0):
    def _process(infos, id1=0, id2=-1, tile=None):
        if id2<0: id2 = len(infos)
        ids = list(range(id1, id2))
        for i in tqdm(ids, desc=tile):
            if not infos[i]['key_frame']:
                continue
            gt_boxes = infos[i]['gt_boxes']
     
            lidar_path = infos[i]['lidar_path']
            lidar_path = os.path.join(data_root, lidar_path)
            points = load_lidar_points(lidar_path)
            num_lidar_pts = calc_num_lidar_pts(points, gt_boxes)
            infos[i]['num_lidar_pts'] = num_lidar_pts

    data = load_pkl(src_pkl)
    infos = data['infos']
    if workers<2:
        _process(infos, 0, -1)
    else:
        num_info = len(infos)
        import threading
        bsz = num_info // workers
        tasks = []
        for i in range(workers):
            id1 = i * bsz
            id2 = id1 + bsz
            if i+1 == workers:
                id2 = num_info
            proc = threading.Thread(target=_process, args=(infos, id1, id2, f'thread:{i}'))
            proc.start()
            tasks.append(proc)
        for t in tasks:
            t.join()

    dump_pkl(data, dst_pkl)


def load_pkl(pkl_file):
    data = pickle.load(open(pkl_file, 'rb'))
    return data

def dump_pkl(data, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)
        f.close()

if __name__ == '__main__':
    '''
        计算3d gt框内lidar点个数
    '''
    args = parse_config()
    data_root = args.data_root
    src_pkl = args.src_pkl
    dst_pkl = args.dst_pkl
    workers = args.workers
    mode = args.mode
    if mode == 'calc_num_lidar_pts':
        '''
            e.g. 
                python motovis_data_process.py --data_root /data/dataset/bench2drive/base \
                    --src_pkl /data/dataset/bench2drive/infos/base/b2d_infos_merge_all_train03.pkl \
                    --dst_pkl /data/dataset/bench2drive/infos/base/b2d_infos_merge_all_train03_with_gt_num_pts.pkl \
                    --workers 2 --mode calc_num_lidar_pts
            
            发现多线程并不能提太多速度。worker给0也可
        '''
        calc_num_lidar_pts_process(src_pkl, dst_pkl, data_root, workers)
    
    print(mode, 'done')