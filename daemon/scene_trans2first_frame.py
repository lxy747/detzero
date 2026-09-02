import pickle
from detzero_utils import common_utils
import numpy as np
import sys
import os
from tqdm import tqdm
import pdb


def split_extra_info(infos, extra_path, root_dir, worker=0):
    base_keys = ['token', 'timestamp', 'scene_token', 'key_frame', 'gt_names', 'lidar_path', 'lidar2global']
    base_infos = []
    # pdb.set_trace()
    for info in tqdm(infos, desc='split_extra_info'):
        base_dict = dict()
        for k, v in info.items():
            if k in base_keys:
                base_dict[k] = v
        extra_pkl = os.path.join(extra_path, base_dict['scene_token'], base_dict['token'] + '.pkl')
        base_dict['filepath'] = extra_pkl
        # pdb.set_trace()
        base_infos.append(base_dict)
    return base_infos

    def dump_pkls(sub_infos, sub_base_infos, desc=None):
        for sub_info, sub_base_info in tqdm(zip(sub_infos, sub_base_infos), desc=desc, total=len(sub_infos)):
            filepath=sub_base_info['filepath']
            extra_pkl = os.path.join(root_dir, filepath)
            os.makedirs(os.path.split(extra_pkl)[0], exist_ok=True)
            with open(extra_pkl, 'wb') as fp:
                pickle.dump(sub_info, fp)
    if worker<2:
        dump_pkls(infos, base_infos, desc='save_extra_infos')
    else:
        import threading
        num = len(infos)
        bsz = num//worker
        threads = []
        for i in range(worker):
            desc = 'save_extra_infos: thread {}'.format(i)
            id1 = i * bsz
            if (i+1) == worker:
                id2 = num
            else:
                id2 = id1 + bsz
            thread = threading.Thread(target=dump_pkls, args=(infos[id1:id2], base_infos[id1:id2], desc))
            thread.start()
            threads.append(thread)
        for t in threads:
            t.join()
    return base_infos


def move_gt(infos):
    '''
        将gt从anns里面move出来
    '''
    for info in tqdm(infos, desc='move gt'):

        if 'anns' in info.keys():
            anns = info.pop('anns')
            anns['gt_velocity_3d'] = np.array(anns['gt_velocity_3d'])
            anns['gt_velocity'] = anns['gt_velocity_3d'][:, :2]
            for k in anns.keys():
                if k in ['gt_names', 'num_lidar_pts', 'gt_boxes']:
                    anns[k] = np.array(anns[k])
            info.update(anns)
            info['key_frame'] = len(anns)>0
        elif 'key_frame' not in info.keys():
            info['key_frame'] = False

        
if __name__=='__main__':
    # src_pkl = sys.argv[1]
    # dst_pkl = sys.argv[2]

    # src_pkl='/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_10clips_2000samples.pkl'
    # dst_pkl='/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_10clips_2000samples_world_clip.pkl'

    # src_pkl='/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_1001clips_200070samples.pkl'
    # dst_pkl='/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_1001clips_200070samples_world_clip.pkl'

    # src_pkl='/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_nuscenes_3179clips_635800samples.pkl'
    # dst_pkl='/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_nuscenes_3179clips_635800samples_world_clip.pkl'

    src_pkl='/mnt/cfs/e2e/datasets/t4/pkls_navsim/t4_navsim_2clips_1174samples.pkl'
    dst_pkl='/mnt/cfs/e2e/datasets/t4/pkls_navsim/t4_navsim_2clips_1174samples_world_clip.pkl'

    do_move_gt=True

    do_split=False


    print('loading data ...')
    data = pickle.load(open(src_pkl, 'rb'))
    # pdb.set_trace()
    if isinstance(data, list):
        common_utils.clip_trans2first_frame(data)
        if do_move_gt:
            move_gt(data)
    else:
        common_utils.clip_trans2first_frame(data['infos'])
        if do_move_gt:
            move_gt(data['infos'])

    pdb.set_trace()

    if do_split:
        root_path, extra_path = os.path.split(dst_pkl)
        base_infos = split_extra_info(data['infos'], extra_path=extra_path[:-4], root_dir=root_path, worker=4)
        data['infos'] = base_infos

        print('dumping data ...')
        pickle.dump(data, open(
            os.path.join(root_path, dst_pkl), 'wb'))
    else:
        print('dumping data ...')
        pickle.dump(data, open(dst_pkl, 'wb'))
    print('done')
    