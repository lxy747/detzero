import pickle, os
from detzero_utils import common_utils
from tqdm import tqdm
import datetime
import numpy as np
import pdb

def merge(data_list: list):
    '''
        将多个pkl合并为一个, 并去掉重复的clip
    '''
    infos = []
    for data in data_list:
        if isinstance(data, dict):
            infos.extend(data['infos'])
        else:
            infos.extend(data)
    clip_name = []
    infos_new = []
    cnt, id1 = len(infos), 0
    while id1<cnt:
        scene_token = infos[id1]['scene_token']
        if scene_token in clip_name:
            id1 += 1
            continue
        clip_name.append(scene_token)
        infos_new.append(infos[id1])
        id1 += 1
        while id1<cnt:
            if infos[id1]['scene_token'] == clip_name[-1]:
                infos_new.append(infos[id1])
                id1 += 1
            else:
                break
    return dict(
        metadata=dict(clip_name=clip_name),
        infos = infos_new
    )

def add_some_key_to_pkl(infos, data_root, keys=[], desc='', num_worker=0):
    if num_worker < 2:
        for info in tqdm(infos, desc='add keys. '+desc):
            not_need_add = True
            for k in keys:
                if k not in info.keys():
                    # pdb.set_trace()
                    not_need_add = False
                    break
            if not_need_add:
                continue

            filepath = os.path.join(data_root,  info['filepath'])
            extra_info  = pickle.load(open(filepath, 'rb'))
            # pdb.set_trace()
            for k in keys:
                if 'gt_names' == k:
                    info[k] = list(set(extra_info[k]))
                else:
                    if isinstance(extra_info[k], np.ndarray):
                        info[k] = extra_info[k].tolist()
                    else:    
                        info[k] = extra_info[k]
        return
    
    import threading
    num = len(infos)
    step = num // num_worker
    ts = []
    for i in range(num_worker):
        id1 = i*step
        id2 = id1 + step
        if id2 > num: id2 = num
        desc = 'Thread: {}'.format(i)
        t = threading.Thread(target=add_some_key_to_pkl, args=(infos[id1:id2], data_root, keys, desc, 0), daemon=True)
        t.start()
        ts.append(t)

    [t.join() for t in ts]

    print('\n\n\n\n')

        

def load_pkl(pkl_fn):
    return pickle.load(open(pkl_fn, 'rb'))

def dump_pkl(data, pkl_fn):
    print(pkl_fn, 'dump ...')
    pickle.dump(data, open(pkl_fn, 'wb'))
    print('Done')

def formattime(now: datetime.datetime):
    year = '{}'.format(now.year)
    month = '{}'.format(now.month).rjust(2, '0')
    day = '{}'.format(now.day).rjust(2, '0')
    t = year+month+day
    return t
    
    


if __name__=='__main__':
    data_root = '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes'
    src_pkls = [
        '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes/E2E_citynoa_20250804_100_nuscenes_100clips_20000samples_split_train.pkl',
        # '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes/E2E_citynoa_20250815_1002_nuscenes_1001clips_200070samples_split_train.pkl',
        # '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes/E2E_citynoa_20250916_3179_nuscenes_3139clips_627800samples_split_train.pkl',
        # '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes/E2E_citynoa_20251030_2021_nuscenes_2021clips_404198samples_split_train.pkl',

    ]

    now = datetime.datetime.now()
    now = formattime(now)
    # pdb.set_trace()
    dst_pkl = '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes/ontime_merge_{}_clip_num_clips_sam_num_samples_train.pkl'.format(now)
    
    data_list = []
    for pkl in src_pkls:
        data_list.append(load_pkl(pkl))
    # pdb.set_trace()
    data = merge(data_list)

    add_some_key_to_pkl(data['infos'], data_root, keys=['gt_names', 'lidar_path', 'lidar2global'], num_worker=0)

    clip_num =str(len(data['metadata']['clip_name']))
    sam_num = str(len(data['infos']))

    dir, fn = os.path.split(dst_pkl)
    fn = fn.replace('clip_num_', clip_num).replace('sam_num_', sam_num)
    dst_pkl = os.path.join(dir, fn)
    dump_pkl(data, dst_pkl)