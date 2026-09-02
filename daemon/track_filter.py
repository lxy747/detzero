import numpy as np
import pickle
# import cv2
from detzero_utils.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
import torch
import pdb
def load_tk_data(pkl_file):
    return pickle.load(open(pkl_file, 'rb'))

def dump_tk_data(tk_data, pkl_file):
    pickle.dump(tk_data, open(pkl_file, 'wb'))




def openclose_op(hits):
    n = hits.shape[0]
    tmp_hits = hits.copy()
    for i in range(1, n-1):
        if hits[i]>0:
            tmp_hits[i+1] = 1
            tmp_hits[i-1] = 1
    # close
    tmp_hits2 = tmp_hits.copy()
    for i in range(1, n-1):
        if tmp_hits[i]==0:
            tmp_hits2[i+1] = 0
            tmp_hits2[i-1] = 0
    tmp_hits2[0] = hits[0]
    tmp_hits2[n-1] = hits[n-1]
    return tmp_hits2


def _create_mask(track):
    hit = track['hit']
    hit = openclose_op(hit)
    selected = hit > 0
    # selected = np.logical_and(selected, track['num_points']>1)
    return selected

def tk_filter_clip(tracks, with_gt=False):
    keys = [
        'boxes_global', 'name', 'score', 'sample_idx', 
        'hit', 'num_points', 'obj_ids', 'pose', 
        ]
    obj_ids = list(tracks.keys())
    for obj_id in obj_ids:
        if 'track' in tracks[obj_id].keys():
            track = tracks[obj_id]['track']
        else:
            track = tracks[obj_id]
        selected = _create_mask(track)
        n = selected.sum()
        # if n < selected.shape[0]:
        #     pdb.set_trace()
        if n<1:
            tracks.pop(obj_id)
            continue
        if n == selected.shape[0]:
            continue
        # pdb.set_trace()

        for k in keys:
            track[k] = track[k][selected]

        if 'lidar_path' in track.keys():
            track['lidar_path'] = [track['lidar_path'][i] for i, v in enumerate(selected) if v ==True]
        if with_gt:
            # gt dict_keys(['gt_boxes_global', 'gt_boxes_lidar', 'name', 'obj_ids', 'sample_idx'])
            # gt = track_data['gt']
            if 'iou' in track.keys():
                track['iou'] = track['iou'][selected]
    # pdb.set_trace()
    tk_filter_pre(tracks)

def track2frame(tracks):
    frame_infos = dict()
    traj_len = dict()
    obj_ids = list(tracks.keys())
    for obj_id in obj_ids:
        if 'track' in tracks[obj_id].keys():
            track = tracks[obj_id]['track']
        else:
            track = tracks[obj_id]
        
        t_len = sum(track['hit']>0)
        traj_len[obj_id] = t_len

        sample_idxs=track['sample_idx']
        for i, sample_idx in enumerate(sample_idxs):
            if sample_idx not in frame_infos.keys():
                info = dict(
                    boxes_global=[],
                    score=[],
                    obj_id=[],
                    idx_in_traj=[],
                    # pose=track['pose'][i]
                )
                frame_infos[sample_idx] = info
            # pdb.set_trace()
            # frame_infos[sample_idx]['pose'].append(track['pose'][i])
            frame_infos[sample_idx]['boxes_global'].append(track['boxes_global'][i, :])
            frame_infos[sample_idx]['score'].append(track['score'][i])
            frame_infos[sample_idx]['obj_id'].append(obj_id)
            frame_infos[sample_idx]['idx_in_traj'].append(i)
     
    return frame_infos, traj_len

# from detzero_track.utils.transform_utils import transform_boxes3d

def create_iou_key(obj1, i, obj2, j):
    '''
        标识 轨迹obj1在第i帧与obj2在第j帧相交
    '''
    if obj1<=obj2:
        return (obj1, i, obj2, j)
    return (obj2, j, obj1, i)

def calc_frame_iou(frame_infos, traj_len, iou_thr=0.2):
   
    traj_cnt = list(traj_len.items())
    # pdb.set_trace()
    traj_cnt = sorted(traj_cnt, key=lambda x: x[1])

    obj2id = dict()

    obj_keys = [k[0] for k in traj_cnt]
    for i, k in enumerate(obj_keys):
        obj2id[k] = i
    map_trajs = [-1] * len(obj_keys)
    map_ious = [-1] * len(obj_keys)
    
    for sample_idx in frame_infos.keys():
        boxes_global = np.stack(frame_infos[sample_idx]['boxes_global'], axis=0)
        iou = boxes_iou_bev(
                    torch.from_numpy(boxes_global[:, :7]).float().cuda(),
                    torch.from_numpy(boxes_global[:, :7]).float().cuda()
                ).cpu().numpy()

        # 生成下三角掩码（True = 下三角位置）
        mask = np.tril(np.ones_like(iou, dtype=np.bool_))
        iou[mask] = 0.0
        row_idx, col_idx = np.where(iou > iou_thr)
        # if '2654.400' in sample_idx:
        #     pdb.set_trace()
        frame_infos[sample_idx]['iou_dict'] = dict()
        for i, idx1 in enumerate(row_idx):
            idx2 = col_idx[i]
            obj1 = frame_infos[sample_idx]['obj_id'][idx1]
            obj2 = frame_infos[sample_idx]['obj_id'][idx2]

            iou_v = iou[idx1, idx2]

            a = obj2id[obj1]
            b = obj2id[obj2]

            if a > b:
                a, b = b, a
            if iou_v > map_ious[a]:
                map_trajs[a] = b
                map_ious[a] = iou_v

            idx_in_traj1 = frame_infos[sample_idx]['idx_in_traj'][idx1]
            idx_in_traj2 = frame_infos[sample_idx]['idx_in_traj'][idx2]
            frame_infos[sample_idx]['iou_dict'][create_iou_key(obj1, idx_in_traj1, obj2, idx_in_traj2)] = iou_v
           

    
    return obj_keys, map_trajs

def merge2track_all(tracks, obj_keys, map_trajs, frame_infos):
    for obj1, obj2 in enumerate(map_trajs):
        if obj2 < 0:
            continue
        merge2track(tracks, obj_keys[obj2], obj_keys[obj1],frame_infos)

    keys = [
        'name', 'score', 'sample_idx', 'hit', 'num_points', 'pose', 'lidar_path', 'obj_ids'
        ]
    objs = list(tracks.keys())
    for obj_id in objs:
        if 'track' in tracks[obj_id].keys():
            track = tracks[obj_id]['track']
        else:
            track = tracks[obj_id]
        name = set(track['name'])
        if len(name)>1:
            # 投票判断类型
            name_cnt = dict()
            for n in track['name']:
                name_cnt[n] = name_cnt.get(n, 0) + 1
            final_name=track['name'][0]
            cnt = name_cnt[final_name]
            for k, v in name_cnt.items():
                if cnt<v:
                    final_name = k
                    cnt = v
            track['name'] = [final_name] * len(track['name'])
        track['boxes_global'] = np.stack(track['boxes_global'], axis=0)
        for k in keys:
            track[k] = np.array(track[k])
        if 'iou' in track.keys():
            track['iou'] = np.array(track['iou'])


def merge2track(tracks, base_id, ref_id, frame_infos):
    '''
        轨迹拼接、去重
    '''
    keys = [
        'boxes_global', 'name', 'score', 'sample_idx', 'hit', 'num_points', 'pose', 'lidar_path', 'obj_ids'
        ]
    
    has_track = 'track' in tracks[base_id].keys()
    with_gt = 'iou' in tracks[base_id].keys()
    def add__(track_out, track, id):
        for k in keys[:-1]:
            track_out[k].append(track[k][id])
        track_out['obj_ids'].append(base_id)
        if with_gt:
            track_out['iou'].append(track['iou'][id])

    track_out = dict()
    for k in keys:
        track_out[k] = []
    
    # 1. pop 目标key
    track1 = tracks[base_id]
    track2 = tracks[ref_id]
    
    if has_track:
        track1 = track1['track']
        track2 = track2['track']
 
    # 2. 根据sample_idx合并两个轨迹
    sample_idx1 = track1['sample_idx']
    sample_idx2 = track2['sample_idx']
    i = j = 0
    len1, len2 = len(sample_idx1), len(sample_idx2)

    mask2 = np.ones((len2,), dtype=np.bool_)
    # 双指针遍历，取小的放结果
    while i < len1 and j < len2:
        if sample_idx1[i] < sample_idx2[j]:
            i += 1
        elif sample_idx1[i] > sample_idx2[j]:
            j += 1
        else:
            iou_key = create_iou_key(base_id, i, ref_id, j)
            if iou_key in frame_infos[sample_idx1[i]]['iou_dict'].keys():
                mask2[j] = False
            i +=1
            j +=1
    valid_sum = mask2.sum()
    if valid_sum == 0:
        tracks.pop(ref_id)
        return track2
    if valid_sum < len2:
        for k in track2.keys():
            if isinstance(track2[k], np.ndarray):
                track2[k] = track2[k][mask2]
    return tracks

 

def merge2track_bak(tracks, base_id, ref_id, frame_infos):
    '''
        轨迹拼接、去重
    '''
    keys = [
        'boxes_global', 'name', 'score', 'sample_idx', 'hit', 'num_points', 'pose', 'lidar_path', 'obj_ids'
        ]
    
    has_track = 'track' in tracks[base_id].keys()
    with_gt = 'iou' in tracks[base_id].keys()
    def add__(track_out, track, id):
        for k in keys[:-1]:
            track_out[k].append(track[k][id])
        track_out['obj_ids'].append(base_id)
        if with_gt:
            track_out['iou'].append(track['iou'][id])

    track_out = dict()
    for k in keys:
        track_out[k] = []
    
    # 1. pop 目标key
    track1 = tracks.pop(base_id)
    track2 = tracks.pop(ref_id)
    
    if has_track:
        track1 = track1['track']
        track2 = track2['track']
    # if base_id==446:
    #     pdb.set_trace()
    if track1['state'] == track2['state']:
        track_out['state'] = track1['state']
        # pdb.set_trace()
        # 2. 根据sample_idx合并两个轨迹
        sample_idx1 = track1['sample_idx']
        sample_idx2 = track2['sample_idx']
        i = j = 0
        len1, len2 = len(sample_idx1), len(sample_idx2)

        # 双指针遍历，取小的放结果
        while i < len1 and j < len2:
            if sample_idx1[i] < sample_idx2[j]:
                add__(track_out, track1, i)
                i += 1
            elif sample_idx1[i] > sample_idx2[j]:
                add__(track_out, track2, j)
                j += 1
            else:
                # if base_id==446 and '2654.400' in sample_idx1[i]:
                #     pdb.set_trace()
                iou_key = create_iou_key(base_id, i, ref_id, j)
                if iou_key in frame_infos[sample_idx1[i]]['iou_dict'].keys():
                    # pdb.set_trace()
                    hit1 = min(track1['hit'][i], 1)
                    hit2 = min(track2['hit'][j], 1)
                    score1 = track1['score'][i]
                    score2 = track2['score'][j]
                    if (score1 + hit1) < (score2 + hit2):
                        add__(track_out, track2, j)
                    else:
                        add__(track_out, track1, i)
                else:
                    # pdb.set_trace()
                    add__(track_out, track1, i)
                i +=1
                j +=1

        if i < len1:
            for id in range(i, len1):
                add__(track_out, track1, id)
        elif j < len2:
            for id in range(j, len2):
                add__(track_out, track2, id)
    else:
        # 如果一个目标动，一个目标静止，不做合并，舍弃ref_id
        track_out = track1

    # 3. 还原track
    if has_track:
        tracks[base_id] = dict(track = track_out) 
    else:
        tracks[base_id] = track_out
  
    return tracks

 

def tk_filter_pre(tracks):
    '''
        轨迹合并、过滤重复轨迹
    '''
    # pdb.set_trace()
    frame_infos, traj_len = track2frame(tracks)
    obj_keys, map_trajs = calc_frame_iou(frame_infos,traj_len, iou_thr=0.2)

    merge2track_all(tracks, obj_keys, map_trajs, frame_infos)



def tk_filter(tk_data):
    for seq, tk in tk_data.items():
        # pdb.set_trace()
        if 'unlabel' in tk.keys():
            tk_filter_clip(tk['label'], with_gt=True)
            tk_filter_clip(tk['unlabel'], with_gt=False)
        else:
            tk_filter_clip(tk, with_gt=False)
    return tk_data

if __name__=='__main__':
    print('track filter ...')
    import sys
    src_tkfile= sys.argv[1]
    dst_tkfile= src_tkfile[:-4] + '_tkf.pkl'
    tk_data = tk_filter(load_tk_data(src_tkfile))
    dump_tk_data(tk_data, dst_tkfile)
    print('Done')
