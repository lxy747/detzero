import numpy as np
import pickle
# import cv2
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
    pdb.set_trace()
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
