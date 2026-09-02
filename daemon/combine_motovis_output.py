import os
import pickle
from collections import defaultdict
import argparse
from pathlib import Path

from tqdm import tqdm
from functools import partial
import concurrent.futures as futures
import numpy as np

from detzero_utils.common_utils import create_logger

from detzero_track.utils.transform_utils import transform_boxes3d
from detzero_track.utils.data_utils import sequence_list_to_dict, dict_to_sequence_list
import pdb



def load_pkl(path):
    with open(path, 'rb') as f:
        info = pickle.load(f)
    return info

def save_pkl(data, path):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def combine_det(combine_data, drop_path):
    drop_data = load_pkl(drop_path)
    combine_data = sequence_list_to_dict(combine_data)
    seq_names = list(combine_data)
    
    for seq in seq_names:
        frames = list(combine_data[seq].keys())
        for frm in frames:
            # pdb.set_trace()
            for key in ['boxes_lidar', 'name', 'score']:
                combine_data[seq][frm][key] = np.concatenate([
                    combine_data[seq][frm][key],
                    drop_data[seq][frm][key]], axis=0)
            obj_ids = combine_data[seq][frm]['obj_ids']
            n = combine_data[seq][frm]['name'].shape[0] - obj_ids.shape[0]
            if n>0:
                obj_ids = np.pad(obj_ids, (0, n), mode='constant', constant_values=-1)
                combine_data[seq][frm]['obj_ids'] = obj_ids
            
    return dict_to_sequence_list(combine_data)

def convert_frame_format(track_data):
    """
    Function:
        convert track data from seq->obj_id dict format into frame-level list format
    Args:
        track_data: dict of track data {seq_n: {track_id: {xxxx}}}
    Returns:
        frame_res_list: list of frame-level result
    """
    frame_res_list = list()
    order_map = defaultdict(list)

    for tk_id, tk_info in track_data.items():
        sample_idx = tk_info['sample_idx']
        for i, sa_idx in enumerate(sample_idx):
            order_map[sa_idx].append([tk_id, i])

    frames = list(order_map.keys())
    for frm_id in frames:
        map_temp = np.stack(order_map[frm_id])
        obj_ids, orders = map_temp[:, 0], map_temp[:, 1]

        seq = track_data[obj_ids[0]]['sequence_name']
        pose = track_data[obj_ids[0]]['pose'][orders[0]]
        
        obj_num = len(obj_ids)

        boxes_lidar = np.zeros((obj_num, 9), dtype=np.float32)
        boxes_global = np.zeros((obj_num, 9), dtype=np.float32)
        score = np.zeros((obj_num), dtype=np.float32)
        name = np.full(obj_num, 'none', dtype=object)
        
        for i, obj_id in enumerate(obj_ids):
            idx = orders[i]
            if 'boxes_lidar' in track_data[obj_id]:
                boxes_lidar[i] = track_data[obj_id]['boxes_lidar'][idx]
            elif 'boxes_global' in track_data[obj_id]:
                boxes_global[i] = track_data[obj_id]['boxes_global'][idx]
                boxes_lidar[i] = transform_boxes3d(boxes_global[i], pose, inverse=True, with_vel=True).reshape(-1)

            score[i] = track_data[obj_id]['score'][idx]
            name[i] = track_data[obj_id]['name'][idx]

        frame_res_list.append({
            'sequence_name': seq,
            'frame_id': frm_id,
            'obj_ids': obj_ids,
            'name': name,
            'score': score,
            'boxes_lidar': boxes_lidar,
            # 'boxes_global': boxes_global,
            'pose': pose
        })
    
    return frame_res_list

def load_track_pkl(track_pkl):
    track_res = load_pkl(track_pkl)
    track_data = dict()
    for seq in track_res.keys():
        track_data[seq] = dict()
        for tmpk in ['label', 'unlabel']:
            label_tk = track_res[seq].pop(tmpk, None)
            if label_tk is not None:
                track_data[seq].update(label_tk)
        # pdb.set_trace()
        if len(track_res[seq])>0:
            track_data[seq].update(track_res[seq])
    return track_data

def combine_refining(root_path, split, class_names, combine_conf_res=True):
    combine_dict = defaultdict(dict)
    # fp = open('refining_diff_obj.txt', 'w')
    for class_name in class_names:
        geo_path = os.path.join(root_path, split, '%s_geometry_%s.pkl' % (class_name, split))
        pos_path = os.path.join(root_path, split, '%s_position_%s.pkl' % (class_name, split))

        if not os.path.exists(geo_path) or not os.path.exists(pos_path):
            raise FileNotFoundError('Cannot find the input files.')
        
        # geo_res, pos_res只包含部分
        geo_res = load_pkl(geo_path)
        pos_res = load_pkl(pos_path)
        
        if combine_conf_res:
            conf_path = os.path.join(root_path, split, '%s_confidence_%s.pkl' % (class_name, split))
            conf_res = load_pkl(conf_path)
        
        seq_names = list(pos_res.keys())

        for seq in tqdm(seq_names, desc='combine refining'):
            obj_ids = pos_res[seq].keys()
            info = f'{seq}: \n'
            for obj in obj_ids:
                boxes_geo = np.concatenate(geo_res[seq][obj]['boxes_lidar'], axis=0)
                pos_res[seq][obj]['boxes_lidar'] = np.array(pos_res[seq][obj]['boxes_lidar'])
                # info += 'obj_id: {} =============\n'.format(obj)
                # info += 'xyz_diff, org - ref: \n{} \n'.format(boxes_geo[:, :3] - pos_res[seq][obj]['boxes_lidar'][:,:3])
                # info += 'lwh_diff, org - ref: \n{} \n'.format(pos_res[seq][obj]['boxes_lidar'][:, 3:6] - boxes_geo[:, 3:6])
                pos_res[seq][obj]['boxes_lidar'][:, 3:6] = boxes_geo[:, 3:6]
                if combine_conf_res:
                    info += 'socre, org, ref: \n{} \n'.format(np.stack([pos_res[seq][obj]['score'], conf_res[seq][obj]['new_score']], axis=1))
                    # pdb.set_trace()
                    pos_res[seq][obj]['score'] = conf_res[seq][obj]['new_score']
                # pdb.set_trace()
                pos_res[seq][obj]['sample_idx'] = \
                    np.array([str(x) for x in pos_res[seq][obj]['frame_id']])
                
                combine_dict[seq][obj] = pos_res[seq][obj]
                combine_dict[seq][obj]['isrefined'] = True
    #         fp.write(info)
    # fp.close()
    # pdb.set_trace()
    return combine_dict

def combine_track_data(combine_dict, combine_track_path=None):
    '''
        从track pkl里面获取速度
        部分类别做了refining。对于未refinine的目标，直接从track里面获取
    '''
    if combine_track_path is None:
        return combine_dict
    tk_data = load_track_pkl(combine_track_path)
    # tmp_key = 'ABC1_1734586652'
    # pdb.set_trace()
    for seq in tk_data.keys():
        obj_ids = tk_data[seq].keys()

        com_obj_ids = set(combine_dict[seq].keys()) if seq in combine_dict.keys() else []
        for obj in obj_ids:
            track = tk_data[seq][obj]
            if 'track' in track.keys():
                track = track['track']

            if obj in com_obj_ids:
                continue
                pdb.set_trace()
                num_points = track['num_points']
                num_pts_thr = 5
                for i, n in enumerate(num_points):
                    if n>num_pts_thr:
                        continue
                    tk_boxes_global = track['boxes_global'][i]
                    pose = track['pose'][i]
                    tk_boxes_lidar = transform_boxes3d(tk_boxes_global, pose, inverse=True).reshape(-1)
                    combine_dict[seq][obj]['boxes_lidar'][i] = tk_boxes_lidar
                continue
            # pdb.set_trace()  
            # track = filter_track(track)
            combine_dict[seq][obj] = track
            combine_dict[seq][obj]['isrefined'] = False
            combine_dict[seq][obj]['sequence_name'] = seq
    return combine_dict



def combine_final(root_path, class_names, logger, split, combine_track_path, combine_conf_res=True, template_pkl=None,
                    combine_drop_path=None, track_save=True, frame_save=True, output_dir=None, no_refining=False):
    '''
        根据combine_track_path, combine_drop_path, geo_path, pos_path
        修改template_pkl里面boxes_lidar
    '''
    if no_refining:
        combine_dict = defaultdict(dict)
    else:
        combine_dict = combine_refining(root_path, split, class_names=class_names, combine_conf_res=combine_conf_res)
    combine_dict = combine_track_data(combine_dict, combine_track_path)
    seq_names = list(combine_dict.keys())
    # pdb.set_trace()
    if track_save:
        save_path = os.path.join(root_path, split, 'final.pkl')
        save_pkl(combine_dict, save_path)
        logger.info('Track level final result is saved at %s' % save_path)

    if frame_save:
        logger.info('Start to convert track level result into frame level')

        final_res = list()    
        seq_data = [combine_dict[x] for x in seq_names]
        for tmp_data in seq_data:
            final_res.extend(convert_frame_format(tmp_data))

        # converter = partial(convert_frame_format)
        # with futures.ProcessPoolExecutor(max_workers=1) as executor:
        #     thread_bar = tqdm(executor.map(
        #         converter,
        #         seq_data,
        #         chunksize=1),
        #         total=len(seq_names), ascii=True, ncols=140)
            
        # for idx, processed_data in enumerate(thread_bar):
        #     final_res.extend(processed_data)
        # pdb.set_trace()
        # not combine dropped objects when used as auto labels
        if combine_drop_path is not None:
            logger.info('Start to combine the dropped objects from %s' % combine_drop_path)
            final_res = combine_det(final_res, combine_drop_path)
        save_path = os.path.join(root_path, 'final_frame.pkl')
        save_pkl(final_res, save_path)
        logger.info('Frame level final result is saved at %s' % save_path)

    if template_pkl is None:
        return
    template_data = load_pkl(template_pkl)
    template_data = combine2template_data(final_res, template_data)
    save_path = os.path.join(output_dir, 'infer_results_final.pkl')
    save_pkl(template_data, save_path)
    logger.info('annotation final result is saved at %s' % save_path)


def combine2template_data(final_res, template_data):
    def update_template_data(res_data, temp_data):
        keys = ['name', 'score', 'boxes_lidar']
        temp_data['dt_instance_inds'] = res_data['obj_ids']
        for k in keys:
            temp_data[k] = res_data[k].astype(temp_data[k].dtype)
        return temp_data

    temp_token2idx = dict()
    for i, data in enumerate(template_data):
        scene_token = data['scene_token']
        token = data['token']
        temp_token2idx[f'{scene_token}_{token}'] = i
    # pdb.set_trace()ABC1_1734586652
    for data in final_res:
        seq = data['sequence_name']
        frame_id = data['frame_id']
        token = f'{seq}_{frame_id}'
        idx = temp_token2idx[token]
        template_data[idx] = update_template_data(data, template_data[idx])
    return template_data
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--root_path', type=str, required=True, help='the sensor data for test')
    parser.add_argument('--output_dir', type=str, required=True, help='the sensor data for test')

    parser.add_argument('--split', type=str, default='val',
                        help='specify the target split for processing')
    parser.add_argument('--class_names', type=str, nargs='+', default='vehicle',
                        help='')
    
    parser.add_argument('--combine_conf_res', action='store_true', default=False,
                        help='combine the confidence refining results together')
    parser.add_argument('--no_refining', action='store_true', default=False,
                        help=' ')
    
    parser.add_argument('--tk_track_pkl', type=str, required=True,
                        help='determine the dropped results at tracking module')
    parser.add_argument('--tk_drop_pkl', type=str, default=None,
                        help='determine the dropped results at tracking module')

    parser.add_argument('--template_pkl', type=str, default=None,
                        help='the infer pkl from detection')
    
    parser.add_argument('--track_save', action='store_true', default=True,
                        help='save the combined result as original object track dict')
    parser.add_argument('--frame_save', action='store_true', default=True,
                        help='save the combined result as frame level list')
    
    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    logger = create_logger()
    combine_final(
        root_path=args.root_path,
        class_names=args.class_names,
        logger=logger,
        split=args.split,
        combine_track_path=args.tk_track_pkl,
        combine_conf_res=args.combine_conf_res,
        combine_drop_path=args.tk_drop_pkl,
        track_save=args.track_save,
        frame_save=args.frame_save,
        template_pkl=args.template_pkl,
        output_dir=args.output_dir,
        no_refining=args.no_refining
    )
