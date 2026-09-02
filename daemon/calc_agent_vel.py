import pickle
import sys, os
import numpy as np
from tqdm import tqdm
import argparse
import pdb

def scene2frame_list(infos):
    scene2frames = dict()
    for i, info in enumerate(infos):
        scene_token = info['scene_token']
        if scene_token not in scene2frames.keys():
            scene2frames[scene_token] = []
        scene2frames[scene_token].append(i)
    return scene2frames

def get_lidar2global_rt(info):
    lidar2global = np.array(info['lidar2global'])
    r = lidar2global[:3,:3]
    t = lidar2global[:3, 3:]
    return r, t

def get_ego2global_rt(info):
    lidar2global = np.array(info['ego2global'])
    r = lidar2global[:3,:3]
    t = lidar2global[:3, 3:]
    return r, t

def load_extra_info(info, data_root, extra_path):
    if 'filepath' in info:
        # filepath = info['filepath']
        filepath = info.pop('filepath')
        tmp = os.path.join(data_root, extra_path, filepath)
        extra_info = pickle.load(open(tmp, 'rb'))
        info.update(extra_info)
    return info

def scene_agent_traj(infos, frame_ids=None, ref_cs_is_lidar=True, data_root=None, extra_path='.'):
    if frame_ids is None:
        frame_ids = list(range(0, len(infos)))
    obj2traj = dict()
    for idx in frame_ids:
        info = infos[idx]
    
        info = load_extra_info(info, data_root, extra_path)
        # info['boxes_lidar'] = info['anns']['gt_boxes']
        # info['dt_instance_inds'] = info['anns']['track_tokens']
        # info['name'] = info['anns']['gt_names']
        # gt_velocity_3d = info['anns']['gt_velocity_3d']
        # pdb.set_trace()
        boxes_lidar = np.array(info['boxes_lidar'])
        objs_xyz_lidar = boxes_lidar[:, :3].T
        if ref_cs_is_lidar:
            lidar2global_r, lidar2global_t = get_lidar2global_rt(info)
        else:
            lidar2global_r, lidar2global_t = get_ego2global_rt(info)
        objs_xyz_global = lidar2global_r @ objs_xyz_lidar + lidar2global_t
        # gt_velocity_3d_gloabl =lidar2global_r @ gt_velocity_3d.T
        # pdb.set_trace()
        dt_instance_inds = info['dt_instance_inds']
        if dt_instance_inds is None:
            # print(info['scene_token'], info['token'])
            # pdb.set_trace()
            continue
        # if dt_instance_inds is None, it may be not key_frame
        for i, ins_idx in enumerate(dt_instance_inds):
            if ins_idx not in obj2traj.keys():
                traj = dict(
                    frame_ids=[],
                    id_in_frame=[],
                    xyz_global=[],
                    # gt_v_global=[],
                    boxes_size = [],
                    name=info['name'][i],
                    remove=False,
                    timestamp=[]
                )
                obj2traj[ins_idx] = traj
            obj2traj[ins_idx]['frame_ids'].append(idx)
            obj2traj[ins_idx]['id_in_frame'].append(i)
            obj2traj[ins_idx]['xyz_global'].append(objs_xyz_global[:, i])
            # obj2traj[ins_idx]['gt_v_global'].append(gt_velocity_3d_gloabl[:, i])
            obj2traj[ins_idx]['boxes_size'].append(boxes_lidar[i, 3:6])
            obj2traj[ins_idx]['timestamp'].append(float(info['timestamp']) * 1.0e-6)

    filter_traj(obj2traj)

    for obj_inds, traj in obj2traj.items():
        if traj.get('remove', False):
            continue
        vels = calc_vels(traj['xyz_global'], traj['timestamp'])
        traj['vel_global'] = vels
    return obj2traj

def filter_traj(obj2traj: dict, dis_threshold = 0.8):
    obj_ids = list(obj2traj.keys())
    traj_len = []
    for id in obj_ids:
        traj_len.append(len(obj2traj[id]['frame_ids']))

    sorted_ids = np.argsort(traj_len)[::-1]
    obj_ids = [obj_ids[i] for i in sorted_ids]

    for i in tqdm(range(len(obj_ids)-1), total=len(obj_ids) - 1, desc='FILTER_TRAJ', leave=False):
        base_id = obj_ids[i]
        if obj2traj[base_id]['remove']:
            continue
        thr = np.mean(obj2traj[base_id]['boxes_size'][0][:2])
        for j in range(i+1, len(obj_ids)):
            cur_id = obj_ids[j]
            if obj2traj[cur_id]['remove']:
                continue
            # print(base_id, cur_id)
            thr1 = dis_threshold * max(thr, np.mean(obj2traj[cur_id]['boxes_size'][0][:2]))
            dis = calc_traj_dis(obj2traj[base_id], obj2traj[cur_id])
            if dis < thr1:
                obj2traj[cur_id]['remove'] = True

def calc_traj_dis(traj1, traj2):
    # ["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian", "barrier", "traffic_cone"]
    if (traj1['name'] in ["car", "truck", "construction_vehicle", "bus", "bicycle"]):
        if traj2['name'] in ["pedestrian", "barrier", "traffic_cone"]:
            return 100000
    else:
        if traj2['name'] in ["car", "truck", "construction_vehicle", "bus", "bicycle"]:
            return 100000

    i, j = 0, 0
    dis_v= 0.0
    cnt = 0
    xy1 = []
    xy2 = []
    while (i<len(traj1['frame_ids'])) and (j < len(traj2['frame_ids'])):
        id1 = traj1['frame_ids'][i]
        id2 = traj2['frame_ids'][j]
        if id1 < id2:
            i += 1
        elif id1 > id2:
            j += 1
        else:
            xy1.append( traj1['xyz_global'][i][:2])
            xy2.append( traj2['xyz_global'][j][:2])
            # dis = traj1['xyz_global'][i] - traj2['xyz_global'][j]
            # dis_v += np.sqrt(np.sum(dis[:2] * dis[:2]))
            # cnt += 1
            i +=1
            j += 1
    if len(xy1) == 0:
    # if cnt == 0:
        return 100000
    # pdb.set_trace()
    # dis_v = dis_v / cnt
    dis = np.array(xy1) - np.array(xy2)
    dis = dis*dis
    dis_v = np.sqrt(dis.sum(1))
    dis_v = np.mean(dis_v)
    return dis_v

def render_vel2infos(obj2traj, infos, frame_ids=[], ref_cs_is_lidar=True):
    if len(frame_ids)==0:
        frame_ids = list(range(0, len(infos)))
    for i in frame_ids:
        infos[i]['vel_global'] = [None] * len(infos[i]['boxes_lidar'])

    for obj_id, traj in obj2traj.items():
        if traj.get('remove', False):
            continue
        for i in range(len(traj['frame_ids'])):
            frame_id = traj['frame_ids'][i]
            id_in_frame = traj['id_in_frame'][i]
            assert obj_id == infos[frame_id]['dt_instance_inds'][id_in_frame]
            # pdb.set_trace()
            infos[frame_id]['vel_global'][id_in_frame] = traj['vel_global'][:, i]

    keys_tmp = ['boxes_lidar', 'name', 'score', 'dt_instance_inds']
    for i in frame_ids:
        mask = []
        vel_global = []
        vel_global_org = infos[i].pop('vel_global')
        for v in vel_global_org:
            if v is None:
                mask.append(False)
            else:
                mask.append(True)
                vel_global.append(v)
        # if infos[i]['token'] in ['1734586692500000000_640_398']:
        #     pdb.set_trace()
        mask = np.array(mask)
        num_tmp = np.sum(mask)
        if num_tmp < 1:
            continue
        if num_tmp < len(vel_global_org):
            for k in keys_tmp:
                infos[i][k] = infos[i][k][mask]
        if ref_cs_is_lidar:
            global2lidar_r = np.array(infos[i]['global2lidar'])[:3,:3]
        else:
            global2lidar_r = np.array(infos[i]['global2ego'])[:3,:3]
        
        vel_global = np.stack(vel_global, axis=1)
        vel_lidar = global2lidar_r @ vel_global
        vel_lidar = vel_lidar.T.astype(np.float32)
    
        infos[i]['velocity_3d'] = vel_lidar


def calc_vels(agent_traj, timestamps):
    if len(timestamps)==1:
        return np.array([0.0, 0.0, 0.0]).reshape(3, 1)
    if isinstance(agent_traj, list):
        agent_traj = np.stack(agent_traj, axis=1)
        timestamps = np.array(timestamps)
    vel = np.zeros_like(agent_traj)
    
    if len(timestamps) == 2:
        dt = timestamps[1] - timestamps[0]
        v = (agent_traj[:, 1] - agent_traj[:, 0]) / dt
        vel[0, :] = v[0]
        vel[1,:] = v[1]
        return vel
    num = timestamps.shape[0]
    # pdb.set_trace()
    # 前后向差分
    d1 = timestamps[1:] - timestamps[:num-1]
    v1 = (agent_traj[:2, 1:] - agent_traj[:2, :num-1]) / d1[None, :]
    vel[:2, 0] = v1[:, 0]
    vel[:2,1:] = v1

    # t = timestamps - timestamps[0]
    # dt = dt = t[1] - t[0] if np.allclose(np.diff(t), t[1]-t[0]) else np.gradient(t)
    # vx = np.gradient(agent_traj[0,:], dt)
    # vy = np.gradient(agent_traj[1,:], dt)
    # vel = np.stack([vx, vy, np.zeros_like(vx)], axis=0)
    
    return vel


def parse_args():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('pkl_file', type=str)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--extra_path', type=str, default='.')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--ref_cs', type=str, default='lidar')
    args = parser.parse_args()
    return args

def main():
    print('calc velocity_3d ...')
    args = parse_args()
    ref_cs_is_lidar = args.ref_cs == 'lidar'
    pkl_fn = args.pkl_file
    pkl_dir, fn = os.path.split(pkl_fn)
    print('src data load from ', pkl_fn)

    data = pickle.load(open(pkl_fn, 'rb'))
    infos = data['infos'] if isinstance(data, dict) else data


    dst_fn = os.path.join(pkl_dir, fn[:-4] + '_vel')
    if args.split:
        print('dst data dump to dir ', dst_fn)    
    scene2frames = scene2frame_list(infos)
    bar = tqdm(scene2frames.items(), total=len(scene2frames))
    for scene_token, frame_ids in bar:
        bar.set_description(scene_token)
        if scene_token not in ['ABC1_1734586652']:
            continue
        obj2traj = scene_agent_traj(infos, frame_ids, ref_cs_is_lidar, data_root=args.data_root, extra_path=args.extra_path)
        render_vel2infos(obj2traj, infos, frame_ids, ref_cs_is_lidar)
        if args.split:
            os.makedirs(dst_fn, exist_ok=True)
            scene_infos = [infos[i] for i in frame_ids]
            tmp_dst_fn = os.path.join(dst_fn, scene_token + '.pkl')
            pickle.dump(
                scene_infos, open(tmp_dst_fn, 'wb')
            )

    if not args.split:
        dst_fn += '.pkl'
        print('dst data dump to ', dst_fn)
        pickle.dump(
            data, open(dst_fn, 'wb')
        )
    print('Done')

if __name__=='__main__':
    main()
