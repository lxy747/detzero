import os, glob
import os.path as op
import time
import yaml
import pickle
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
import shutil
from tqdm import tqdm
import json
import sys
sys.path.append('./')
import datetime
from easydict import EasyDict
from pathlib import Path

CameraNameMapping = {
    'cam_sy_sg8s30_front': 'CAM_FRONT_NARROW', 
    'cam_sy_sg8s120_front': 'CAM_FRONT', 
    'cam_sy_sg3s100_zs_leftfront': 'CAM_FRONT_LEFT', 
    # 'cam_sy_sg3s190_avm_left': 'CAM_FRONT_LEFT', 
    'cam_sy_sg3s100_zs_rightfront': 'CAM_FRONT_RIGHT', 
    'cam_sy_sg3s100_zs_leftback': 'CAM_BACK_LEFT', 
    'cam_sy_sg3s100_zs_rightback': 'CAM_BACK_RIGHT', 
    'cam_sy_sg8s60_back': 'CAM_BACK'
}

def q2rot(q):
    """
    将四元数转换为旋转矩阵
    
    Args:
        q: 四元数[x, y, z, w]
    
    Returns:
        3x3 旋转矩阵
    """
    # scipy默认使用 [x, y, z, w] 顺序
    rotation = Rotation.from_quat(q)
    return rotation.as_matrix()

class ImageAgent:
    def __init__(self, image_dir):        
        self.jpg_list = glob.glob(os.path.join(image_dir, '*.jpg'))
        self.jpg_list.sort()
        self.parse_timestamps()
        
    
    def parse_timestamps(self):
        img_timestamps = [os.path.split(fn)[-1]   for fn in self.jpg_list]
        self.img_timestamps = np.array([float(fn[:-4])  for fn in img_timestamps])
        
    
    def __call__(self, timestamp, tolerance=0.020):
        '''
            tolerance: unit s
        '''
        diff = np.abs(float(timestamp) - self.img_timestamps)
        idx = np.argmin(diff)
        if diff[idx] < tolerance:
            return self.jpg_list[idx]
        return None
    
class T68Converter:
    def __init__(self, args: argparse.Namespace):
        clips = [
            params.workspace
        ]
         
        self.args = args
        self.clip_dirs = clips
        self.save_dir = self.args.output_dir
        self.batch_name = "batch_detzero"
        self.need_copy = False
    
    def _dst_pcd_dir(self, batch_name, clip_name):
        pcd_dir = os.path.join(
            batch_name, 'samples',
            clip_name, 
            'lidar_hs_pan128_middle', 'las_undistort'
            )
        abs_dir = os.path.join(self.save_dir, pcd_dir)
        os.makedirs(abs_dir, exist_ok=True)
        return pcd_dir, abs_dir

    def _dst_image_dir(self, batch_name, clip_name, cam):
        image_dir = os.path.join(
            batch_name, 'samples', clip_name, 'images', cam
        )
        abs_dir = os.path.join(self.save_dir, image_dir)
        os.makedirs(abs_dir, exist_ok=True)
        return image_dir, abs_dir

    def _dst_pkl_dir(self, batch_name):
        tmp_dir = os.path.join(self.save_dir, 'pkls_motovis_demotion', batch_name)
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def _get_batch_clip_name(self, clip_dir):
        p1, clip_name = os.path.split(clip_dir)
        if self.batch_name is None:
            batch_name = os.path.split(p1)[-1]
        else:
            batch_name = self.batch_name
        return batch_name, "clip_detzero"

    def _load_pcd_fns(self, clip_dir):
        pcd_fns = self.args.data_root
        pcd_fns = glob.glob(os.path.join(pcd_fns, '*.las'))
        # pdb.set_trace()
        return pcd_fns

    def _load_pose(self, clip_dir, clip_info:dict):
        pose_file = self.args.trajectory_file
        
        with open(pose_file, 'r') as fp:
            poses = fp.readlines()
            poses = [
                        [float(v) for v in line.strip().split(',') ]     
                     for line in poses ]
            
            
        timestamps = list(clip_info.keys())
        timestamps.sort()

        tolerance = 0.01
        poses = np.array(poses)
        for i in range(len(timestamps)):
            t1 = timestamps[i]
            diff = np.abs(t1 - poses[:, 0])
            idx = np.argmin(diff)
            if diff[idx] < tolerance:
                clip_info[t1]['pose'] = poses[idx]

        ego2lidar = self.params['ego2lidar']
        lidar2ego = self.params['lidar2ego']
        # min_delta = 10000000
        for t1 in timestamps:
            if 'pose' not in clip_info[t1]:
                # delta = np.min(np.abs(t1 - poses_ts))
                # if delta < min_delta:
                #     min_delta = delta
                # print(t1, 'delta=', delta)
                # pdb.set_trace()
                clip_info.pop(t1)
                continue

            pose = clip_info[t1].pop('pose')
            lidar2global_translation = np.array(pose[1:4])
            lidar2global_rotation = q2rot(pose[4:])
            # pdb.set_trace()
            lidar2global = np.eye(4)
            lidar2global[:3,:3] = lidar2global_rotation
            lidar2global[:3, 3] = lidar2global_translation
            global2lidar = np.linalg.inv(lidar2global)

            global2ego = lidar2ego @ global2lidar
            ego2global = lidar2global @ ego2lidar

            tmp_info = {
                'ego2lidar': ego2lidar,
                'lidar2ego':lidar2ego,
                'lidar2ego_rotation':lidar2ego[:3,:3],
                'lidar2ego_translation':lidar2ego[:3,3],

                'ego2global': ego2global,
                'global2ego': global2ego,
                'ego2global_rotation': ego2global[:3,:3],
                'ego2global_translation': ego2global[:3, 3],

                'lidar2global': lidar2global,
                'global2lidar': global2lidar,
                'lidar2global_rotation': lidar2global[:3,:3],
                'lidar2global_translation': lidar2global[:3, 3],

            }
            clip_info[t1].update(tmp_info)
        return clip_info                        

    def _load_calib(self, clip_dir):
        calib_file = self.args.calib_file
        with open(calib_file, 'r', encoding='utf-8') as file:
            params = yaml.safe_load(file)['rig']  # 推荐使用safe_load


        extrinsic = params[self.args.lidar_name]['extrinsic']
        r = q2rot(extrinsic[3:])
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = r
        lidar2ego[:3, 3] = np.array(extrinsic[:3])
        ego2lidar = np.linalg.inv(lidar2ego)
        
        params['lidar2ego'] = lidar2ego
        params['ego2lidar'] = ego2lidar
        return params

    def _create_cam_info(self):
        '''
            外参是extrinsic [tx, ty, tz, qx, qy, qz, qw] 都是T_ego_sensor
            畸变, pinhole相机取poly, 鱼眼相机取inv_poly
            poly [k1, k2, k3, k4]   inv_poly [k1, k2, p1, p2, k3, k4, k5]
            焦距 focal [fx, fy]  主点 pp [cx, cy]

            'cam_intrinsic': cam_intrinsic.tolist(),
            'sensor2ego': cam2ego.tolist(),
            'sensor2ego_rotation': cam2ego_rotation.tolist(),
            'sensor2ego_translation': cam2ego_translation.tolist(),
            'sensor2lidar': cam2lidar.tolist(),
            'sensor2lidar_rotation': cam2lidar_rotation.tolist(),
            'sensor2lidar_translation': cam2lidar_translation.tolist()
        '''
        cam_params = dict()
        for cam, cam_new in CameraNameMapping.items():
            print(cam, cam_new)
            # 'extrinsic', 'focal', 'fov_fit'
            extrinsic = self.params[cam]['extrinsic'] 
            focal = self.params[cam]['focal']
            pp = self.params[cam]['pp'] # 主点
            dist_paras = self.params[cam]['inv_poly'] if 'fish' in cam else  self.params[cam]['poly']

            cam_intrinsic = [focal[0], 0, pp[0],
                             0,  focal[1], pp[1],
                             0, 0, 1
                             ]
            
            sensor2ego_translation = np.array(extrinsic[:3])
            sensor2ego_rotation = q2rot(extrinsic[3:])
            sensor2ego = np.eye(4)
            sensor2ego[:3, :3] = sensor2ego_rotation
            sensor2ego[:3, 3] = sensor2ego_translation
            sensor2lidar = self.params['ego2lidar'] @ sensor2ego
            info_tmp = dict(
                cam_intrinsic=cam_intrinsic,
                dist_paras=dist_paras,
                sensor2ego=sensor2ego,
                sensor2ego_rotation=sensor2ego[:3,:3],
                sensor2ego_translation=sensor2ego[:3,3],
                sensor2lidar=sensor2lidar,
                sensor2lidar_rotation= sensor2lidar[:3,:3],
                sensor2lidar_translation=sensor2lidar[:3,3]
            )
            cam_params[cam_new] = info_tmp
        return cam_params

    def _load_cam_info(self, clip_dir, clip_info:dict):
        cam_params = self._create_cam_info()
        image_agent_dict = dict()
        for t  in clip_info.keys():
            cam_info = dict()
            timestamp = clip_info[t]['timestamp']
            for cam, cam_new in CameraNameMapping.items():
                if cam_new not in image_agent_dict.keys():
                    img_agent = ImageAgent(os.path.join(clip_dir, self.args.parse_dir, cam, 'jpg'))
                    image_agent_dict[cam_new] = img_agent

                cam_path = image_agent_dict[cam_new](timestamp, tolerance = 0.02)                
                # pdb.set_trace()
                if cam_path is None:
                    cam_path = os.path.join(clip_dir, self.args.parse_dir, cam, 'jpg', f'{timestamp}.jpg')
                tmp_info = dict(
                    type=cam_new,
                    data_path=cam_path,
                    timestamp = timestamp,
                    sample_data_token = f'{cam}_{timestamp}'
                )
                tmp_info.update(cam_params[cam_new])
                cam_info[cam_new] = tmp_info        
            clip_info[t]['cams'] = cam_info
        return clip_info


    def _process_pcd(self,batch_name, clip_name, info):
        '''
        cp pcd file. 
        maybe save to .bin
        '''
        lidar_path = info['lidar_path']
        if not os.path.exists(lidar_path):
            return False, lidar_path
        if self.need_copy:
            try:
                pcd_dir, abs_dir = self._dst_pcd_dir(batch_name, clip_name)
                fn = os.path.split(lidar_path)[-1]
                dst_path =  os.path.join(abs_dir, fn)
                if not os.path.exists(dst_path):
                    shutil.copy2(lidar_path, dst_path)
                info['lidar_path'] = os.path.join(pcd_dir, fn)
            except Exception as e:
                print(e)
                return False, lidar_path
            return True, (lidar_path, dst_path)
        else:
            return True, (lidar_path, lidar_path)
        

    def _process_images(self, batch_name, clip_name, info):
        pairs = []
        try:
            for cam in info['cams'].keys():
                data_path = info['cams'][cam]['data_path']
                if not os.path.exists(data_path):
                    return False, data_path
                
                if data_path is None:
                    # pdb.set_trace()
                    return False, []
                # if not os.path.exists(data_path):
                #     return False, data_path
                
                # sunchao  拷贝逻辑  ---------------------
                
                if self.need_copy:
                    img_dir, abs_dir = self._dst_image_dir(batch_name, clip_name, cam)
                    fn = os.path.split(data_path)[-1]

                    dst_fn = os.path.join(abs_dir, fn)
                    if not os.path.exists(dst_fn):
                        shutil.copy2(data_path, dst_fn)

                    info['cams'][cam]['data_path'] = os.path.join(img_dir, fn)
                    
                    pairs.append(
                        (data_path, os.path.join(abs_dir, fn))
                    )
                else:
                    pairs.append(
                        (data_path, data_path)
                    )
                # sunchao  拷贝逻辑结束  ---------------------
                
                # pdb.set_trace()
                
        except Exception as e:
            print(e)
            return False, []
        return True, pairs


    def _process_single_frame(self, batch_name, clip_name, info):
        '''
            return: 
                -1: no pcd
                0: no images
                1: has pcd and images
        '''
        info['timestamp'] = float(info['timestamp'])*1.0e6 # unit from s to 100us
        for cam in info['cams']:
            info['cams'][cam]['timestamp'] = float(info['cams'][cam]['timestamp']) * 1.0e6

        # flag, pcd_fns = self._process_pcd(batch_name, clip_name, info)
        # if not flag:
        #     log =f'pcd: {pcd_fns} not exists'
        #     return -1
        # if not os.path.exists(pcd_fns[1]):
        # shutil.copy2(pcd_fns[0], pcd_fns[1])
   
        flag, cam_fns = self._process_images(batch_name, clip_name, info)
        if not flag:
            log = f'image: {cam_fns} not exitst'
            print(log)
            return -1
        # shutil.copy2(pcd_fns[0], pcd_fns[1])
        # for f1, f2 in cam_fns:
            # if not os.path.exists(f2):
            # shutil.copy2(f1, f2)
        
        flag, pcd_fns = self._process_pcd(batch_name, clip_name, info)
        if not flag:
            log =f'pcd: {pcd_fns} not exists'
            return -1
        return 1
    
    def _process_single_frame_multi(self, info):
        return self._process_single_frame(*info)

    

    def _process_groundmap(self, clip_dir, batch_name, clip_name):
        tmp_gm_name='generate_xy_groundmap_1'
        groundmap_dir = os.path.join(clip_dir, 
                                     'temp_dir',
                                     tmp_gm_name
                                     )
        assert os.path.exists(groundmap_dir), f'no {tmp_gm_name}'
            
        dst_dir = os.path.join(self.save_dir, batch_name,
                               'samples',
                               clip_name, tmp_gm_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        print('copying groundmap to ', clip_name)
        shutil.copytree(groundmap_dir, dst_dir, dirs_exist_ok=True)

    def _process_single_clip(self, clip_dir):
        
        # 临时用params
        self.params = self._load_calib(clip_dir)

        batch_name, clip_name = self._get_batch_clip_name(clip_dir)
        if not clip_name:
            clip_name = "test"
        print(batch_name, clip_name)
        # try:
        #     self._process_groundmap(clip_dir, batch_name, clip_name)
        # except Exception as e:
        #     print(e)
        #     return False,f'{clip_name}: no map'

        clip_info = dict()
        pcd_fns = self._load_pcd_fns(clip_dir)
        for fn in pcd_fns:
            pcd_basename = os.path.split(fn)[-1]
            timestamp = os.path.splitext(pcd_basename)[0]
            tmp_info = dict(
                token = clip_name+'_'+timestamp,
                scene_token= clip_name,
                timestamp = timestamp,
                lidar_path=fn,
                key_frame= True,
                sweeps = [],
            )
            clip_info[float(timestamp)] = tmp_info
        # pdb.set_trace()
        self._load_pose(clip_dir, clip_info)
        self._load_cam_info(clip_dir, clip_info)
        
        del self.params

        timestamps = list(clip_info.keys())
        timestamps.sort()

        cnt = 0
        max_cnt = 10000
        infos = []
        for t in tqdm(timestamps, desc=clip_name, leave=True):
            info = clip_info[t]
            flag = self._process_single_frame(batch_name, clip_name, info)
            if flag==-1:
                continue
            if flag==0:
                info['key_frame'] = False
            else:
                cnt += 1
            infos.append(info)
            if cnt>max_cnt:
                break
        # pdb.set_trace()
        print(clip_name, len(infos))
        pkl_dir = self._dst_pkl_dir(batch_name)
        pkl_fn = os.path.join(pkl_dir, 
                                f'{clip_name}.pkl'
                              )
        with open(pkl_fn, 'wb') as fp:
            pickle.dump(infos, fp)
        


        return True, f'{clip_name}: {len(infos)}'


    def run(self):
        for _clip_dir in self.clip_dirs:
            print('process ', _clip_dir)
            s = time.time()
            process_flag, process_log = self._process_single_clip(_clip_dir)
            e = time.time()


if __name__ == '__main__':
    from MitConfigTools import MitConfigTools
    
    print("\n" + "="*60)
    print("脚本启动信息")
    print("="*60)
    print(f"[当前工作目录] {os.getcwd()}")
    print(f"[当前用户] {os.getenv('USER') or os.getenv('USERNAME')}")
    print(f"[Python版本] {sys.version}")
    print("="*60)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str,
        default="/home/motovis/sunchao/detzero/config.json"
    )
    args = parser.parse_args()

    config_path = args.config
    tool_name   = "generate_annos_with_detzero"

    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    params = EasyDict()

    if MitConfigTools.init_config(config_path, tool_name):
        print("\n[INFO] 成功读取配置文件")

        try:
            params.data_root       = MitConfigTools.get_param("lidar_root")
            params.calib_file      = MitConfigTools.get_param("calib_file")
            params.output_dir      = MitConfigTools.get_param("output_dir")
            params.parse_dir       = MitConfigTools.get_param("parse_dir")
            params.workspace       = MitConfigTools.get_param("workspace")
            params.split           = "test"
            params.lidar_name      = MitConfigTools.get_param("lidar_name")
            params.trajectory_file = MitConfigTools.get_param("trajectory_file")

            # ★ 把命令行开关写入 params
            params.od_infer = bool(MitConfigTools.get_param("od_infer"))
            params.diffubox  = bool(MitConfigTools.get_param("diffubox"))
            params.tracking  = bool(MitConfigTools.get_param("tracking"))
            params.refining  = bool(MitConfigTools.get_param("refining"))
            params.detection_ckpt      = MitConfigTools.get_param("detection_ckpt")

            output_path     = Path(params.output_dir).resolve()
            params.scene_id = output_path.parent.parent.name
            print(f"[INFO] 自动推导 scene_id: {params.scene_id}")

            print("\n" + "="*60)
            print("配置参数:")
            print("="*60)
            for k, v in params.items():
                print(f"  - {k}: {v}")
            print("="*60)

           
            converter = T68Converter(params)
            converter.run()

            print("\n" + "="*60)
            print("✅ 全部流程执行完成")
            print("="*60)

            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run_msg  = {"success": True, "errMsg": None}

        except Exception as e:
            end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            run_msg  = {"success": False, "errMsg": str(e)}
            print(f"\n❌ {tool_name} 执行失败! 错误: {e}")

            import traceback
            traceback.print_exc()
            raise

    else:
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        run_msg  = {"success": False, "errMsg": "配置初始化失败"}
        print("❌ 配置初始化失败")

    config_data['tools_config']['toolsRunMsg'] = [{
        "toolName":  tool_name,
        "startTime": start_time,
        "endTime":   end_time,
        **run_msg
    }]

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] 执行结果已写入: {config_path}")
