
import os
# import mmcv
import pickle
import cv2
import argparse
import numpy as np

from tqdm import tqdm
from copy import deepcopy

from colors import LINE_COLORS, ROAD_MARKER_COLORS

from utils import (load_point_cloud, get_bbox_corners3d, 
                   get_matrix4x4, get_matrix4x4_2, proj_lidar2img, create_color_maps)
import matplotlib.pyplot as plt
import pdb
from main import Visualizer, load_extra_info




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--nuscenes', action='store_true')
    parser.add_argument('--show_gt', action='store_true')
    parser.add_argument('--show_dt', action='store_true')
    parser.add_argument('--show_axis', action='store_true')
    parser.add_argument('--show_bev_lidar', action='store_true')
    parser.add_argument('--show_cam_lidar', action='store_true')
    parser.add_argument('--show_maps', action='store_true')
    parser.add_argument('--show_agents', action='store_true')
    parser.add_argument('--od_score', type=float, default=0.001)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--format', type=str, choices=['images', 'clip_video', 'video'], default='clip_video')
    args = parser.parse_args()

    visualizer = Visualizer(show_gt=args.show_gt,
                            show_dt=args.show_dt,
                            is_nuscenes=args.nuscenes, 
                            show_axis=args.show_axis, 
                            show_agents=args.show_agents, 
                            show_bev_lidar=args.show_bev_lidar,
                            show_cam_lidar=args.show_cam_lidar,
                            show_maps=args.show_maps,
                            od_score=args.od_score,
                            format=args.format,
                            fps=args.fps
                            )

    # load pkl
    # infos = mmcv.load(args.pkl_file, file_format="pkl")   # 加载pkl
    infos = pickle.load(open(args.pkl_file, 'rb'))

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(args.pkl_file), "vis")
    print('vis results are saved in ', save_dir)
    if isinstance(infos, dict):
        if 'infos' in infos.keys():
            infos = infos['infos']
    for info in tqdm(infos):
        if 'filepath' in info.keys():
            info.update(load_extra_info(info, root_dir=os.path.join(args.data_dir, 'pkls')))
            
        sample_token = info['token']
        # result_fig = visualizer.render(info, args.data_dir)   # 可视化
        result_fig = visualizer.render_scene(info, args.data_dir)   # 可视化
        visualizer.dump(result_fig, save_dir, sample_token, scene_token=info['scene_token'])  # 保存
    visualizer.release()
    print('######### vis Done ##########')