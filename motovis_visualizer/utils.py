import os
import matplotlib


import numpy as np
import laspy
from pyquaternion import Quaternion
import pdb

__all__ = ['load_point_cloud', 'get_bbox_corners3d', 
           'get_matrix4x4', 'get_matrix4x4_2', 'proj_lidar2img', 'create_color_maps']

b2d_lidar2motovis_lidar = np.array([
                            [0,1,0,0],
                            [1,0,0,0.39],
                            [0,0,1,-1.84],
                            [0,0,0,1],
            ])

def load_point_cloud(lidar_file: str, dim: int = 3):
    extname = os.path.splitext(lidar_file)[-1]
    if extname == '.pcd':
        import open3d as o3d

        points = o3d.io.read_point_cloud(lidar_file)
        points = np.asarray(points)
    elif extname in ['.las', '.laz']:
        las_data = laspy.read(lidar_file)
        points = las_data.xyz
    else:
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, dim)
        # points = np.fromfile(lidar_file)
    return points.astype(np.float32)[:,:3]


# def load_point_cloud(filename, dim=3):
#     extname = os.path.splitext(filename)[-1].lower()
#     filename = os.path.abspath(filename)

#     if extname == '.pcd':
#         cloud_points = o3d.io.read_point_cloud(filename)
#         points = np.asarray(cloud_points.points).astype(np.float32)
#     else:
#         points = np.fromfile(filename, dtype=np.float32)
        
#     return points.reshape(-1, dim)[:, :3]


def get_bbox_corners3d(bbox3d, cam_intrinsic=None, sensor2lidar=None, on_images=True, on_lidar=False, return_front_idx=False, pts_orders='flu'):
    """
    bbox3d: [x, y, z, dx, dy, dz, yaw]
    pts_orders: 标识xyz的方向, flu: 前左上; rfu: 右前上
    """
    x, y, z, dx, dy, dz, yaw = bbox3d[:7]

    # 计算 3D box 的 8 个角点（局部坐标系）
    dx, dy, dz = dx / 2, dy / 2, dz / 2
    corners_3d = np.array([
        [dx, dy, dz], [dx, -dy, dz], [-dx, -dy, dz], [-dx, dy, dz],  # 上面四个点
        [dx, dy, -dz], [dx, -dy, -dz], [-dx, -dy, -dz], [-dx, dy, -dz]  # 下面四个点
    ]).T  # 变成 8x3 形状

    # 旋转角点 (yaw 旋转)
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    rotated_corners = rot_mat @ corners_3d

    # 变换到激光雷达坐标系 (4x8)
    corners_lidar = rotated_corners + np.array([[x], [y], [z]])


    if on_images:   
        assert cam_intrinsic is not None, "cam_intrinsic must be provided when on_images is True"
        assert sensor2lidar is not None, "sensor2lidar must be provided when on_images is True"

        ones = np.ones((1, 8))
        corners_lidar_homo = np.vstack((corners_lidar, ones)) 

        # 变换到图像坐标系
        lidar2img         = cam_intrinsic @ np.linalg.inv(sensor2lidar)
        projected_corners = lidar2img @ corners_lidar_homo  
        projected_corners = projected_corners.T
        
        projected_corners[:, :2] /= projected_corners[:, [2]]
        projected_corners = projected_corners[projected_corners[:, 2] > 0.1][:, :2].astype(np.int32)
        if return_front_idx:
            if pts_orders == 'flu':
               front_index = [0,1,4,5]
            elif pts_orders == 'rfu':
                # front_index = [0, 3, 4, 7]
                front_index = [0,1,4,5]
            else:
                assert False, 'pts_order must be in [flu, rfu], but get {}'.format(pts_orders)
            projected_corners = (projected_corners, front_index)

        return projected_corners
    
    if on_lidar:
        if return_front_idx:
            if pts_orders == 'flu':
               front_index = [0,1]
            elif pts_orders == 'rfu':
                # front_index = [0, 3]
                front_index = [0,1]
            else:
                assert False, 'pts_order must be in [flu, rfu], but get {}'.format(pts_orders)
            corners_lidar = (corners_lidar, front_index)
        return corners_lidar


def get_matrix4x4(m):
    output = np.eye(4)
    output[:3, :3] = m
    return output


def get_matrix4x4_2(rotation, translation, quat=True, inverse=True):
    output = np.eye(4)
    if quat:
        output[:3, :3] = Quaternion(rotation).rotation_matrix
    else:
        output[:3, :3] = rotation

    output[:3, 3]  = translation
    
    if inverse:
        output = np.linalg.inv(output)

    return output


def proj_lidar2img(lidar_points, cam_intrinsic, sensor2lidar, result_dim=2):
    """
    将 BEV 空间中的点映射到图像坐标系。
    
    Args:
        lidar_points (np.ndarray): 形状为 (N, 2) 的 BEV 坐标点 (x, y)。
        cam_intrinsic (np.ndarray): 3x3 相机内参矩阵。
        sensor2lidar (np.ndarray): 4x4 变换矩阵，从 BEV 坐标转换到 LiDAR 坐标系。
    
    Returns:
        img_points (np.ndarray): 形状为 (N, 2) 的图像坐标点 (u, v)。
        mask (np.ndarray): 形状为 (N,) 的布尔数组，表示是否投影到图像内的有效点。
    """
    lidar2sensor = np.linalg.inv(sensor2lidar)
    if lidar_points.shape[1] == 2:
        lidar_points_homo = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1)) * -sensor2lidar[3, 3], np.ones((lidar_points.shape[0], 1))))
    else:
        lidar_points_homo = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))

    lidar_points_homo = lidar_points_homo.T

    # 变换到图像坐标系
    lidar2img = cam_intrinsic @ lidar2sensor # cam2img @ cam2lidar
    projected_points = lidar2img @ lidar_points_homo  
    projected_points = projected_points.T

    projected_points[:, :2] /= projected_points[:, [2]]
    projected_points = projected_points[projected_points[:, 2] > 0]

    return projected_points[:, :result_dim]


def create_color_maps(total_steps, colormap='autumn'):
    cmap = matplotlib.cm.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, total_steps))[:, :3] * 255
    # colors = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3] * 255
    colors = colors[:, ::-1]  # RGB to BGR

    return colors
