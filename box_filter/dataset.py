import torch
from torch.utils.data import Dataset
from detzero_utils.common_utils import get_matrix
from detzero_utils.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou3d_cpu

import numpy as np
import random
import cv2
import pickle
import os
from torchvision import transforms

import pdb

def xyxy2area(boxes):
    '''
        boxes: [-1, 4]
    '''
    w = boxes[:,2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return w * h

def boxes3d_lidar_corners(boxes_lidar):
    if len(boxes_lidar.shape)==1:
        boxes_lidar = boxes_lidar.reshape(1, -1)
    xyz = boxes_lidar[:, :3]
    dxyz = boxes_lidar[:, 3:6]
    yaw = boxes_lidar[:, 6:7]

    corners_3d = np.array([
        [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5],  # 上面四个点
        [0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5]  # 下面四个点
    ])

    corners_3d = dxyz[:, None, :] * corners_3d[None, :, :] # [-1, 8, 3]
    cs = np.cos(yaw)
    ss = np.sin(yaw)
    zero = np.zeros_like(cs)

    # 旋转角点 (yaw 旋转) [-1, 3, 3]
    rot_mat = np.stack(
        [
            np.concatenate([cs, -ss, zero], axis=1), # [-1, 3]
            np.concatenate([ss, cs, zero], axis=1),
            np.concatenate([zero, zero, zero + 1], axis=1),
        ], axis=1
    )
    
    # (rot_mat @ corners_3d.T).T
    rotated_corners = np.einsum('bjk,bnk->bnj', rot_mat, corners_3d)
    corners_lidar = rotated_corners + xyz[:, None, :]
    return corners_lidar

def corners3d2imgbb(corners_lidar, lidar2cam, K, expand_ratio = 1.0, hw=None):
    '''
        corners_liar: [-1, 8, 3]
        lidar2cam: [4, 4]
        K: [3,3]
    '''
    assert expand_ratio > 0.0

    id_in_3d = np.array(list(np.arange(0, corners_lidar.shape[0])), dtype=np.int32)

    eps = 0.1
    ones = np.ones_like(corners_lidar[:, :, :1])
    corners_lidar = np.concatenate([corners_lidar, ones], axis=2)
    corners_cam = np.einsum('bnj,ij->bni', corners_lidar, lidar2cam)[:, :, :3]

    cam_maks = corners_cam[:, :, 2] > eps
    id_mask = cam_maks.sum(1)==cam_maks.shape[1]
    
    id_in_3d = id_in_3d[id_mask]
    if id_in_3d.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.int32), id_in_3d
    corners_cam = corners_cam[id_mask, :, :]

    corners_img = np.einsum('bnj,ij->bni', corners_cam, K)
    corners_img = corners_img[:, :, :2] / corners_img[:, :, 2:3]
    x1 = np.min(corners_img[:, :, 0], axis=1)
    y1 = np.min(corners_img[:, :, 1], axis=1)
    x2 = np.max(corners_img[:, :, 0], axis=1)
    y2 = np.max(corners_img[:, :, 1], axis=1)

    expand_ratio -= 1.0
    if np.abs(expand_ratio) > 0.0001:
        expand_ratio = 0.5*expand_ratio
        expand_w2 = (x2 - x1) * expand_ratio
        expand_h2 = (y2 - y1) * expand_ratio
        x1 -= expand_w2
        x2 += expand_w2
        y1 -= expand_h2
        y2 += expand_h2

    
    if hw is not None:
        # pdb.set_trace()
        # s = np.max(np.stack([y2 - y1, x2 - x1], axis=-1), axis=-1) * 0.5
        # cx = (x1 + x2)*0.5
        # cy = (y1 + y2)*0.5
        # x1, y1, x2, y2 = cx - s, cy - s, cx + s, cy + s

        x1 = np.clip(x1.astype(np.int32), a_min=0, a_max=hw[1])
        x2 = np.clip(x2.astype(np.int32), a_min=0, a_max=hw[1])
        y1 = np.clip(y1.astype(np.int32), a_min=0, a_max=hw[0])
        y2 = np.clip(y2.astype(np.int32), a_min=0, a_max=hw[0])

        w = x2 - x1
        h = y2 - y1

        mask = np.logical_and(
            w > 8, h > 8
        )

        mask2 = np.logical_and(
            w/(h + 0.01) > 0.2, h/(w + 0.01) > 0.2
        )
        mask = np.logical_and(mask, mask2)
        id_in_3d = id_in_3d[mask]

        imgbb = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)
        imgbb = imgbb[mask, :]
    else:
        imgbb = np.stack([x1, y1, x2, y2], axis=1).astype(np.int32)

        

    return imgbb, id_in_3d
    
def drawbb(img_cv, imgbb):
    for x1, y1, x2, y2 in imgbb:
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), thickness=1)
    return img_cv


def scale_img(img_cv, dst_hw=None):
    if dst_hw is None:
        return img_cv

    h, w = img_cv.shape[:2]
    # cnt = f"{h}x{w}"
    # cv2.imwrite(f'vis/scale1_{cnt}.jpg', img_cv)
    
    # pdb.set_trace()
    dh, dw = dst_hw
    scale = min(dh/h, dw/w)
    h1, w1 = int(h*scale), int(w*scale)
    out = np.zeros((dh, dw, 3), dtype=img_cv.dtype)
    res = cv2.resize(img_cv, (w1, h1), interpolation=cv2.INTER_LINEAR)
    x1 = (dw - w1)//2
    y1 = (dh - h1)//2
    out[y1:y1+h1, x1:x1+w1, :] = res
    # cv2.imwrite(f'vis/scale2_{cnt}.jpg', out)
    # pdb.set_trace()
    return out, scale, (x1, y1)

def crop_patch(img_cv, imgbbs, dst_hw):
    patchs = []
    for bb in imgbbs:
        x1, y1, x2, y2 = bb
        patch = img_cv[y1:y2+1, x1:x2+1]
        # h, w = patch.shape[:2]
        # if (h==0 or w==0):
        #     pdb.set_trace()
        patch, s, t = scale_img(patch, dst_hw)
        patchs.append(patch)

    return patchs

def aug3DBox(boxes3d:np.ndarray, ts=0.2, ss=0.08, rots=5):
    boxes3d = boxes3d.copy()
    num_obj = boxes3d.shape[0]
    # trans 
    # txy = ts* 2.0 * np.random.rand(num_obj, 2) - ts # 

    txy = (ts* 2.0 * np.random.rand(num_obj, 2) - ts) * boxes3d[:, 3:5]
    txy = np.clip(txy, -2, 2.0)

    boxes3d[:, 3:4] = boxes3d[:, 3:4] +  0.2*np.random.rand(num_obj, 1) - 0.1

    boxes3d[:, :2] = boxes3d[:, :2] + txy
    # scale, 
    s = np.random.rand(num_obj, 3)*(2*ss) + (1.0 - ss)
    boxes3d[:, 3:6] = boxes3d[:, 3:6] * s
    # rot 
    max_a = 2.0*rots/180 * np.pi  # 2*5 deg
    a = np.random.rand(num_obj, 1) * max_a - 0.5*max_a
    boxes3d[:, 6:7] = boxes3d[:, 6:7] + a
    return boxes3d

def sample_neg_3DBox(boxes3d: np.ndarray):
    neg_3dbb = boxes3d.copy()
    prob = np.random.uniform(0, 1.0)
    if prob < 0.15:
        neg_3dbb[:, 2:3] = neg_3dbb[:, 2:3] - neg_3dbb[:, 5:6] - np.random.uniform(0.5, 1.5)
        return neg_3dbb
    if prob < 0.3:
        neg_3dbb[:, 2:3] = neg_3dbb[:, 2:3] - neg_3dbb[:, 5:6] - np.random.uniform(0.5, 1.5)
        return neg_3dbb
    
    num = boxes3d.shape[0]
    size =  (np.random.rand(num, 3) * 0.4 +0.8) * boxes3d[:, 3:6]
    rot = np.zeros_like(size[:, :1]) + np.random.randn(size.shape[0], 1)
    dis = 80 # 150
    xy = np.random.rand(num, 2) * (2*dis) - dis
    z = boxes3d[:, 2:3]
    neg_3dbb = np.concatenate([xy, z, size, rot], axis=1)
    # iou3d = boxes_iou3d_gpu(
    #     torch.from_numpy(neg_3dbb).to(torch.float32).cuda(), 
    #     torch.from_numpy(boxes3d[:, :7]).to(torch.float32).cuda()
    # )
    iou3d = boxes_iou3d_cpu(
        torch.from_numpy(neg_3dbb).to(torch.float32), 
        torch.from_numpy(boxes3d[:, :7]).to(torch.float32)
    )
    valid =  (iou3d>0.1).sum(1) < 1
    valid = valid.cpu().numpy()
    neg_3dbb = neg_3dbb[valid]
    return neg_3dbb

def expand_3dbb(boxes3d, expand_ratio=0.1):
    boxes3d[:, 3:6] = boxes3d[:, 3:6] * (1.0 + expand_ratio)
    return boxes3d



from aug_images import RandomHorizontalFlip, Normalize, CustomColorJitter
def get_data_transforms(train=True):
    transforms_list = []
    if train:
        transforms_list.extend([
            RandomHorizontalFlip(p=0.5),
            CustomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, is_RGB=False)
        ])

    transforms_list.extend([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_normaled=False)
    ])
    return transforms.Compose(transforms_list)

class BFDataset(Dataset):
    class_names = ["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian", "barrier", "traffic_cone"]
    dis_map  = {"car":150, "truck":150, "construction_vehicle":150, "bus":150, 
            "bicycle":50, "tricycle":50, "pedestrian":50, "barrier":50, "traffic_cone":50}
    
    def __init__(self, data_root, ann_file, input_size=(96, 96), extra_path='pkls_motovis_demotion',
                # class_names=["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian", "barrier", "traffic_cone"], 
                class_names=["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian", "traffic_cone"], 
                is_training=False, is_val=False, num_sample=32, expand_ratio=0.1,
                do_key_frame=False
                ):
        super().__init__()
        self.expand_ratio=expand_ratio
        self.is_training=is_training
        self.is_val = is_val
        self.do_key_frame=do_key_frame
        self.input_size=input_size
        self.classes=class_names
        self.data_root = data_root
        self.load_ann(ann_file)
        self.extra_path = extra_path
        self.transform = get_data_transforms(is_training)
        self.num_sample=num_sample

    def load_ann(self, ann_file):
        self.ann_file_dir = os.path.split(ann_file)[0]
        self.data = pickle.load(open(ann_file, 'rb'))
        if isinstance(self.data, dict):
            self.infos_raw = self.data['infos']
        else:
            self.infos_raw = self.data
        
        # self.infos_raw = [info for info in self.infos_raw if info['scene_token'] in ['ABC1_1734586652'] ]

        if self.is_training or self.is_val or self.do_key_frame:
            self.infos = []
            self.infos2raw_id = []
            for i, info in enumerate(self.infos_raw):
                if info['key_frame']:
                    self.infos.append(info)
                    self.infos2raw_id.append(i)
        else:
            self.infos = self.infos_raw
            self.infos2raw_id = list(range(0, len(self.infos_raw)))

    def save_ann(self, dst_ann_file):
        print('dumping ann info to ', dst_ann_file)
        pickle.dump(
            self.data, 
            open(dst_ann_file, 'wb')
        )
        print('Done')

    def __len__(self):
        return len(self.infos)


    def empty_sample(self):
        return dict(
                img= torch.zeros((0, 3, *self.input_size)).to(torch.float32),
                label= torch.zeros((0, )).to(torch.int64),
                sample_id= torch.zeros((0,)).to(torch.int64),
                ids_in_3d = torch.zeros((0,)).to(torch.int64)
            )

    def __getitem__(self, index):
        # index=1866
        # if self.is_training:
        #     # return self.gen_data(index)
        #     while True:
        #         try:
        #             return self.gen_data(index)
        #         except Exception as e:
        #             print(e)
        #             print('index: ', index, 'no gt bb')
        #             index = np.random.randint(0, self.__len__()-1)
            # return self.get_sample_from_cached()
        # index += 3529
        return self.gen_data(index)
            
    def get_pos_sample(self, info):
        if 'gt_boxes' not in info.keys():
            filepath = os.path.join(self.ann_file_dir, info['filepath'])
            extra_info = pickle.load(open(filepath, 'rb'))
            info.update(extra_info)
        boxes_lidar = info['gt_boxes']
        num_lidar_pts = info['num_lidar_pts']
        names = info['gt_names']

        mask = num_lidar_pts >3
        boxes_lidar = boxes_lidar[mask, :]
        return boxes_lidar
        # 不做距离筛选
        names = names[mask]
        dis = np.sqrt((boxes_lidar[:, :2] *boxes_lidar[:, :2]).sum(1))
        boxes_lidar_new = []
        for cls in self.classes:
            m = names == cls
            tmp_bb = boxes_lidar[m, :7]
            m2 = dis[m] < self.dis_map[cls]
            boxes_lidar_new.append(tmp_bb[m2, :])
        boxes_lidar_new = np.concatenate(boxes_lidar_new, axis=0)
        return boxes_lidar_new

    def load_extra_info(self, info):
        if 'filepath' in info.keys():
            extra_info = pickle.load(
                open(os.path.join(self.data_root, self.extra_path, info['filepath']), 'rb')
            )
            info.update(extra_info)
            return info
        return info

    def gen_data(self, index):
        # index = 12341
        info = self.infos[index]
        # pdb.set_trace()
        info = self.load_extra_info(info)
        # pdb.set_trace()
        if self.is_training or self.is_val:
            boxes_lidar = self.get_pos_sample(info)
            dim = boxes_lidar.shape[-1]
            boxes_lidar = boxes_lidar.reshape(-1, dim)[:, :7]
            boxes_lidar = aug3DBox(boxes_lidar)
            num = boxes_lidar.shape[0]
            neg_num = 0
            neg_bb = []
            while neg_num < num:
                neg_boxes_lidar = sample_neg_3DBox(boxes_lidar)
                neg_bb.append(neg_boxes_lidar)
                neg_num += neg_boxes_lidar.shape[0]
            neg_boxes_lidar = np.concatenate(neg_bb, axis=0) if len(neg_bb)>0 else np.zeros((0, 7), dtype=boxes_lidar.dtype)
            boxes_lidar = np.concatenate([boxes_lidar, neg_boxes_lidar], axis=0)

            label = np.zeros((boxes_lidar.shape[0] + neg_boxes_lidar.shape[0],), dtype=np.int32)
            label[:num] = 1
            expand_ratio = np.random.uniform(0.8, 1.2) * self.expand_ratio

            # if boxes_lidar.shape[0] > self.num_sample*2:
            #     ids = list(np.arange(0, boxes_lidar.shape[0]))
            #     ids = random.sample(ids, k = self.num_sample*2)
            #     boxes_lidar = boxes_lidar[ids, :]
            #     label = label[ids]

        else:
            # if self.is_val:
            #     boxes_lidar = self.get_pos_sample(info)
            # else:
            boxes_lidar = info['boxes_lidar']
            dim = boxes_lidar.shape[-1]
            boxes_lidar = boxes_lidar.reshape(-1, dim)
            boxes_lidar = boxes_lidar[:, :7]
            label = np.ones((boxes_lidar.shape[0],), dtype=np.int32)
            expand_ratio = self.expand_ratio

        boxes_lidar = expand_3dbb(boxes_lidar, expand_ratio)

        boxes_lidar_corners = boxes3d_lidar_corners(boxes_lidar)
        
        patchs = np.zeros((0, *self.input_size, 3), dtype=np.uint8) # []
        patchs_bb = np.zeros((0, 4), dtype=np.int32) # only used in self.patch_deduplication
        ids_in_3d = np.zeros((0,), dtype=np.int32)

        cams = list(info['cams'].keys())
        # pdb.set_trace()
        if self.is_training:
            np.random.shuffle(cams)
        # print(index)
        for cam in cams:
            # print(cam)
            cam_info = info['cams'][cam]
            img_path = os.path.join(self.data_root,  cam_info['data_path'])
            img_cv = cv2.imread(img_path)
            hw = img_cv.shape[:2]
            lidar2sensor = self.get_lidar2sensor(cam_info)
            K = np.array(cam_info['cam_intrinsic'])
            imgbb, id_in_3d = corners3d2imgbb(boxes_lidar_corners, lidar2sensor, K, expand_ratio=1.0, hw=hw)
            # if cam =='CAM_BACK_RIGHT':
            #     pdb.set_trace()
            imgbb, id_in_3d, patchs_bb, patchs, ids_in_3d = self.patch_deduplication(imgbb, id_in_3d, patchs_bb, patchs, ids_in_3d)
            # pdb.set_trace()
            # patchs.extend(crop_patch(img_cv, imgbb, self.input_size))
            # pdb.set_trace()
            cur_patchs = [p[None, ...] for p in crop_patch(img_cv, imgbb, self.input_size)]
            patchs = np.concatenate([patchs] + cur_patchs, axis=0)
            # ids_in_3d.append(id_in_3d) 
            ids_in_3d = np.concatenate([ids_in_3d, id_in_3d], axis=0)
            patchs_bb = np.concatenate([patchs_bb, imgbb], axis=0)
            if self.is_training and (len(patchs)>self.num_sample):
                break
              
        # ids_in_3d = np.concatenate(ids_in_3d, axis=0)

        if len(patchs)==0:
            return self.empty_sample()
        # patchs = np.stack(patchs, axis=0)  

        labels = label[ids_in_3d]

        if self.is_training:
            num = patchs.shape[0]
            ids = list(np.arange(0, num))
            if self.num_sample>num:
                # 有放回取
                ids.extend(random.choices(ids, k = self.num_sample - num))
            else:
                # 无放回取
                ids = random.sample(ids, k = self.num_sample)

            patchs = patchs[ids, ...]
            labels = labels[ids]
            ids_in_3d = ids_in_3d[ids]

        samples_id = np.array([index] * len(ids_in_3d))

        # self.debug_save_patchs(patchs, ids_in_3d, index)

        patchs = torch.from_numpy(patchs).float().permute(0, 3, 1,2) # [b, 3, h, w]

        patchs = self.transform(patchs)
        data = dict(
            img= patchs,
            label= torch.from_numpy(labels),
            sample_id=torch.from_numpy(samples_id),
            ids_in_3d = torch.from_numpy(ids_in_3d)
        )

        return data

    def debug_save_patchs(self, patchs, ids_in_3d, index):
        save_dir = os.path.join('vis', str(index))
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(patchs)):
            patch = patchs[i]
            id = ids_in_3d[i]
            cv2.imwrite(
                os.path.join(save_dir, '{}.png'.format(id)),
                patch
            )


    def patch_deduplication(self, cur_imgbbs: np.ndarray, cur_ids: np.ndarray, total_imgbbs: np.ndarray, total_patchs:np.ndarray, total_ids: np.ndarray):
        '''
            针对同一个id, 删除面积小的框或者patchs
            total_patchs: shape [-1, 96, 96, 3]
            total_ids: list, each ele is np.ndarry
        '''
        if (len(total_ids)==0) or (len(cur_ids)==0):
            return cur_imgbbs, cur_ids, total_imgbbs, total_patchs, total_ids
        
        
        
        cur_set = set(cur_ids)
        total_set = set(total_ids)
        and_set = cur_set & total_set

        if len(and_set)>0:
            tmp = list(cur_ids)
            de_cur_ids = np.array([tmp.index(id) for id in and_set])
            tmp = list(total_ids)
            de_total_ids = np.array([tmp.index(id) for id in and_set])

            cur_area = xyxy2area(cur_imgbbs[de_cur_ids, :])
            total_area = xyxy2area(total_imgbbs[de_total_ids, :])
    
            de_cur_mask = cur_area <= total_area
            if de_cur_mask.sum()>0:
                de_ids = de_cur_ids[de_cur_mask]
                keep_mask = np.ones_like(cur_ids).astype(np.bool)
                keep_mask[de_ids] = False
                cur_imgbbs = cur_imgbbs[keep_mask, :]
                cur_ids = cur_ids[keep_mask]

            de_total_mask = cur_area > total_area
            if de_total_mask.sum()>0:
                de_ids = de_total_ids[de_total_mask]

                keep_mask = np.ones_like(total_ids).astype(np.bool)
                keep_mask[de_ids] = False
                total_imgbbs = total_imgbbs[keep_mask, :]
                total_ids = total_ids[keep_mask]
                total_patchs = total_patchs[keep_mask, ...]
        return cur_imgbbs, cur_ids, total_imgbbs, total_patchs, total_ids

    def get_lidar2sensor(self, info):
        if 'lidar2sensor' in info.keys():
            return np.array(info['lidar2sensor'])
        if 'sensor2lidar' in info.keys():
            return np.linalg.inv(np.array(info['sensor2lidar']))
        if 'sensor2lidar_rotation' in info.keys():
            r = np.array(info['sensor2lidar_rotation']).reshape(3,3)
            t = -np.array(info['sensor2lidar_translation']).reshape(3, 1)
            r_inv = r.T
            t = r_inv @ t
            m = np.eye(4, 4, dtype=r.dtype)
            m[:3, :3] = r_inv
            m[:3, 3:4] = t
            return m
        assert False, 'cannot find lidar2sensor matrix'

    
    def deduplication_boxes3d(self, pred_scores, sample_id, ids_in_3d,
                              keys = ['boxes_lidar', 'name', 'dt_instance_inds', 'score']
                              ):
        scores, pred_lbs = torch.max(pred_scores, dim=1)
        mask_all = pred_lbs == 0
        mask_all = torch.logical_and(mask_all, scores>0.6)

        sample_id = sample_id.cpu().numpy()

        for i in set(sample_id):
            # pdb.set_trace()
            sample_mask = sample_id == i
            mask = mask_all[sample_mask]
            ids_in_3d_sample = ids_in_3d[sample_mask]
            # de_ids: need to remove 
            de_ids = ids_in_3d_sample[mask].numpy()
            if len(de_ids)==0:
                continue
            raw_id = self.infos2raw_id[i]
            info = self.infos_raw[raw_id]
            num = info[keys[0]].shape[0]

            mask = np.ones(num)
            # pdb.set_trace()
            mask[de_ids] = 0
            mask = mask.astype(np.bool_)
            
            for key in keys:
                v = info.get(key, None)
                if v is not None:
                    info[key] = info[key][mask]

    
    @staticmethod
    def collate_fn(batch_data):
        if len(batch_data)==1:
            return batch_data[0]
        keys = batch_data[0].keys()
        out = dict()
        for key in keys:
            v  = torch.cat([data[key] for data in batch_data], dim=0)
            out[key] = v
        return out
    

