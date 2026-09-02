import sys
import numpy as np
import cv2
from openai import OpenAI
import pickle
import base64
import os
from tqdm import tqdm
import pdb
class VQAConversions:
    def __init__(self, 
                # vllm
                model_name:str='./vllm_models/Qwen3-VL-30B-A3B-Instruct-FP8',
                api_key: str='vllm', # 可以不写
                base_url = "http://localhost:8000/v1",
                 ):
        self.model_name=model_name
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        self.samples = self.create_samples()
    
    def create_samples(self):
        sample_contents = []
        #                 neg,      neg,        pos
        samples_path = ['t68_neg.jpg', 't4_neg.png', 't68_pos.jpg']
        samples_images = [self.extract_key_frames(cv2.imread(fn))  for fn in samples_path]
        
        prompt = 'I will give you 3 samples firstly. \n Example one: \n input: '
        sample_contents.append(dict(
             role='user',
            content=[{'type':'text', 'text': prompt}]
        ))
        # pdb.set_trace()
        sample_contents.extend(samples_images[0])
        out = 'output: 0. because the pseudo-3D box is in the head of ego car,  and there is no real object.'
        sample_contents.append(dict(
            role='user',
            content=[{'type':'text', 'text': out}])
        )
        
        prompt = 'Example two: \n input: '
        sample_contents.append(dict(
             role='user',
            content=[{'type':'text', 'text': prompt}]
        ))
        sample_contents.extend(samples_images[2])
        out = 'output: 0. because the pseudo-3D box is in the head of ego car,  and there is no real object.'
        sample_contents.append(dict(
            role='user',
            content=[{'type':'text', 'text': out}])
        )
        
        prompt = 'Example three: \n input: '
        sample_contents.append(dict(
             role='user',
            content=[{'type':'text', 'text': prompt}]
        ))
        sample_contents.extend(samples_images[2])
        out = 'output: 10. because the pseudo-3D box perfectly encloses the objects.'
        sample_contents.append(dict(
            role='user',
            content=[{'type':'text', 'text': out}])
        )
        return sample_contents
        
        
        
    
    def extract_key_frames(self, frame: np.ndarray):
        self.frames_got = True
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')
        contents = []
        image_message = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_frame}",
                    "detail": "high"  # 控制图片细节级别: low/medium/high
                }} ]

        contents.append(dict(
                role='user',
                content=image_message
            ))
        return contents

    def chat_conversion(self, question, frames):

        contents = [] 
            
        contents.append(dict(
            role='user',
            content=[{'type':'text', 'text': question}]
        ))
        
        contents.extend(self.samples)
        
        contents.append(dict(
            role='user',
            content=[{'type':'text', 'text': 'Thr current input:'}]
        ))
        cur_conts = self.extract_key_frames(frames)
        
        contents.extend(cur_conts)
        
        response_q = self.client.chat.completions.create(
                                    model=self.model_name,
                                    messages=contents,
                                    stream=True,
                                    # 长度控制
                                    max_completion_tokens=512,  # vLLM 有时也支持这个参数
                                    # OpenAI 标准参数
                                    max_tokens=512,           # 最大生成 token 数
                                    timeout=2.0
                                )
                                
        res_content_q = self.get_response(response_q)
        return res_content_q
    
    def get_response(self, response):
        res_contens = ''
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                res_contens += chunk.choices[0].delta.content
        return res_contens

def get_prompt(name, size):
    prompt = \
    f'''
    In an autonomous driving front-view image, determine whether the pseudo-3D box (drawn in green) encloses the real object.
The pseudo-3D box is projected from a ground-truth 3D box of the {name} class.
Score on a scale of 0–10 (integer only) based on how well the pseudo-3D box encloses the object:
10: The pseudo-3D box perfectly encloses the object.
7–9: The object occupies about two-thirds of the pseudo-3D box, or the pseudo-3D box encloses about two-thirds of the object.
4–6: The object occupies about one-third of the pseudo-3D box, or the pseudo-3D box encloses about one-third of the object.
1–3: The pseudo-3D box encloses only a tiny portion of the object.
0: The object is completely outside the pseudo-3D box.
Output only an integer between 0 and 10.
    '''
    prompt = \
    f'''
    In an autonomous driving front-view image, determine false detection for pseudo-3D box (drawn in green):
Output 0: pseudo-3D box is on ego-vehicle hood with no real object.
Output 10: all other cases.
Output only an integer 0 or 10.
    '''
    return prompt

def filter_by_vlm(data_dir, data_dict, openai_agent: VQAConversions, ref_cs='lidar'):
    # pdb.set_trace()
    # if 'chunk_1_24dbb0c7-2ec1-422b-9558-e331ecc246a7_2025-08-13-10-12-13_p0900_1' not in data_dict['scene_token']:
    #     return data_dict, set()
    # if 'ede26' not in data_dict['token']:
    #     return data_dict, set()
    num_bb = data_dict['boxes_lidar'].shape[0]
    if num_bb==0:
        return data_dict, set()
  
    boxes_corners = boxes3d_lidar_corners(data_dict['boxes_lidar'])
    boxes_lidar = data_dict['boxes_lidar']
    # boxes3d_dis = np.linalg.norm(data_dict['boxes_lidar'][:, :2], ord=2, axis=1)
    # tmp_id = np.where(data_dict['dt_instance_inds'] == 259)[0]
    # # pdb.set_trace()
    # if tmp_id.shape[0]>0:
    #     print(data_dict['token'], boxes_lidar[tmp_id], data_dict['dt_instance_inds'][tmp_id])

    REF_CS_IS_LIDAR =  (ref_cs == 'lidar')
    cam_list = list(data_dict['cams'].keys())
    # ['CAM_FRONT_NARROW', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    # pdb.set_trace()
    # cam_list = ['CAM_FRONT', 'CAM_BACK']
    if 'CAM_FRONT_WIDE' in data_dict['cams'].keys():
        cam_list = ['CAM_FRONT_WIDE'] # t4
    else:
        cam_list = [ 'CAM_FRONT']
    res_list = [-1] * boxes_corners.shape[0]
    
    confi_thr = 3
    lat_dis = 1.1 # 3 # 10
    lon_dis = 5 # 15
    cam2image = dict()
    for cam in cam_list:
        cam_info = data_dict['cams'][cam]
        # pdb.set_trace()
        data_path = cam_info['data_path']
        if data_path is None:
            continue
        data_path = os.path.join(data_dir, cam_info['data_path'])
        if not os.path.exists(data_path):
            continue
        # pdb.set_trace()
        if REF_CS_IS_LIDAR:
            cam2lidar = cam_info['sensor2lidar']
        else:
            if 'sensor2ego' in cam_info.keys():
                cam2lidar = np.array(cam_info['sensor2ego']).reshape(4,4)
            else:
                s2e_r = np.array(cam_info['sensor2ego_rotation']).reshape(3,3)
                s2e_t = np.array(cam_info['sensor2ego_translation'])
                cam2lidar = np.eye(4, 4)
                cam2lidar[:3, :3] = s2e_r
                cam2lidar[:3, 3] = s2e_t
                
                
        lidar2cam = np.linalg.inv(cam2lidar)
        K = np.array(cam_info['cam_intrinsic']).reshape(3,3)
        corners_img, mask = corners3d2imgbb(boxes_corners, lidar2cam, K)
        # pdb.set_trace()
        for i in range(num_bb):
            # if i<33:
            #     continue
            if res_list[i] > confi_thr:
                continue
            # if boxes3d_dis[i] > 50:
            #     continue
            if REF_CS_IS_LIDAR: # rfu
                if np.abs(boxes_lidar[i, 0]) > lat_dis:
                    continue
                if boxes_lidar[i, 1] < 0 or boxes_lidar[i, 1] > lon_dis:
                    continue
            else: # flu
                if np.abs(boxes_lidar[i, 1]) > lat_dis:
                    continue
                if boxes_lidar[i, 0] < 0 or boxes_lidar[i, 0] > lon_dis:
                    continue
            if mask[i].sum() < 4: # 5:
                continue

            name = data_dict['name'][i]
            # if name not in ["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian"]:
            #     continue
            # CLASS_NAMES: ["car", "truck", "construction_vehicle", "bus", "bicycle", "tricycle", "pedestrian", "barrier", "traffic_cone"]
            if 'cam' not in cam2image.keys():
                cam2image[cam] = cv2.imread(data_path)
            img_cv = cam2image[cam]
            # pdb.set_trace()
            image_patch = create_patch(corners_img[i, :, :], mask[i, :], img_cv)
            # pdb.set_trace()
            if image_patch is None:
                continue
            h, w = image_patch.shape[:2]
            # pdb.set_trace()
            if (h < 8) or (w < 8): continue
            # pdb.set_trace()
            question = get_prompt(name, boxes_lidar[i, 3:6])
            # pdb.set_trace()
            try:
                res = openai_agent.chat_conversion(question, image_patch)
                # print(res)
                # pdb.set_trace()
                res = int(res.strip()[0])
            except Exception as e:
                print(e)
                print('image size:', image_patch.shape)
                continue
            
            # if res < confi_thr:
            #     print(data_dict['dt_instance_inds'][i])
            #     pdb.set_trace()
            #     cv2.imwrite(f'{cam}_{i}.jpg', image_patch)
            res_list[i] = max(res, res_list[i])
    # pdb.set_trace()
    res_list = np.array(res_list) 
    mask = np.logical_or(
            res_list > confi_thr,
            res_list < 0
    ) 
    
    del_instances = set()
    if mask.sum() < num_bb:
        # data_dict['name'] = data_dict['name'][mask]
        # data_dict['boxes_lidar'] = data_dict['boxes_lidar'][mask]
        # data_dict['score'] = data_dict['score'][mask]
        del_inds = data_dict['dt_instance_inds'][~mask]
        del_instances = set(del_inds)
        # data_dict['dt_instance_inds'] = data_dict['dt_instance_inds'][mask]
    return data_dict, del_instances

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

def corners3d2imgbb(corners_lidar, lidar2cam, K):
    '''
        corners_liar: [-1, 8, 3]
        lidar2cam: [4, 4]
        K: [3,3]
        return 
            corners_img: [-1, 8, 2]
            mask:        [-1, 8]

    '''
    # pdb.set_trace()
    eps = 0.1
    ones = np.ones_like(corners_lidar[:, :, :1])
    corners_lidar = np.concatenate([corners_lidar, ones], axis=2)
    corners_cam = np.einsum('bnj,ij->bni', corners_lidar, lidar2cam)[:, :, :3]

    mask = corners_cam[:, :, 2] > eps
    corners_img = np.einsum('bnj,ij->bni', corners_cam, K)
    corners_img = corners_img[:, :, :2] / np.clip(corners_img[:, :, 2:3], a_min=0.00001, a_max=None)
    corners_img = corners_img.astype(np.int32)
    return corners_img, mask
    
def is_points_inside(pts, w, h):
    '''
        pts: [n, 2]
    '''
    x = pts[:, 0]
    y = pts[:, 1]
    mask =np.logical_and(x>=0, y>=0)
    mask = np.logical_and(x < w, mask)
    mask = np.logical_and(y<h, mask)
    return mask 

def line_intersection(p1, p2, p3, p4):
    """
    计算两条线段的交点
    
    参数:
    p1, p2: 第一条线段的两个端点，格式为 (x, y)
    p3, p4: 第二条线段的两个端点，格式为 (x, y)
    
    返回:
    - 如果线段相交且不共线，返回交点坐标 (x, y)
    - 如果线段共线且有重叠，返回 None（因为有无穷多个交点）
    - 如果线段不相交，返回 None
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
     # 计算向量
    v = (x2 - x1, y2 - y1)  # B - A
    w = (x4 - x3, y4 - y3)  # D - C
    r = (x3 - x1, y3 - y1)  # C - A
    # 计算叉积 (2D cross product = determinant)
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    cross_vw = cross(v, w)
    # 如果d为0，说明线段平行或共线
    if abs(cross_vw) < 1e-10:  # 考虑浮点精度
        return None
    
    # 计算参数 t 和 u
    t = cross(r, w) / cross_vw
    u = cross(r, v) / cross_vw
    
    # 检查是否在线段上
    if 0 <= t <= 1 and 0 <= u <= 1:
        # 计算交点
        x = int(x1 + t * v[0])
        y = int(y1 + t * v[1])
        return (x, y)
    else:
        return None


def line_segment_intersection_with_image(p1, p2, w, h, p1_in, p2_in):
    """
    求线段 p1-p2 与图像边界的交点
    若不相交，返回原始p1,p2。
    若相交，返回交点交点
    """
    intersect_pts = []
    if p1_in:
        intersect_pts.append(p1)
    if p2_in:
        intersect_pts.append(p2)
    if len(intersect_pts)==2:
        return intersect_pts

    w1 = w-1
    h1 = h-1
    edges = [
        ((0,0), (w1,0)),   # 上
        ((w1,0), (w1,h1)),   # 右
        ((w1,h1), (0,h1)),   # 下
        ((0,h1), (0,0))    # 左
    ]
    
    for (a1, a2) in edges:
        pt = line_intersection(p1, p2, a1,a2)
        if pt is not None:
            intersect_pts.append(pt)
            if len(intersect_pts)==2:
                return intersect_pts
    return intersect_pts

def create_patch(image_bb, bb_mask, image_cv, expand = 1.1):
    '''
        image_bb: [8, 2]
        bb_mask: [8,]
    '''
    line_pair = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)
                 ]
    h, w = image_cv.shape[:2]
    x1 = 1000000
    x2 = -100000
    y1 = 1000000
    y2 = -100000
    pts = []
    pts_in = is_points_inside(image_bb, w, h)
    for (id1, id2) in line_pair:
        if not (bb_mask[id1] and bb_mask[id2]):
            continue   

        p1 = image_bb[id1, :]
        p2 = image_bb[id2, :]
        intersect_pts = line_segment_intersection_with_image(p1, p2, w, h, pts_in[id1], pts_in[id2])
        if len(intersect_pts)<2:
            continue
        p1, p2 = intersect_pts
        x1 = min(x1,  min(p1[0], p2[0]))
        x2 = max(x2,  max(p1[0], p2[0]))
        y1 = min(y1,  min(p1[1], p2[1]))
        y2 = max(y2,  max(p1[1], p2[1]))
        pts.append([p1, p2])
    if len(pts)<3:
        return None
    if (x2-x1) < 8 or (y2-y1)<8:
        return None
    xc = (x1+x2)//2
    yc = (y1+y2)//2
    w1 = expand * (x2-x1)
    h1 = expand * (y2 - y1)
    half_w = int(w1*0.5)
    half_h = int(h1*0.5)
    x1 = max(0,  xc - half_w)
    x2 = min(w-1, xc + half_w)
    y1 = max(0, yc - half_h)
    y2 = min(h-1, yc + half_w)
    
    shift = np.array([x1, y1])

    patch_cv = image_cv[y1:y2, x1:x2, :].copy()

    h, w = patch_cv.shape[:2]
    max_s = 1280
    if h>max_s or w>max_s:
        scale = min(max_s/h, max_s/w)
        h1, w1 = int(scale * h), int(scale*w)
    
        patch_cv = cv2.resize(patch_cv, dsize=(w1, h1), interpolation=cv2.INTER_BITS)
        for (p1, p2) in pts:
            p1 = (scale*(p1 - shift)).astype(np.int32)
            p2 = (scale*(p2 - shift)).astype(np.int32)
            cv2.line(patch_cv, p1, p2, color=(0, 255, 0), thickness=3)
    else:
        for (p1, p2) in pts:
            p1 = p1 - shift
            p2 = p2 - shift
            cv2.line(patch_cv, p1, p2, color=(0, 255, 0), thickness=3)

    return patch_cv    

from concurrent.futures import ThreadPoolExecutor

def do_filter(data_dir, pkl_fn, vlm_cfg=dict, ref_cs='lidar'):
    infos = pickle.load(open(pkl_fn, 'rb'))
    base_urls = ['http://localhost:8000/v1',
                #  'http://localhost:8001/v1'
                 ]
    openai_agent = VQAConversions(**vlm_cfg)
    num_worker =0
    del_instances_all = dict()
    if num_worker <2:
        for info in tqdm(infos, desc='filter fp by vlm'):
            info, del_instances = filter_by_vlm(data_dir, info, openai_agent, ref_cs=ref_cs)
            scene_token = info['scene_token']
            # pdb.set_trace()
            if scene_token not in del_instances_all.keys():
                del_instances_all[scene_token] = set()
            del_instances_all[scene_token] = del_instances_all[scene_token] | del_instances
    else:
        
        agents = [openai_agent]
        for base_url in base_urls[1:]:
            vlm_cfg['base_url'] = base_url
            tmp_agent = VQAConversions(**vlm_cfg)
            agents.append(tmp_agent)

        with ThreadPoolExecutor(max_workers=num_worker) as executor:
            ts = []
            # 提交任务到线程池
            for i, info in enumerate(infos):
                cur_agent_idx = i % len(agents)
                cur_agent = agents[cur_agent_idx]
                future1 = executor.submit(filter_by_vlm, data_dir, info, cur_agent, ref_cs=ref_cs)

                ts.append(future1)
         
            for fu in tqdm(ts, desc='filter fp by vlm'):
                info, del_instances =  fu.result()
                scene_token = info['scene_token']
                if scene_token not in del_instances_all.keys():
                    del_instances_all[scene_token] = set()
                del_instances_all[scene_token] = del_instances_all[scene_token] | del_instances

    for scene_token in del_instances_all.keys():
        del_ins = list(del_instances_all[scene_token])
        del_instances_all[scene_token] = np.array(del_ins)
    # pdb.set_trace()
    # print('del_instance_all', del_instances_all)
    for info in tqdm(infos, desc='del instances'):
        scene_token = info['scene_token']
        del_instance = del_instances_all[scene_token]
        if len(del_instance)==0:
            continue
        # pdb.set_trace()
        dt_instance = info['dt_instance_inds']
        mask = np.isin(dt_instance, del_instance)
        if mask.sum()>0:
            mask = ~mask
            info['name'] = info['name'][mask]
            info['boxes_lidar'] = info['boxes_lidar'][mask]
            info['score'] = info['score'][mask]
            info['dt_instance_inds'] = info['dt_instance_inds'][mask]
        
    dst_pkl = pkl_fn+ '_vlm_f.pkl'
    with open(dst_pkl, 'wb') as fp:
        pickle.dump(infos, fp)

if __name__ == '__main__':
    '''
            - NAME: filter_by_vlm
      REF_CS_IS_LIDAR: True
      VLM_CFG: {
        base_url: 'http://localhost:8000/v1',
        model_name: './vllm_models/Qwen3-VL-30B-A3B-Instruct-FP8',
        # 硅基流动的key
        api_key: 'sk-xnvcfuoqibbbycakzjstmjslpqlgwqetcumohyswvulvdddv'
      }
    '''
    data_dir = sys.argv[1]
    pkl_fn = sys.argv[2]
    ref_cs = sys.argv[3]
    vlm_cfg={
        'base_url': 'http://localhost:8000/v1',
        'model_name': './vllm_models/Qwen3-VL-30B-A3B-Instruct-FP8',
        # 硅基流动的key
        'api_key': 'sk-xnvcfuoqibbbycakzjstmjslpqlgwqetcumohyswvulvdddv'
    }
    do_filter(data_dir, pkl_fn, vlm_cfg, ref_cs)