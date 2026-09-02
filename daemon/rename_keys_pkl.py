import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
from detzero_det.datasets.motovis.evaluation.detection.data_class import OD_CATEGROY_MAPPING
import pdb

keys_pair = dict(
    gt_boxes='boxes_lidar',
    gt_names='name'
)

def rename_keys(src_pkl, dst_pkl):
    src_data = pickle.load(open(src_pkl, 'rb'))
    for info in tqdm(src_data, desc='rename key'):
        # pdb.set_trace()
        for k,v in keys_pair.items():
            info[v] = info[k] if k in info.keys() else info['anns'][k]
        
        
        name = [OD_CATEGROY_MAPPING.get(v, None) for v in info['name']]
        mask = np.array([(v is not None) for v in name], dtype=np.bool_)
        # pdb.set_trace()
        info['boxes_lidar'] = np.array(info['boxes_lidar'])[mask, :]
        info['name'] = np.array(name)[mask]
        info['score'] = np.zeros((info['boxes_lidar'].shape[0],)) + 0.2
        

    os.makedirs(os.path.split(dst_pkl)[0], exist_ok=True)
    pickle.dump(src_data, open(dst_pkl, 'wb'))

if __name__=='__main__':
    src_pkl = sys.argv[1]
    dst_pkl = sys.argv[2]

    print('rename keys: ')
    print('src pkl: ', src_pkl)
    print('dst pkl: ', dst_pkl)
    rename_keys(src_pkl, dst_pkl)






