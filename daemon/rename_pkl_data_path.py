import pickle
from tqdm import tqdm
import pdb

def modify_data_infos(data):
    if isinstance(data, dict):
        infos = data['infos']
    else:
        infos = data
    prefix='/Users/ace/workspace/ad/datasets/t4/data/'
    for info in tqdm(infos):
        info['lidar_path'] = info['lidar_path'][len(prefix):]
 
        for cam in info['cams']:
            info['cams'][cam]['data_path'] = info['cams'][cam]['data_path'][len(prefix):]
  

    if isinstance(data, dict):
        data['infos'] = infos
    else:
        data = infos
    return data

if __name__=='__main__':
    src_pkl = '/data/dataset/t4/chunk_5_24dbb0c7-2ec1-422b-9558-e331ecc246a7_2025-08-08-09-18-41_p0900_5_infos_90samples.pkl'
    dst_pkl = '/data/dataset/t4/chunk_5_24dbb0c7-2ec1-422b-9558-e331ecc246a7_2025-08-08-09-18-41_p0900_5_infos_90samples_local.pkl'
    data = pickle.load(open(src_pkl, 'rb'))
    data = modify_data_infos(data)
    pickle.dump(data,open(dst_pkl, 'wb'))
    pass