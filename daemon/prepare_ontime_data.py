import pickle
import os
from tqdm import tqdm
import pdb
# src_pkl = '/data/dataset/ontime/pkls/ontime_20250816_1001clips_40014frames_split_pickles.pkl'
# dst_pkl= '/data/dataset/ontime/pkls/ontime_20250816_1001clips_40014frames_split_pickles_with_gt_names.pkl'
src_pkl = '/data/dataset/ontime/pkls/ontime_demo_10clips_400frames_split_pickles.pkl'
dst_pkl= '/data/dataset/ontime/pkls/ontime_demo_10clips_400frames_split_pickles.pkl'
# src_pkl = '/data/dataset/ontime/pkls/ontime_20250804_100clips_4000frames_split_pickles.pkl'
# dst_pkl= '/data/dataset/ontime/pkls/ontime_20250804_100clips_4000frames_split_pickles.pkl'
src_pkl = '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes/E2E_citynoa_20251030_2021_nuscenes_2021clips_404198samples_split_train.pkl'
dst_pkl= '/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes/E2E_citynoa_20251030_2021_nuscenes_2021clips_404198samples_split_train_with_gtnames.pkl'

def main():
    root_dir = os.path.split(src_pkl)[0]
    data = pickle.load(open(src_pkl, 'rb'))
    for info in tqdm(data['infos']):
        filepath = os.path.join(root_dir, info['filepath'])
        extra_info  = pickle.load(open(filepath, 'rb'))
        info['gt_names'] = extra_info['gt_names']
        info['token'] = extra_info['token']
    # import pdb; pdb.set_trace()
    pickle.dump(data, open(dst_pkl, 'wb'))


if __name__=='__main__':
    main()
