import os
import time
import pickle
import random
import argparse
import numpy as np
import multiprocessing

from tqdm import tqdm


TEST_CLIPS = ['ABC1_1737593566', 'ABC2_1735976715', 'ABC1_1736477365', 'ABC1_1738998404', 
              'ABC1_1735979438', 'ABC1_1738034458', 'ABC1_1735094260', 'ABC1_1739251552', 
              'ABC2_1739351823', 'ABC1_1737254416', 'ABC2_1739267740', 'ABC2_1735969545', 
              'ABC1_1734935839', 'ABC2_1736985396', 'ABC1_1736291285', 'ABC1_1735874097', 
              'ABC1_1736292357', 'ABC1_1734593776', 'ABC1_1739352588', 'ABC2_1736221669', 
              'ABC1_1734586648', 'ABC2_1739332338', 'ABC2_1736562861', 'ABC2_1736486844', 
              'ABC2_1727580042', 'ABC2_1738725057', 'ABC2_1734940381', 'ABC1_1735370692', 
              'ABC1_1734684442', 'ABC1_1739241387', 'ABC1_1735271467', 'ABC2_1738984926', 
              'ABC2_1735962053', 'ABC1_1738400591', 'ABC1_1738568366', 'ABC2_1736650441', 
              'ABC1_1738398654', 'ABC1_1736988159', 'ABC1_1737160979', 'ABC1_1738289558',
              'ABC001_1733642874', 'ABC001_1733643000', 'ABC001_1733643104', 'ABC001_1733643312', 'ABC001_1733643332',
              'ABC002_1729065219', 'ABC002_1729065279', 'ABC002_1729065519', 'ABC002_1729065579', 'ABC002_1729065639'
]


def process_train_clips(args, train_clips):
    num_clips = len(train_clips)
    if num_clips == 0:
        return
    
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count()
    num_processes = min(num_workers, num_clips)

    processed_infos = []
    processed_clips = []
    
    if num_processes > 1:
        print(f'Using {num_processes} processes to process {num_clips} train clips.')
        
        avg_times = []
        num_processed = 0
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            s = time.time()
            for clip_name, batch_infos in pool.imap_unordered(process_single_clip, train_clips):
                processed_infos.extend(batch_infos)
                processed_clips.append(clip_name)
                
                e = time.time()
                avg_times.append(e - s)
                s = e

                num_processed += 1
                if num_processed % 2 == 0:
                    eta = np.mean(avg_times) * (num_clips - num_processed)
                    print(f'[{num_processed}/{num_clips}] - done. eta: {eta:.2f}s')
    else:
        for clip in tqdm(train_clips):
            clip_name, batch_infos = process_single_clip(clip)

            processed_infos.extend(batch_infos)
            processed_clips.append(clip_name)

    if args.split_pickles:
        flag = '_split'
    else:
        flag = ''

    if args.filter_no_agents:
        flag += '_dropempty'
    
    if args.only_keyframes:
        flag += '_keyframe'
    
    if args.ego_fut_trajs_num > 0:
        flag += f'_{args.ego_fut_trajs_num}fut_trajs'

    num_processed_clips = len(processed_clips)
    num_processed_frames = len(processed_infos)
    if num_processed_frames > 0:
        print(f'---------- train ----------')
        print(f'{num_processed_clips} clips, {num_processed_frames} samples')

        save_filepath = f'{args.dataroot}/motovis_train_{num_processed_clips}clips_{num_processed_frames}frames{flag}.pkl'
        with open(save_filepath, 'wb') as f:
            pickle.dump(processed_infos, f)
        print(f'保存到 {save_filepath}')


def process_test_clips(args, test_clips):
    num_clips = len(test_clips)
    if num_clips == 0:
        return
    
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count()
    num_processes = min(num_workers, num_clips)

    processed_infos = []
    processed_clips = []
    
    if num_processes > 1:
        print(f'Using {num_processes} processes to process {num_clips} test clips.')
        
        avg_times = []
        num_processed = 0
        
        with multiprocessing.Pool(processes=num_processes) as pool:
            s = time.time()
            for clip_name, batch_infos in pool.imap_unordered(process_single_clip, test_clips):
                processed_infos.extend(batch_infos)
                processed_clips.append(clip_name)
                
                e = time.time()
                avg_times.append(e - s)
                s = e

                num_processed += 1
                if num_processed % 2 == 0:
                    eta = np.mean(avg_times) * (num_clips - num_processed)
                    print(f'[{num_processed}/{num_clips}] - done. eta: {eta:.2f}s')
    else:
        for clip in tqdm(test_clips):
            clip_name, batch_infos = process_single_clip(clip)

            processed_infos.extend(batch_infos)
            processed_clips.append(clip_name)

    flag = ''
    if args.filter_no_agents:
        flag += '_dropempty'
    
    if args.only_keyframes:
        flag += '_keyframe'
    
    if args.ego_fut_trajs_num > 0:
        flag += f'_{args.ego_fut_trajs_num}fut_trajs'

    num_processed_clips = len(processed_clips)
    num_processed_frames = len(processed_infos)
    if num_processed_frames > 0:
        print(f'---------- test ----------')
        print(f'{num_processed_clips} clips, {num_processed_frames} samples')

        save_filepath = f'{args.dataroot}/motovis_test_{num_processed_clips}clips_{num_processed_frames}frames{flag}.pkl'
        with open(save_filepath, 'wb') as f:
            pickle.dump(processed_infos, f)
        print(f'保存到 {save_filepath}')


def process_single_clip(args_tuple):
    args, pkl_filepath, batch_name, clip_name, split_pickles = args_tuple
    with open(pkl_filepath, 'rb') as f:
        clip_samples = pickle.load(f)
    
    keep_samples = []
    for sample in clip_samples:
        if args.only_keyframes:
            if not sample.get('key_frame', False):
                continue

        if args.filter_no_agents:
            if len(sample['anns']['gt_boxes']) == 0:
                continue

        if args.ego_fut_trajs_num is not None:
            if sample.get('ego_fut_trajs_num', 0) < args.ego_fut_trajs_num:
                continue
        
        if split_pickles:
            sample_token = sample['token']
            keep_samples.append(
                {
                    'filepath': f'{batch_name}/split_pickles/{clip_name}/{sample_token}.pkl',
                    'key_frame': sample.get('key_frame', False),
                    'timestamp': sample['timestamp'],
                    'scene_token': clip_name,
                    'token': sample_token,
                    'ego_fut_trajs_num': sample.get('ego_fut_trajs_num', 0)
                }
            )

            save_dir = os.path.join(args.dataroot, batch_name, 'split_pickles', clip_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            with open(f'{save_dir}/{sample_token}.pkl', 'wb') as f:
                pickle.dump(sample, f)
        else:
            keep_samples.append(sample)
    
    keep_samples = list(sorted(keep_samples, key=lambda e: e["timestamp"]))

    return clip_name, keep_samples


def main(args: argparse.Namespace) -> None:
    batch_names = args.batch_names

    # 先获取所有clip的信息
    total_clips = {}
    for batch_name in batch_names:
        batch_root = os.path.join(args.dataroot, batch_name)
        if not os.path.exists(batch_root):
            continue

        for pkl_file in os.listdir(batch_root):
            clip_name, extname = os.path.splitext(pkl_file)
            if extname != '.pkl':
                continue

            total_clips[clip_name] = [
                args,
                os.path.join(batch_root, pkl_file),
                batch_name,
                clip_name,
                args.split_pickles
            ]
    
    num_clips = len(total_clips)
    print(f'Found {num_clips} clips')
    if num_clips == 0:
        exit()
    
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = multiprocessing.cpu_count()

    test_clips = []
    train_clips = []
    if args.num_test > 0:
        assert args.num_test <= num_clips
        test_clip_names = random.sample(list(total_clips.keys()), k=args.num_test)
        for clip_name in test_clip_names:
            clip_info = total_clips[clip_name]
            clip_info[-1] = False
            test_clips.append(clip_info)
    
    for clip_name in total_clips:
        if clip_name in test_clips:
            continue

        if clip_name in TEST_CLIPS:
            clip_info = total_clips[clip_name]
            clip_info[-1] = False
            test_clips.append(clip_info)
            continue

        train_clips.append(total_clips[clip_name])
    
    # 训练集
    process_train_clips(args, train_clips)

    # 测试集
    process_test_clips(args, test_clips)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='合并pkl文件')
    parser.add_argument('dataroot', type=str, help='pkl文件所在目录')
    parser.add_argument('--num_test', type=int, default=0, help='测试集clip数')
    parser.add_argument('--batch_names', type=str, nargs='+', required=True, help='要处理的batch名称列表')
    parser.add_argument('--filter_no_agents', action='store_true', help='是否丢弃没有目标的帧')
    parser.add_argument('--only_keyframes', action='store_true', help='是否只保留关键帧')
    parser.add_argument('--ego_fut_trajs_num', type=int, default=-1, help='有效轨迹帧数, 如果为-1,该条件实效')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--split_pickles', action='store_true')
    args = parser.parse_args()
    '''
        python cat_pkls.py /mnt/cfs/e2e/datasets/t4/pkls_motovis_tmp --batch_names 2025-10-17 --split_pickles
    '''
    main(args)
