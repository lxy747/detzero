

data_dir=/mnt/cfs/e2e/datasets/ontime
src_pkl=/mnt/cfs/e2e/datasets/ontime/pkls_nuscenes_10hz_raw_lidar/ontime_1001clips_200070samples_world_clip.pkl
dst_pkl=./data/ontime/ontime_1001clips_200070samples_world_clip_numpts.pkl

python motovis_data_process.py --data_root ${data_dir} \
    --src_pkl ${src_pkl} \
    --dst_pkl ${dst_pkl} \
    --workers 2 --mode calc_num_lidar_pts