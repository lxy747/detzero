
import argparse
from pathlib import Path
import datetime
import pdb
from detzero_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from detzero_utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--db_info_path', type=str, required=True, help='相对目录, 相对于dataset.root_dir')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return args, cfg

if __name__ == '__main__':
    '''
        python create_motovis_db_infos.py --cfg_file  cfgs/det_model_cfgs/centerpoint_motovis_1sweep.yaml --db_info_path ./dbinfos/1002clips
    '''
    args, cfg = parse_config()

    from detzero_det.datasets import __all__ as ds_dict

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / ('log_create_db_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=0)
    cfg.DATA_CONFIG.DATA_AUGMENTOR = None
    cfg.DATA_CONFIG.BALANCED_RESAMPLING = False
    dataset = ds_dict[cfg.DATA_CONFIG.DATASET](
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,
        logger = logger
    )

    # pdb.set_trace()
    dataset.use_next_frame=False
    dataset.create_groundtruth_database(
        used_classes=dataset.class_names, max_sweeps=1, save_path=args.db_info_path, num_worker=args.workers
    )
    pass
