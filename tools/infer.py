import os
import re
import glob
import time
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from detzero_utils import common_utils
from detzero_utils.config_utils import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from detzero_utils.model_utils import load_params_from_file

from detzero_det.datasets import build_dataloader
from detzero_det.models import build_network, load_data_to_gpu
import tqdm
import pickle
import pdb

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--save_dir', type=str, default=None, help='')
    parser.add_argument('--merge_meta_data', action='store_true', default=False, help='')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    cfg.TAG = Path(args.cfg_file).stem
    # cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[0:-1])  # remove 'xxxx.yaml'
    return args, cfg


def eval_single_ckpt(cfg, model, test_loader, args, eval_output_dir, logger, dist_test=False):
    # load checkpoint
    load_params_from_file(model, filename=args.ckpt, logger=logger, to_cpu=dist_test, fix_pretrained_weights=False)
    model.cuda()

    # start evaluation
    infer_one_epoch(
        cfg, model, test_loader, logger,
        result_dir=eval_output_dir, merge_meta_data=args.merge_meta_data
    )

def do_merge_meta_data(sample_info, dt_info):
    assert sample_info['token'] == dt_info['token']
    sample_info['dt_boxes'] = dt_info['boxes_lidar']
    sample_info['dt_names'] = dt_info['name']
    sample_info['dt_scores'] = dt_info['score']
    return sample_info

def infer_one_epoch(cfg, model, dataloader, logger, 
                   result_dir=None, merge_meta_data=False):
    
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    model.eval()
    progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        batch_dict['eval_iter'] = i
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            result = model(batch_dict)
            if len(result) > 2:
                pred_dicts, ret_dict, batch_dict = result
            else:
                pred_dicts, ret_dict = result
        
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=None
        )
        pdb.set_trace()
        if merge_meta_data:
            annos = [do_merge_meta_data(dataset.infos[i], annos[0])]
        det_annos += annos
        progress_bar.update()
    
    progress_bar.close()

    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    with open(result_dir / 'infer_result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)


    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Infer done.*****************')
    return ret_dict

def main():
    np.set_printoptions(precision=3, linewidth=500, threshold=np.inf, suppress=True)
    args, cfg = parse_config()
    dist_test = False
    total_gpus = 1

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    # pdb.set_trace()
    eval_output_dir = output_dir / 'infer'

    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)


    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_test,
        workers=args.workers,
        logger=logger,
        training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    # pdb.set_trace()
    with torch.no_grad():
        eval_single_ckpt(cfg, model, test_loader, args, eval_output_dir, logger, dist_test=dist_test)


if __name__ == '__main__':
    main()
