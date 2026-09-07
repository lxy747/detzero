import pickle
import time
from pathlib import Path
import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter

from detzero_utils import common_utils

from detzero_det.models import load_data_to_gpu
import pdb

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])
    max_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[-1]
    disp_dict['recall_%s' % str(max_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(max_thresh)], metric['recall_rcnn_%s' % str(max_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False,
                   save_to_file=False, result_dir=None, save_tb=True):
    
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'data'

    tb_output_dir = result_dir / 'tensorboard'
    if save_tb:
        tb_output_dir.mkdir(parents=True, exist_ok=True)
        tb_log = SummaryWriter(log_dir=str(tb_output_dir), flush_secs=1) \
            if cfg.LOCAL_RANK == 0 else None

    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    
    # pdb.set_trace()
    # res_pkl = result_dir / 'result.pkl'
    # if res_pkl.exists():
    #     det_annos = pickle.load(open(res_pkl, 'rb'))
    #     result_str, result_dict = dataset.evaluation(
    #         det_annos, class_names,
    #         eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    #         output_path=final_output_dir
    #     )
    # pdb.set_trace()

    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
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

        if ('tb_dict' in pred_dicts[0].keys()) and (cfg.LOCAL_RANK == 0):
            # Local Rank
            tb_dict = pred_dicts[0]['tb_dict']
            for key, val in tb_dict.items():    
                if key.startswith('visual:'):
                    tb_log.add_image('eval/'+key, val, i)
                else:
                    tb_log.add_scalar('eval/'+key, val, i)

        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )

        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
    
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), 
                                                    tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]
    
    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def do_merge_meta_data(sample_infos, dt_infos):
    assert len(sample_infos) == len(dt_infos)
    for sample_info, dt_info in zip(sample_infos, dt_infos):
        assert sample_info['token'] == dt_info['token']
        sample_info['boxes_lidar'] = dt_info['boxes_lidar']
        sample_info['name'] = dt_info['name']
        sample_info['dt_instance_inds'] = dt_info.get('dt_instance_inds', None)
        sample_info['score'] = dt_info['score']
    return sample_infos

def cvt_dt2eval_type(dt_infos, class_names):
    det_anno ={
                'name': dt_infos['name'],
                'label': dt_infos['label'] if 'label' in dt_infos.keys() else np.array([class_names.index(n)  for n in dt_infos['name']], dtype=np.int32),
                'score': dt_infos['score'],
                'boxes_lidar': dt_infos['boxes_lidar'],
                'scene_token': dt_infos['scene_token'],
                'token': dt_infos['token'],
                # 'frame_id': dt_infos['frame_id'] if 'frame_id' in dt_infos.keys() else Path(dt_infos['lidar_path']).stem,
            }
    return det_anno

def infer_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False,
                   save_to_file=False, result_dir=None, save_tb=False, infer_mode=1):
    
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'data'

    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)


    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s Infer *****************' % epoch_id)
    print('dist_test', dist_test)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        print(f'num_gpus: {num_gpus}, local_rank: {local_rank}')
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    info_token2index = dict()
    for i, info in enumerate(dataset.infos):
        token = info['token']
        info_token2index[token] = i

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    print('strat to eval, datalen: ', len(dataloader))
    # dataloader_iter = iter(dataloader)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        # batch_dict = next(dataloader_iter)
        # print('iter: ', i)
        # pdb.set_trace()
        batch_dict['eval_iter'] = i
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            result = model(batch_dict)
            # pdb.set_trace()
            if len(result) > 2:
                pred_dicts, ret_dict, batch_dict = result
            else:
                pred_dicts, ret_dict = result

        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        # pdb.set_trace()
        if infer_mode>1:
            # id1 = len(det_annos) 
            # id2 = id1 + len(annos)
            # annos = do_merge_meta_data(dataset.infos[id1: id2], annos)
            # pdb.set_trace()
            info_idxs = [info_token2index[ann['token']]    for ann in annos]
            tmp_infos = [dataset.infos[idx] for idx in info_idxs]
            annos = do_merge_meta_data(tmp_infos, annos)
        det_annos += annos
        # pdb.set_trace()
        if cfg.LOCAL_RANK == 0:
            progress_bar.update()
    
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), 
                                                    tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    with open(result_dir / 'infer_result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    
    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Infer done.*****************')
    return ret_dict

def eval_results(cfg, det_pkl, dataloader, epoch_id, logger, dist_test=False,
                   save_to_file=False, result_dir=None, save_tb=True):
    
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    
    logger.info('\n Load det_annos from %s \n' % det_pkl)
    
    det_annos = pickle.load(open(det_pkl, 'rb'))
    # pdb.set_trace()
    det_annos = [dt_info for dt_info in det_annos if dt_info['key_frame'] > 0 ]
    det_annos = [cvt_dt2eval_type(dt_info, cfg.CLASS_NAMES) for dt_info in det_annos]
    
    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=None
    )
    logger.info(result_str)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return result_str


if __name__ == '__main__':
    pass
