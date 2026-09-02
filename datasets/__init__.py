import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler
import torch.distributed as dist
from detzero_utils import common_utils

from .dataset import DatasetTemplate
from .waymo.waymo_dataset import WaymoDetectionDataset
from .motovis.motovis_dataset import MotovisDetectionDataset, OntimeDetectionDataset
from .motovis.motovis_infer_dataset import MotovisInferDataset, MotovisInferOntimeDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'WaymoDetectionDataset': WaymoDetectionDataset,
    'MotovisDetectionDataset': MotovisDetectionDataset,
    'OntimeDetectionDataset': OntimeDetectionDataset,
    'MotovisInferDataset': MotovisInferDataset,
    'MotovisInferOntimeDataset': MotovisInferOntimeDataset,


}


class ConcatDataset(Dataset):
    def __init__(self, ds_list):
        self.ds_list = ds_list

    def __len__(self):
        num = 0
        for ds in self.ds_list:
            num += len(ds)
        return num
    
    def index2ds(self, index):
        ds_id=0
        ds_index = index
        for i, ds in enumerate(self.ds_list):
            ds_len = len(ds)
            if ds_index < ds_len:
                return i, ds_index
            else:
                ds_index -= ds_len
        return ds_id, ds_index
    
    def __getitem__(self, index):
        ds_id, ds_index = self.index2ds(index)
        return self.ds_list[ds_id].__getitem__(ds_index)

class DistributedSampler(_DistributedSampler):
    '''
        is_dd: 数据分布式存储
    
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class DistributedSamplerPro(_DistributedSampler):
    '''
        is_dd: 数据分布式存储
    
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, is_dd=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.is_dd = is_dd
        if self.is_dd:
            import math
            self.max_data_len = self.get_dataset_max_len()
            # If the dataset length is evenly divisible by # of replicas, then there
            # is no need to drop any data, since the dataset will be split equally.
            if self.drop_last and len(self.max_data_len) % self.num_replicas != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible.
                # This is to ensure each rank receives the same amount of data when
                # using this Sampler.
                self.num_samples = math.ceil(
                    (len(self.max_data_len) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                self.num_samples = math.ceil(len(self.max_data_len) / self.num_replicas)  # type: ignore[arg-type]
  
            self.total_size = self.num_samples * self.num_replicas
            

    def get_dataset_max_len(self):
        assert self.is_dd, 'must be is_dd==True'
        local_len = len(self.dataset)
        max_len = torch.tensor(local_len, device="cuda")
        dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
        return max_len.item()

    def _make_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        repeat = self.total_size // len(indices)
        for i in range(repeat):
            indices += indices
        
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        return indices

    def __iter__(self):
        indices = self._make_indices()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

def build_dataloader(dataset_cfg, class_names, batch_size, dist,
                     root_path=None, workers=4, logger=None,
                     training=True, merge_all_iters_to_one_epoch=False,
                     total_epochs=0, length=0):
    
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)


    if dist:
        if training:
            is_dd = dataset_cfg.get('DD', False) # 数据分片存储
            import os
            # torchrun启动
            if is_dd and ('LOCAL_RANK' in os.environ):
                local_rank = int(os.environ['LOCAL_RANK'])
                world_size = torch.cuda.device_count()
                sampler = DistributedSampler(dataset, world_size, local_rank, shuffle=True)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            print('dist eval sample: DistributedSampler')
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    
    new_dataset = dataset
    if length > 0:
        indices = torch.arange(len(dataset))[:length]
        new_dataset = Subset(dataset, indices)

    dataloader = DataLoader(
        new_dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=False, 
        sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler
