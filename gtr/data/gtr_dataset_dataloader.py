import copy
import logging
import torch
import torch.utils.data
from detectron2.utils.comm import get_world_size


from torch.utils.data.sampler import Sampler
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
from detectron2.data.samplers import InferenceSampler
from detectron2.data.build import worker_init_reset_seed
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils import comm
import itertools

from .custom_dataset_dataloader import MultiDatasetSampler

def single_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    assert len(batch) == 1
    return batch[0]


def build_gtr_train_loader(cfg, mapper):
    """
    Modified from detectron2.data.build.build_custom_train_loader, but supports
    different samplers
    """
    dataset_dicts = get_video_dataset_dicts(
        cfg.DATASETS.TRAIN,
        gen_inst_id=cfg.INPUT.VIDEO.GEN_IMAGE_MOTION,
    )
    sizes = [0 for _ in range(len(cfg.DATASETS.TRAIN))]
    for d in dataset_dicts:
        sizes[d['dataset_source']] += 1
    dataset = DatasetFromList(dataset_dicts, copy=False)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training videos {}".format(sampler_name))
    if len(cfg.DATASETS.TRAIN) > 1:
        assert sampler_name == 'MultiDatasetSampler'

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "MultiDatasetSampler":
        sampler = MultiDatasetSampler(cfg, sizes, dataset_dicts)
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    world_size = get_world_size()
    batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size
    assert batch_size == 1
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=True
    )
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=single_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )


def build_gtr_test_loader(cfg, dataset_name, mapper):
    """
    """
    dataset = get_video_dataset_dicts(
        [dataset_name],
    )
    dataset = DatasetFromList(dataset, copy=False)
    dataset = MapDataset(dataset, mapper)

    assert comm.is_main_process()
    sampler = SingleGPUInferenceSampler(len(dataset))

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=batch_sampler,
        collate_fn=single_batch_collator,
    )
    return data_loader


def get_video_dataset_dicts(
    dataset_names, gen_inst_id=False,
):
    assert len(dataset_names)
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in dataset_names]
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
    
    inst_count = 1000000
    tot_inst_count = 0
    video_datasets = []
    for source_id, (dataset_name, dicts) in enumerate(
        zip(dataset_names, dataset_dicts)):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)
        videos = {}
        single_video_id = 1000000
        id_map = {}
        for image in dicts:
            video_id = image.get('video_id', -1)
            if video_id == -1:
                single_video_id = single_video_id + 1
                video_id = single_video_id
            if video_id not in videos:
                videos[video_id] = {
                    'video_id': video_id, 'images': [],
                    'dataset_source': source_id}
            if gen_inst_id:
                for x in image['annotations']:
                    if 'instance_id' not in x or x['instance_id'] <= 0:
                        inst_count += 1
                        x['instance_id'] = inst_count
            if 0: # Rename inst id
                for x in image['annotations']:
                    if x['instance_id'] not in id_map:
                        tot_inst_count = tot_inst_count + 1
                        id_map[x['instance_id']] = tot_inst_count
                    x['instance_id'] = id_map[x['instance_id']]
            videos[video_id]['images'].append(image)
        video_datasets.append([v for v in videos.values()])
    video_datasets = list(itertools.chain.from_iterable(video_datasets))
    return video_datasets


class SingleGPUInferenceSampler(Sampler):
    def __init__(self, size: int):
        """
        self._world_size = 1
        self._rank = 0
        """
        self._size = size
        assert size > 0
        self._rank = 0
        self._world_size = 1

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
