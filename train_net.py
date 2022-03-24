import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import sys

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    COCOEvaluator,
)

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.solver import build_optimizer
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config

from gtr.config import add_gtr_config
from gtr.data.custom_build_augmentation import build_custom_augmentation
from gtr.data.custom_dataset_dataloader import  build_custom_train_loader
from gtr.data.custom_dataset_mapper import CustomDatasetMapper
from gtr.data.gtr_dataset_dataloader import build_gtr_train_loader
from gtr.data.gtr_dataset_dataloader import build_gtr_test_loader
from gtr.data.gtr_dataset_mapper import GTRDatasetMapper
from gtr.costom_solver import build_custom_optimizer
from gtr.evaluation.custom_lvis_evaluation import CustomLVISEvaluator
from gtr.evaluation.mot_evaluation import MOTEvaluator
from gtr.modeling.freeze_layers import check_if_freeze_model


logger = logging.getLogger("detectron2")

def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type)
    return total_norm


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis":
            evaluator = CustomLVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'mot':
            evaluator = MOTEvaluator(dataset_name, cfg, \
                    False, output_folder)
        else:
            assert 0, evaluator_type

        if not cfg.VIDEO_INPUT:
            mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else \
                DatasetMapper(
                    cfg, False, augmentations=build_custom_augmentation(cfg, False))
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
            results[dataset_name] = inference_on_dataset(
                model, data_loader, evaluator)
        else:
            if not comm.is_main_process():
                continue
            # TODO (Xingyi): currently holistic test only works on 1 gpus 
            #   due to unknown system issue. Try to fix.
            torch.multiprocessing.set_sharing_strategy('file_system')
            if cfg.INPUT.TEST_INPUT_TYPE == 'default':
                mapper = GTRDatasetMapper(cfg, False)
            else:
                mapper = GTRDatasetMapper(
                    cfg, False, augmentations=build_custom_augmentation(cfg, False))
            data_loader = build_gtr_test_loader(cfg, dataset_name, mapper)
            # TODO (Xingyi): create a new video inference pipeline
            results[dataset_name] = inference_on_dataset(
                model, data_loader, evaluator, 
            )
        
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, resume=False):
    model = check_if_freeze_model(model, cfg)
    model.train()
    if cfg.SOLVER.USE_CUSTOM_SOLVER:
        optimizer = build_custom_optimizer(cfg, model)
    else:
        assert cfg.SOLVER.OPTIMIZER == 'SGD'
        assert cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE != 'full_model'
        assert cfg.SOLVER.BACKBONE_MULTIPLIER == 1.
        optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    DatasetMapperClass = GTRDatasetMapper if cfg.VIDEO_INPUT else \
        CustomDatasetMapper
    mapper = DatasetMapperClass(
        cfg, True, augmentations=build_custom_augmentation(cfg, True))
    if cfg.VIDEO_INPUT:
        data_loader = build_gtr_train_loader(cfg, mapper=mapper)
    else:
        data_loader = build_custom_train_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)
            
            losses = sum(
                loss for k, loss in loss_dict.items() if 'loss' in k)
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for k, loss in loss_dict_reduced.items() \
                if 'loss' in k)
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_gtr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="centernet")
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
