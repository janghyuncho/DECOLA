# Copyright (c) Facebook, Inc. and its affiliates.
# From https://github.com/facebookresearch/Detic/blob/main/train_net.py.
# Modified by Jang Hyun Cho. 

import logging
import os
import sys
from collections import OrderedDict
import torch
import numpy as np 
from torch.nn.parallel import DistributedDataParallel
import time
import json 
import datetime
import wandb 

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
    LVISEvaluator,
    COCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.utils.logger import setup_logger
from torch.cuda.amp import GradScaler

sys.path.insert(-1, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config

sys.path.insert(-1, 'third_party/Deformable-DETR')
from decola.config import add_detic_config
from decola.modeling.decola.decola_config import add_decola_config 
from decola.data.custom_build_augmentation import build_custom_augmentation
from decola.data.custom_dataset_dataloader import  build_custom_train_loader
from decola.data.custom_dataset_mapper import CustomDatasetMapper, DetrDatasetMapper
from decola.custom_solver import build_custom_optimizer
from decola.evaluation.oideval import OIDEvaluator
from decola.evaluation.custom_coco_eval import CustomCOCOEvaluator
from decola.modeling.utils import reset_cls_test
from decola.evaluation.custom_lvis_evaluator import CustomLVISEvaluator
from decola.evaluation.null_evaluator import NullEvaluator
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        if cfg.MODEL.RESET_CLS_TESTS:
            reset_cls_test(
                model,
                cfg.MODEL.TEST_CLASSIFIERS[d],
                cfg.MODEL.TEST_NUM_CLASSES[d])
                
        # NOTE: ugly hack for getting annotation during testing.
        if cfg.GEN_PSEDO_LABELS:
            mapper = CustomDatasetMapper(cfg, False) 
        else:
            mapper = DatasetMapper(cfg, False)
        mapper.is_train = True 
        
        data_loader = build_detection_test_loader(cfg, dataset_name, batch_size=cfg.TEST.BATCH_SIZE, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == 'lvis' or cfg.GEN_PSEDO_LABELS:
            evaluator = CustomLVISEvaluator(dataset_name, cfg, True, output_folder, 
                                            max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE,
                                            max_dets_per_class_per_image=cfg.TEST.DETECTIONS_PER_CLASS_IMAGE,
                                            just_store_predictions=cfg.TEST.JUST_STORE_PREDICTIONS,
                                            prediction_save_filename=cfg.TEST.PREDICTION_SAVE_FILENAME)
        
        elif evaluator_type == 'coco':
            if dataset_name == 'coco_generalized_zeroshot_val':
                # Additionally plot mAP for 'seen classes' and 'unseen classes'
                evaluator = CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
            else:
                evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'oid':
            evaluator = OIDEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        wandb.log(results)
    return results

def do_train(cfg, model, resume=False):
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

    start_iter = checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    if not resume:
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter,
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

    use_custom_mapper = cfg.WITH_IMAGE_LABELS
    MapperClass = CustomDatasetMapper #if use_custom_mapper else DatasetMapper
    mapper = MapperClass(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DetrDatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == 'DETR' else \
        MapperClass(cfg, True, 
        augmentations=build_custom_augmentation(cfg, True), 
        recompute_boxes=(cfg.INPUT.CUSTOM_AUG == 'LSJ'))
    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    else:
        data_loader = build_custom_train_loader(cfg, mapper=mapper)

    if cfg.FP16:
        scaler = GradScaler()

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
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            
            # NOTE: exclude cardinality error logs from ddetr (no contribution to total loss)
            losses_reduced = sum(loss for key, loss in loss_dict_reduced.items() \
                                    if "loss" in key)
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            if cfg.FP16:
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                losses.backward()
                optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter):
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
    add_detic_config(cfg)
    add_decola_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.DEBUG:
        cfg.OUTPUT_DIR  = cfg.OUTPUT_DIR.replace('/auto', '/')
        cfg.OUTPUT_DIR += 'debug'
        logger.info('[DEBUG] OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    elif '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, \
        distributed_rank=comm.get_rank(), name="decola")
    return cfg


def main(args):
    cfg = setup(args)
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        run_name = cfg.WANDB.RUN_NAME
        wandb.init(project=cfg.WANDB.PROJECT, sync_tensorboard=True, name=run_name,
         group=cfg.WANDB.GROUP, config=cfg, dir=cfg.OUTPUT_DIR)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = do_test(cfg, model)
        if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
            wandb.finish()
        return res 

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=cfg.FIND_UNUSED_PARAM, 
            # static_graph=cfg.USE_ACT_CHECKPOINT, # for activation checkpoint
        )

    do_train(cfg, model, resume=args.resume)
    res = do_test(cfg, model)
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        wandb.finish()
    return res 


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        if args.dist_url == 'host':
            args.dist_url = 'tcp://{}:12345'.format(
                os.environ['SLURM_JOB_NODELIST'])
        elif not args.dist_url.startswith('tcp'):
            tmp = os.popen(
                    'echo $(scontrol show job {} | grep BatchHost)'.format(
                        args.dist_url)
                ).read()
            tmp = tmp[tmp.find('=') + 1: -1]
            args.dist_url = 'tcp://{}:12345'.format(tmp)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
