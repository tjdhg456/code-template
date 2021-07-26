#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
import sys
import argparse
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.data import DatasetMapper, build_detection_test_loader

from detectron2.modeling import GeneralizedRCNNWithTTA
import yaml
import pathlib
import module.hook
import neptune.new as neptune
from glob import glob

def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.
    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.
    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:
Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    parser.add_argument("--output_dir", type=str, default='checkpoint')
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--data_dir", type=str, default="/home/sung/dataset/KDN/0719_detection", help="number of gpus *per machine*")
    return parser

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg, logger):
        self.logger = logger
        super(Trainer, self).__init__(cfg)


    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(module.hook.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results, logger=self.logger))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))

        # Custom Hook (Add-On) -> Loss Logging
        ret.append(module.hook.LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            ),
            logger=self.logger))
        return ret


def setup(args, resume=False):
    """
    Create base_configs and perform basic setups.
    """

    if not resume:
        # load config from file and command-line arguments
        with open(args.config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        config_data['_BASE_'] = os.path.join(args.base_dir, config_data['_BASE_'])
        config_data['MODEL']['WEIGHTS'] = os.path.join(args.base_dir, config_data['MODEL']['WEIGHTS'])
        config_data['OUTPUT_DIR'] = os.path.join(args.base_dir, args.output_dir)

        with open(args.config_file.replace('.yaml', '_new.yaml'), "w") as f:
            yaml.dump(config_data, f)

        args.config_file = args.config_file.replace('.yaml', '_new.yaml')

    # Configure File
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_data(args):
    from detectron2.data.datasets import register_coco_instances
    DATA_DIR = args.data_dir

    register_coco_instances("KDN_train", {}, os.path.join(DATA_DIR, 'annotation', "det_train2017.json"),
                                             os.path.join(DATA_DIR, 'images_enhancement', 'dark', '400'))
    register_coco_instances("KDN_val", {}, os.path.join(DATA_DIR, 'annotation', "det_val2017.json"),
                            os.path.join(DATA_DIR, 'images_enhancement', 'dark', '400'))

def main(args):
    # args.config_file = '/home/sung/src/code-template/detectron2/checkpoint_enhancement/config.yaml'
    args.resume = False
    args.log = True
    args.neptune_id = None

    cfg = setup(args, resume=args.resume)

    # Logger
    token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='

    if args.log:
        mode = 'async'
    else:
        mode = 'debug'

    if args.resume:
        neptune_id = os.path.basename(glob(os.path.join(cfg.OUTPUT_DIR, 'ID_*'))[0])
        neptune_id = neptune_id.split('_')[-1]
        run = neptune.init('sunghoshin/test', api_token=token,
                           capture_stdout=False,
                           capture_stderr=False,
                           capture_hardware_metrics=False,
                           run=neptune_id,
                           mode=mode
                           )

    else:
        run = neptune.init('sunghoshin/test', api_token=token,
                           capture_stdout=False,
                           capture_stderr=False,
                           capture_hardware_metrics=False,
                           mode=mode
                           )

        neptune_id = str(run.__dict__['_short_id'])
        with open(os.path.join(cfg.OUTPUT_DIR,  'ID_%s' %neptune_id), 'w') as f:
            pass

    # Register New Dataset
    register_data(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        if comm.is_main_process():
            verify_results(cfg, res)
        return res


    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg, logger=run)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    # Load Default Arguments
    args = default_argument_parser().parse_args()

    # BASE FOLDER
    args.base_dir = str(pathlib.Path(__file__).parent.resolve())

    # Launch
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )