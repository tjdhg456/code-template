import json
import numpy as np
import pickle
import argparse
import os
import neptune

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, option, resume, save_folder):
    # Basic Options
    resume_path = os.path.join(save_folder, 'last_dict.pt')

    num_gpu = len(option.result['train']['gpu'].split(','))

    total_epoch = option.result['train']['total_epoch']
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    scheduler = option.result['train']['scheduler']
    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    # Logger
    if (rank == 0) or (rank == 'cuda'):
        neptune.init('sunghoshin/imp', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzdlYWFkMjctOWExMS00YTRlLWI0MWMtY2FhNmIyNzZlYTIyIn0=')
        exp_name, exp_num = save_folder.split('/')[-2], save_folder.split('/')[-1]
        neptune.create_experiment(params={'exp_name':exp_name, 'exp_num':exp_num}, tags=[])

    # Load Model
    model = load_model(option)
    criterion = load_loss(option)
    save_module = train_module(total_epoch, model, criterion, multi_gpu)

    if resume:
        save_module.import_module(resume_path)
        model.load_state_dict(save_module.save_dict['model'][0])

    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        model.to(rank)
        model = DDP(model, device_ids=[rank])
        model = apply_gradient_allreduce(model)

        criterion.to(rank)

    else:
        if multi_gpu:
            model = nn.DataParallel(model).to(rank)
        else:
            model = model.to(rank)

    # Optimizer and Scheduler
    if resume:
        # Load Optimizer
        optimizer = load_optimizer(option, model.parameters())
        optimizer.load_state_dict(save_module.save_dict['optimizer'][0])

        # Load Scheduler
        if scheduler is not None:
            scheduler = load_scheduler(option, optimizer)
            scheduler.load_state_dict(save_module.save_dict['scheduler'][0])

    else:
        optimizer = load_optimizer(option, model.parameters())
        if scheduler is not None:
            scheduler = load_scheduler(option, optimizer)

    # Early Stopping
    early = EarlyStopping(patience=option.result['train']['patience'])

    # Dataset and DataLoader
    tr_dataset = load_data(option, data_type='train')
    val_dataset = load_data(option, data_type='val')

    if ddp:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_dataset,
                                                                     num_replicas=num_gpu, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=pin_memory,
                                                  sampler=tr_sampler)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=pin_memory,
                                                  sampler=val_sampler)
    else:
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=4*num_gpu)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4*num_gpu)

    # Mixed Precision
    mixed_precision = option.result['train']['mixed_precision']
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Training
    from module.trainer import naive_trainer
    for epoch in range(save_module.init_epoch, save_module.total_epoch):
        model.train()
        model, optimizer, save_module = naive_trainer.train(option, rank, epoch, model, criterion, optimizer, \
                                                            tr_loader, scaler, save_module, neptune, save_folder)

        model.eval()
        result = naive_trainer.validation(option, rank, epoch, model, criterion, val_loader, neptune)

        if scheduler is not None:
            scheduler.step()
            save_module.save_dict['scheduler'] = [scheduler.state_dict()]
        else:
            save_module.save_dict['scheduler'] = None


        # Save the last-epoch module
        if (rank == 0) or (rank == 'cuda'):
            save_module_path = os.path.join(save_folder, 'last_dict.pt')
            save_module.export_module(save_module_path)

            save_config_path = os.path.join(save_folder, 'last_config.json')
            option.export_config(save_config_path)


        # Early Stopping
        if multi_gpu:
            param = deepcopy(model.module.state_dict())
        else:
            param = deepcopy(model.state_dict())

        early(result['val_loss'], param, result)
        if early.early_stop == True:
            break


    if (rank == 0) or (rank == 'cuda'):
        # Save the best_model
        torch.save(early.model, os.path.join(save_folder, 'best_model.pt'))


    if ddp:
        cleanup()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/HDD1/sung/checkpoint/')
    parser.add_argument('--exp_name', type=str, default='imagenet_norm')
    parser.add_argument('--exp_num', type=int, default=1)
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)
    option.get_config_data()
    option.get_config_network()
    option.get_config_train()

    # Resume Configuration
    resume = option.result['train']['resume']
    resume_path = os.path.join(save_folder, 'last_dict.pt')
    config_path = os.path.join(save_folder, 'last_config.json')

    if resume:
        if (os.path.isfile(resume_path) == False) or (os.path.isfile(config_path) == False):
            resume = False
        else:
            option = config(save_folder)
            option.import_config(config_path)

    # Data Directory
    option.result['data']['data_dir'] = os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'])


    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    if ddp:
        mp.spawn(main, args=(option,resume,save_folder,), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, resume, save_folder)

