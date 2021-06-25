import argparse
import os
import neptune
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data, IncrementalSet

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

def main(rank, option, task_id, save_folder, init_path):
    # Basic Options
    resume_path = init_path

    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    num_class = option.result['train']['num_init_segment']
    new_model = load_model(option, num_class)
    criterion = load_loss(option, 0, num_class)

    total_epoch = option.result['train']['total_epoch']

    # Load Init Setting
    save_module = train_module(total_epoch, new_model, criterion, multi_gpu)
    save_module.import_module(resume_path)
    new_model.load_state_dict(save_module.save_dict['model'][0])

    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        new_model.to(rank)
        new_model = DDP(new_model, device_ids=[rank])
        new_model = apply_gradient_allreduce(new_model)

    else:
        if multi_gpu:
            new_model = nn.DataParallel(new_model).to(rank)
        else:
            new_model = new_model.to(rank)

    start = 0
    end = option.result['train']['num_init_segment']

    # Training Set
    tr_target_list = list(range(start, end))
    tr_dataset = load_data(option, data_type='train')
    ex_dataset = load_data(option, data_type='exemplar')
    tr_dataset = IncrementalSet(tr_dataset, ex_dataset, start, target_list=tr_target_list, shuffle_label=True)

    if ddp:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_dataset,
                                                                     num_replicas=num_gpu, rank=rank)
        tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=pin_memory,
                                                  sampler=tr_sampler)
    else:
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4*num_gpu)


    # Save Exemplar
    if option.result['train']['train_type'] == 'icarl':
        m = int(option.result['train']['num_exemplary'] / num_class)

        for n in tr_target_list:
            n_data = tr_dataset.get_image_class(n)
            if multi_gpu:
                new_model.module.get_new_exemplar(n_data, m, rank)
            else:
                new_model.get_new_exemplar(n_data, m, rank)

        if multi_gpu:
            torch.save(new_model.module.exemplar_list, os.path.join(save_folder, 'task_%d_exemplar.pt' % (task_id)))
        else:
            torch.save(new_model.exemplar_list, os.path.join(save_folder, 'task_%d_exemplar.pt' % (task_id)))

    else:
        raise('select proper train_type')


    # Save the Module
    save_module_path = os.path.join(save_folder, 'task_%d_dict.pt' %task_id)
    save_module.export_module(save_module_path)

    save_config_path = os.path.join(save_folder, 'task_%d_config.json' %task_id)
    option.export_config(save_config_path)

    if ddp:
        cleanup()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/HDD1/sung/checkpoint/')
    parser.add_argument('--exp_name', type=str, default='imagenet_norm')
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--init_path', type=str, default='')
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)
    option.get_config_data()
    option.get_config_network()
    option.get_config_train()
    option.get_config_optimizer()

    # Resume Configuration
    resume = option.result['train']['resume']

    # GPU
    task_id = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    # Task
    init_path = args.init_path

    if ddp:
        mp.spawn(main, args=(option, task_id, save_folder, init_path), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, task_id, save_folder, init_path)
