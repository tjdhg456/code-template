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

def main(rank, option, task_id, save_folder):
    # Basic Options
    if task_id == 0:
        resume = False
    else:
        resume = True
        resume_path = os.path.join(save_folder, 'task_%d_dict.pt' %(task_id-1))

    num_gpu = len(option.result['train']['gpu'].split(','))

    total_epoch = option.result['train']['total_epoch']
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    scheduler = option.result['train']['scheduler']
    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']


    # Load Model
    def calc_num_class(task_id):
        if task_id == 0:
            num_class = option.result['train']['num_init_segment']
        else:
            num_class = option.result['train']['num_init_segment'] + option.result['train']['num_segment'] * task_id
        return num_class

    if task_id == 0:
        new_class = calc_num_class(0)
        old_class = calc_num_class(0)
    else:
        new_class = calc_num_class(task_id)
        old_class = calc_num_class(task_id - 1)

    old_model = load_model(option, old_class)
    criterion = load_loss(option)

    if resume:
        save_module = train_module(total_epoch, old_model, criterion, multi_gpu)
        save_module.import_module(resume_path)
        old_model.load_state_dict(save_module.save_dict['model'][0])


    # New Model
    if option.result['train']['pretrain_new_model'] and task_id > 0:
        new_model = deepcopy(old_model)
        new_model.model_cls.fc = nn.Linear(new_model.model_cls.fc.in_features, new_class)
        new_model.model_cls.fc.weight.data[:old_class] = old_model.model_cls.fc.weight.data
        new_model.model_cls.fc.bias.data[:old_class] = old_model.model_cls.fc.bias.data
    else:
        new_model = load_model(option, new_class)

    save_module = train_module(total_epoch, new_model, criterion, multi_gpu)

    # Load Old Exemplary Samples
    if (option.result['train']['num_exemplary'] > 0) and (task_id > 0):
        new_model.exemplar_list = torch.load(os.path.join(save_folder, 'task_%d_exemplar.pt' %(task_id-1)))

    else:
        new_model.exemplar_list = []

    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        old_model.to(rank)
        new_model.to(rank)

        old_model = DDP(old_model, device_ids=[rank])
        new_model = DDP(new_model, device_ids=[rank])

        old_model = apply_gradient_allreduce(old_model)
        new_model = apply_gradient_allreduce(new_model)

        criterion.to(rank)

    else:
        if multi_gpu:
            old_model = nn.DataParallel(old_model).to(rank)
            new_model = nn.DataParallel(new_model).to(rank)
        else:
            old_model = old_model.to(rank)
            new_model = new_model.to(rank)


    # Optimizer and Scheduler
    optimizer = load_optimizer(option, new_model.parameters())
    if scheduler is not None:
        scheduler = load_scheduler(option, optimizer)

    # Early Stopping
    early = EarlyStopping(patience=option.result['train']['patience'])


    # Dataset and DataLoader
    if task_id == 0:
        start = 0
        end = option.result['train']['num_init_segment']
    else:
        start = option.result['train']['num_init_segment'] + option.result['train']['num_segment'] * (task_id - 1)
        end = start + option.result['train']['num_segment']

    # Training Set
    tr_target_list = list(range(start, end))
    val_target_list = list(range(0, end))

    tr_dataset = load_data(option, data_type='train')
    tr_dataset = IncrementalSet(tr_dataset, start, target_list=tr_target_list, shuffle_label=True)

    if (task_id > 0) and (option.result['train']['num_exemplary'] > 0):
        if multi_gpu:
            tr_dataset.update_exemplar(new_model.module.exemplar_list)
        else:
            tr_dataset.update_exemplar(new_model.exemplar_list)

    # Validation Set
    val_dataset = load_data(option, data_type='val')
    val_dataset = IncrementalSet(val_dataset, start, target_list=val_target_list, shuffle_label=False)

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
    if option.result['train']['train_type'] == 'naive':
        from module.trainer import naive_trainer

        old_model.eval()
        for epoch in range(0, save_module.total_epoch):
            new_model.train()
            new_model, optimizer, save_module = naive_trainer.train(option, rank, epoch, task_id, new_model, old_model, \
                                                                    criterion, optimizer, tr_loader, scaler, save_module)

            new_model.eval()
            result = naive_trainer.validation(option, rank, epoch, task_id, new_model, old_model, criterion, val_loader)

            if scheduler is not None:
                scheduler.step()
                save_module.save_dict['scheduler'] = [scheduler.state_dict()]
            else:
                save_module.save_dict['scheduler'] = None

            # Early Stop
            if multi_gpu:
                param = deepcopy(new_model.module.state_dict())
            else:
                param = deepcopy(new_model.state_dict())

            early(result['val_loss'], param, result)
            if early.early_stop == True:
                break


    elif option.result['train']['train_type'] == 'icarl':
        from module.trainer import icarl_trainer

        old_model.eval()
        for epoch in range(0, save_module.total_epoch):
            new_model.train()
            new_model, optimizer, save_module = icarl_trainer.train(option, rank, epoch, task_id, new_model, old_model, \
                                                                    criterion, optimizer, tr_loader, scaler, save_module)

            new_model.eval()
            result = icarl_trainer.validation(option, rank, epoch, task_id, new_model, old_model, criterion, val_loader)

            if scheduler is not None:
                scheduler.step()
                save_module.save_dict['scheduler'] = [scheduler.state_dict()]
            else:
                save_module.save_dict['scheduler'] = None

            # Early Stop
            if multi_gpu:
                param = deepcopy(new_model.module.state_dict())
            else:
                param = deepcopy(new_model.state_dict())

            early(result['val_loss'], param, result)
            if early.early_stop == True:
                break

        # Save Exemplary
        if (option.result['train']['num_exemplary'] > 0) and ((rank == 0) or (rank == 'cuda')):
            m = int(option.result['train']['num_exemplary'] / new_class)

            if task_id > 0:
                if multi_gpu:
                    new_model.module.reduce_old_exemplar(m)
                else:
                    new_model.reduce_old_exemplar(m)

            for n in tr_target_list:
                n_data = tr_dataset.get_image_class(n)
                if multi_gpu:
                    new_model.module.get_new_exemplar(n_data, m, rank)
                else:
                    new_model.get_new_exemplar(n_data, m, rank)

            if multi_gpu:
                torch.save(new_model.module.exemplar_list, os.path.join(save_folder, 'task_%d_exemplar.pt' %(task_id)))
            else:
                torch.save(new_model.exemplar_list, os.path.join(save_folder, 'task_%d_exemplar.pt' %(task_id)))


    elif option.result['train']['train_type'] == 'wo_exemple':
        from module.trainer import wo_exemple_trainer

        old_model.eval()
        for epoch in range(0, save_module.total_epoch):
            new_model.train()
            new_model, optimizer, save_module = wo_exemple_trainer.train(option, rank, epoch, task_id, new_model, old_model, \
                                                                         criterion, optimizer, tr_loader, scaler, save_module)

            new_model.eval()
            result = wo_exemple_trainer.validation(option, rank, epoch, task_id, new_model, old_model, criterion, val_loader)

            if scheduler is not None:
                scheduler.step()
                save_module.save_dict['scheduler'] = [scheduler.state_dict()]
            else:
                save_module.save_dict['scheduler'] = None

            # Early Stop
            if multi_gpu:
                param = deepcopy(new_model.module.state_dict())
            else:
                param = deepcopy(new_model.state_dict())

            early(result['val_loss'], param, result)
            if early.early_stop == True:
                break

    else:
        raise('select proper train_type')


    # Saver
    if (rank == 'cuda') or (rank==0):
        # Load the best model
        best_param = early.model
        best_result = early.result
        save_module.save_dict['model'] = [best_param]

        # Save the best result
        val_acc1, val_acc5, val_loss = best_result['acc1'], best_result['acc5'], best_result['val_loss']

        if task_id == 0:
            mode = 'w'
        else:
            mode = 'a'

        with open(os.path.join(save_folder, 'result.txt'), mode) as f:
            f.write('task_%d - val_acc@1: %.2f, val_acc@5: %.2f, val_loss: %.3f \n' %(task_id, val_acc1, val_acc5, val_loss))

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
    if resume:
        resume_task_id = option.result['train']['resume_task_id']
    else:
        resume_task_id = 0

    resume_path = os.path.join(save_folder, 'task_%d_dict.pt' %(resume_task_id-1))
    config_path = os.path.join(save_folder, 'task_%d_config.json' %(resume_task_id-1))

    if resume:
        if (os.path.isfile(resume_path) == False) or (os.path.isfile(config_path) == False):
            resume = False
            resume_task_id = 0
        else:
            option = config(save_folder)
            option.import_config(config_path)

    # Logger
    neptune.init('sunghoshin/imp', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzdlYWFkMjctOWExMS00YTRlLWI0MWMtY2FhNmIyNzZlYTIyIn0=')
    exp_name, exp_num = save_folder.split('/')[-2], save_folder.split('/')[-1]
    neptune.create_experiment(params={'exp_name':exp_name, 'exp_num':exp_num},
                              tags=[])


    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False


    # RUN
    if option.result['train']['only_init']:
        resume_task_id = 0
        num_task = 1
    else:
        num_task = 1 + int((option.result['data']['num_class'] - option.result['train']['num_init_segment']) / option.result['train']['num_segment'])


    # Task
    for task_id in range(resume_task_id, num_task):
        if ddp:
            mp.spawn(main, args=(option, task_id, save_folder, ), nprocs=num_gpu, join=True)
        else:
            main('cuda', option, task_id, save_folder)

        # Logging the result
        with open(os.path.join(save_folder,'result.txt'), 'r') as f:
            result_list = f.readlines()

        result_line = result_list[-1]
        start =[pos for pos, char in enumerate(result_line) if char == ':']
        end =[pos for pos, char in enumerate(result_line) if char == ',']

        acc1 = float(result_line[start[0]+1 : end[0]].strip())
        acc5 = float(result_line[start[1]+1 : end[1]].strip())
        val_loss = float(result_line[start[2]+1 : -1].strip())

        neptune.log_metric('val_acc1', task_id, acc1)
        neptune.log_metric('val_acc5', task_id, acc5)
        neptune.log_metric('val_loss', task_id, val_loss)