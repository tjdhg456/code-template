import numpy as np
import torch
from tqdm import tqdm
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import os
from torch.autograd import Variable
from copy import deepcopy
from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(option, rank, epoch, task_id, new_model, old_model, criterion, optimizer, tr_loader, scaler, save_module):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    for tr_data in tqdm(tr_loader):
        input, label = tr_data
        input, label = input.to(rank), label.to(rank)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = new_model(input)

                if task_id == 0:
                    loss = criterion(output, label)

                else:
                    with torch.no_grad():
                        output_old = old_model(input)
                        output_old = output_old.data
                    output_old = Variable(output_old).to(rank)
                    loss = criterion(output, label, output_old)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = new_model(input)

            if task_id == 0:
                loss = criterion(output, label)

            else:
                with torch.no_grad():
                    output_old = old_model(input)
                    output_old = output_old.data
                output_old = Variable(output_old).to(rank)
                loss = criterion(output, label, output_old)

            loss.backward()
            optimizer.step()

        acc_result = accuracy(output, label, topk=(1, 5))

        if (num_gpu > 1) and (option.result['train']['ddp']):
            mean_loss += reduce_tensor(loss.data, num_gpu).item()
            mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
            mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

        else:
            mean_loss += loss.item()
            mean_acc1 += acc_result[0]
            mean_acc5 += acc_result[1]

    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_acc5 /= len(tr_loader)
    mean_loss /= len(tr_loader)

    # Saving Network Params
    if multi_gpu:
        save_module.save_dict['model'] = [new_model.module.state_dict()]
    else:
        save_module.save_dict['model'] = [new_model.state_dict()]

    save_module.save_dict['optimizer'] = [optimizer.state_dict()]
    save_module.save_dict['save_epoch'] = epoch

    # Logging
    if (rank == 0) or (rank == 'cuda'):
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_ACC@5-%.2f, tr_loss:%.3f' %(epoch, option.result['train']['total_epoch']-1, \
                                                                            mean_acc1, mean_acc5, mean_loss))
    return new_model, optimizer, save_module


def validation(option, rank, epoch, task_id, new_model, old_model, criterion, val_loader):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    with torch.no_grad():
        for val_data in tqdm(val_loader):
            input, label = val_data
            input, label = input.to(rank), label.to(rank)

            output = new_model(input)
            acc_result = accuracy(output, label, topk=(1, 5))

            if (num_gpu > 1) and (option.result['train']['ddp']):
                mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
                mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

            else:
                mean_acc1 += acc_result[0]
                mean_acc5 += acc_result[1]

        # Train Result
        mean_acc1 /= len(val_loader)
        mean_acc5 /= len(val_loader)

        # Logging
        if (rank == 0) or (rank == 'cuda'):
            print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_ACC@5-%.2f, val_loss:%.3f' % (epoch, option.result['train']['total_epoch']-1, \
                                                                                    mean_acc1, mean_acc5, mean_loss))
    result = {'acc1':mean_acc1, 'acc5':mean_acc5, 'val_loss':mean_loss}
    return result


def test(option, rank, new_model, val_loader, task_id):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    if task_id > 0:
        if multi_gpu:
            new_model.module.update_center(rank)
        else:
            new_model.update_center(rank)

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    for img, label in val_loader:
        img, label = img.to(rank), label.to(rank)
        with torch.no_grad():
            if task_id == 0 or option.result['exemplar']['augment'] == 'adversarial':
                output = new_model(img)
            else:
                if multi_gpu:
                    output = new_model.module.icarl_classify(img)
                else:
                    output = new_model.icarl_classify(img)

        acc_result = accuracy(output.cpu(), label.cpu(), topk=(1, 5))
        mean_acc1 += acc_result[0]
        mean_acc5 += acc_result[1]

    # Train Result
    mean_acc1 /= len(val_loader)
    mean_acc5 /= len(val_loader)

    print('Final Result - val_ACC@1: %.2f, val_ACC@5-%.2f, val_loss:%.3f' % (mean_acc1, mean_acc5, mean_loss))
    result = {'acc1':mean_acc1, 'acc5':mean_acc5, 'val_loss':mean_loss}
    return result


def run(option, new_model, old_model, new_class, old_class, tr_loader, val_loader, tr_dataset, val_dataset, tr_target_list, val_target_list,
        optimizer, criterion, scaler, scheduler, early, early_stop, save_folder, save_module, multi_gpu, rank, task_id, ddp):
    old_model.eval()

    for epoch in range(0, save_module.total_epoch):
        # Training
        new_model.train()
        new_model, optimizer, save_module = train(option, rank, epoch, task_id, new_model, old_model, \
                                                                criterion, optimizer, tr_loader, scaler, save_module)

        # Validate
        new_model.eval()
        result = validation(option, rank, epoch, task_id, new_model, old_model, criterion, val_loader)

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

        if option.result['train']['early_criterion_loss']:
            early(result['val_loss'], param, result)
        else:
            early(-result['acc1'], param, result)

        if early.early_stop == True:
            break

    # After training
    if (option.result['exemplar']['num_exemplary'] > 0) and ((rank == 0) or (rank == 'cuda')):
        if early_stop:
            # Load Best Models
            del old_model, new_model

            new_model = load_model(option, new_class)
            new_model.load_state_dict(early.model)
            if (option.result['exemplar']['num_exemplary'] > 0) and (task_id > 0):
                new_model.exemplar_list = torch.load(os.path.join(save_folder, 'task_%d_exemplar.pt' % (task_id - 1)))
            else:
                new_model.exemplar_list = []

            # Multi-Processing GPUs
            if ddp:
                new_model.to(rank)
                new_model = DDP(new_model, device_ids=[rank])
            else:
                if multi_gpu:
                    new_model = nn.DataParallel(new_model).to(rank)
                else:
                    new_model = new_model.to(rank)

        # Save Exemplary Sets
        m = int(option.result['exemplar']['num_exemplary'] / new_class)
        print('Update Old Exemplary Set')
        if task_id > 0:
            if multi_gpu:
                new_model.module.reduce_old_exemplar(m)
            else:
                new_model.reduce_old_exemplar(m)

        print('Update New Exemplary Set')
        for n in tqdm(tr_target_list):
            n_data = tr_dataset.get_image_class(n)
            if multi_gpu:
                new_model.module.get_new_exemplar(n_data, m, rank)
            else:
                new_model.get_new_exemplar(n_data, m, rank)

        if multi_gpu:
            torch.save(new_model.module.exemplar_list, os.path.join(save_folder, 'task_%d_exemplar.pt' % (task_id)))
        else:
            torch.save(new_model.exemplar_list, os.path.join(save_folder, 'task_%d_exemplar.pt' % (task_id)))

        # Final Validation
        new_model.eval()
        result = test(option, rank, new_model, val_loader, task_id)
        early.result = result
        return early, save_module, option
















