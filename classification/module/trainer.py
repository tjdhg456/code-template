import numpy as np
import torch
from tqdm import tqdm

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

def train(option, rank, epoch, model, criterion, optimizer, tr_loader, scaler, save_module, neptune):
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1

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
                output = model(input)
                loss = criterion(output, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = model(input)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

        acc_result = accuracy(output, label, topk=(1, 5))
        mean_acc1 += acc_result[0]
        mean_acc5 += acc_result[1]
        mean_loss += loss.item()

    # Train Result
    mean_acc1 /= len(tr_loader)
    mean_acc5 /= len(tr_loader)
    mean_loss /= len(tr_loader)

    # Saving Network Params
    if multi_gpu:
        save_module.save_dict['model'] = [model.module.state_dict()]
    else:
        save_module.save_dict['model'] = [model.state_dict()]

    save_module.save_dict['optimizer'] = [optimizer.state_dict()]
    save_module.save_dict['save_epoch'] = epoch

    # Logging
    if (rank == 0) or (rank == 'cuda'):
        print('Epoch-(%d/%d) - tr_ACC@1: %.2f, tr_ACC@5-%.2f, tr_loss:%.3f' %(epoch, option.result['train']['total_epoch'], \
                                                                            mean_acc1, mean_acc5, mean_loss))
        neptune.log_metric('tr_loss', mean_loss)
        neptune.log_metric('tr_acc1', mean_acc1)
        neptune.log_metric('tr_acc5', mean_acc5)

    return model, optimizer, save_module


def validation(option, rank, epoch, model, criterion, val_loader, neptune):
    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    with torch.no_grad():
        for val_data in tqdm(val_loader):
            input, label = val_data
            input, label = input.to(rank), label.to(rank)

            output = model(input)
            loss = criterion(output, label)

            acc_result = accuracy(output, label, topk=(1, 5))

            mean_acc1 += acc_result[0]
            mean_acc5 += acc_result[1]

            mean_loss += loss.item()

        # Train Result
        mean_acc1 /= len(val_loader)
        mean_acc5 /= len(val_loader)
        mean_loss /= len(val_loader)

        # Logging
        if (rank == 0) or (rank == 'cuda'):
            print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_ACC@5-%.2f, val_loss:%.3f' % (epoch, option.result['train']['total_epoch'], \
                                                                                    mean_acc1, mean_acc5, mean_loss))
            neptune.log_metric('val_loss', mean_loss)
            neptune.log_metric('val_acc1', mean_acc1)
            neptune.log_metric('val_acc5', mean_acc5)

    result = {'acc1':mean_acc1, 'acc5':mean_acc5, 'val_loss':mean_loss}
    return result

def test():
    pass