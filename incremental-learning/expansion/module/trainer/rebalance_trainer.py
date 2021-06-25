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

new_features = []
old_features = []

def get_ref_features(self, inputs, outputs):
    global old_features
    old_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global new_features
    new_features = inputs[0]


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


def imprint():
    if iteration > start_iter and args.imprint_weights:
        # input: tg_model, X_train, map_Y_train
        # class_start = iteration*nb_cl class_end = (iteration+1)*nb_cl
        print("Imprint weights")
        #########################################
        # compute the average norm of old embdding
        old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
        average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
        #########################################
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features
        novel_embedding = torch.zeros((args.nb_cl, num_features))
        for cls_idx in range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl):
            cls_indices = np.array([i == cls_idx for i in map_Y_train])
            assert (len(np.where(cls_indices == 1)[0]) == dictionary_size)
            evalset.test_data = X_train[cls_indices].astype('uint8')
            evalset.test_labels = np.zeros(evalset.test_data.shape[0])  # zero labels
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                     shuffle=False, num_workers=2)
            num_samples = evalset.test_data.shape[0]
            cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
            # cls_features = cls_features.T
            # cls_features = cls_features / np.linalg.norm(cls_features,axis=0)
            # cls_embedding = np.mean(cls_features, axis=1)
            norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
            cls_embedding = torch.mean(norm_features, dim=0)
            # novel_embedding[cls_idx-iteration*args.nb_cl] = cls_embedding
            novel_embedding[cls_idx - iteration * args.nb_cl] = F.normalize(cls_embedding, p=2,
                                                                            dim=0) * average_old_embedding_norm
        tg_model.to(device)
        # torch.save(tg_model, "tg_model_before_imprint_weights.pth")
        tg_model.fc.fc2.weight.data = novel_embedding.to(device)
        # torch.save(tg_model, "tg_model_after_imprint_weights.pth")




    pass


def train(option, rank, epoch, task_id, new_model, old_model, old_class, criterion, optimizer, tr_loader, scaler, save_module):
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1

    # For Log
    mean_loss = 0.
    mean_acc1 = 0.
    mean_acc5 = 0.

    # Sigma Parameter
    if multi_gpu:
        sigma = new_model.module.model_fc.sigma
    else:
        sigma = new_model.model_fc.sigma


    # Lambda
    lambda_param = 5
    K = 2


    # Training
    for tr_data in tqdm(tr_loader):
        input, label = tr_data
        input, label = input.to(rank), label.to(rank)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = new_model(input)

                if task_id == 0:
                    loss = nn.CrossEntropyLoss()(output, label)
                else:
                    with torch.no_grad():
                        _ = old_model(input)

                    loss1 = nn.CosineEmbeddingLoss()(new_features, old_features.detach(), torch.ones(input.size(0).to(rank))) * lambda_param
                    loss2 = nn.CrossEntropyLoss()(output, label)

                    # Hard Negative
                    output_bs = output / sigma

                    gt_index = torch.zeros(output_bs.size()).to(rank)
                    gt_index = gt_index.scatter(1, label.view(-1, 1), 1).ge(0.5)
                    gt_scores = output_bs.masked_select(gt_index)

                    # get top-K scores on novel classes
                    max_novel_scores = output_bs[:, old_class:].topk(K, dim=1)[0]

                    # the index of hard samples, i.e., samples of old classes
                    hard_index = label.lt(old_class)
                    hard_num = torch.nonzero(hard_index).size(0)

                    # print("hard examples size: ", hard_num)
                    if hard_num > 0:
                        gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                        max_novel_scores = max_novel_scores[hard_index]
                        assert (gt_scores.size() == max_novel_scores.size())
                        assert (gt_scores.size(0) == hard_num)
                        loss3 = nn.MarginRankingLoss(margin=0.5)(gt_scores.view(-1, 1), \
                                                                  max_novel_scores.view(-1, 1),
                                                                  torch.ones(hard_num * K).to(rank))
                    else:
                        loss3 = torch.zeros(1).to(rank)

                    loss = loss1 + loss2 + loss3

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            output = new_model(input)

            if task_id == 0:
                loss = nn.CrossEntropyLoss()(output, label)

            else:
                with torch.no_grad():
                    _ = old_model(input)

                loss1 = nn.CosineEmbeddingLoss()(new_features, old_features.detach(),
                                                 torch.ones(input.size(0)).to(rank)) * lambda_param
                loss2 = nn.CrossEntropyLoss()(output, label)

                # Hard Negative
                output_bs = output / sigma

                gt_index = torch.zeros(output_bs.size()).to(rank)
                gt_index = gt_index.scatter(1, label.view(-1, 1), 1).ge(0.5)
                gt_scores = output_bs.masked_select(gt_index)

                # get top-K scores on novel classes
                max_novel_scores = output_bs[:, old_class:].topk(K, dim=1)[0]

                # the index of hard samples, i.e., samples of old classes
                hard_index = label.lt(old_class)
                hard_num = torch.nonzero(hard_index).size(0)

                # print("hard examples size: ", hard_num)
                if hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, K)
                    max_novel_scores = max_novel_scores[hard_index]
                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    loss3 = nn.MarginRankingLoss(margin=0.5)(gt_scores.view(-1, 1), \
                                                             max_novel_scores.view(-1, 1),
                                                             torch.ones(hard_num * K).to(rank))
                else:
                    loss3 = torch.zeros(1).to(rank)

                loss = loss1 + loss2 + loss3

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



def validation(option, rank, epoch, task_id, new_model, val_loader):
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
            val_loss = nn.CrossEntropyLoss()(output, label)
            acc_result = accuracy(output, label, topk=(1, 5))

            if (num_gpu > 1) and (option.result['train']['ddp']):
                mean_loss += reduce_tensor(val_loss.item(), num_gpu)
                mean_acc1 += reduce_tensor(acc_result[0], num_gpu)
                mean_acc5 += reduce_tensor(acc_result[1], num_gpu)

            else:
                mean_loss += val_loss.item()
                mean_acc1 += acc_result[0]
                mean_acc5 += acc_result[1]

        # Train Result
        mean_loss /= len(val_loader)
        mean_acc1 /= len(val_loader)
        mean_acc5 /= len(val_loader)

        # Logging
        if (rank == 0) or (rank == 'cuda'):
            print('Epoch-(%d/%d) - val_ACC@1: %.2f, val_ACC@5-%.2f, val_loss:%.3f' % (epoch, option.result['train']['total_epoch']-1, \
                                                                                    mean_acc1, mean_acc5, mean_loss))
    result = {'acc1':mean_acc1, 'acc5':mean_acc5, 'val_loss':mean_loss}
    return result


def run(option, new_model, old_model, new_class, old_class, tr_loader, val_loader, tr_dataset, val_dataset, tr_target_list, val_target_list,
        optimizer, criterion, scaler, scheduler, early, early_stop, save_folder, save_module, multi_gpu, rank, task_id, ddp):

    # Register Hook
    if multi_gpu:
        handle_ref_features = old_model.module.model_fc.register_forward_hook(get_ref_features)
        handle_cur_features = new_model.module.model_fc.register_forward_hook(get_cur_features)
    else:
        handle_ref_features = old_model.model_fc.register_forward_hook(get_ref_features)
        handle_cur_features = new_model.model_fc.register_forward_hook(get_cur_features)

    old_model.eval()
    for epoch in range(0, save_module.total_epoch):
        # Training
        new_model.train()
        new_model, optimizer, save_module = train(option, rank, epoch, task_id, new_model, old_model, old_class, \
                                                                criterion, optimizer, tr_loader, scaler, save_module)

        # Validate
        new_model.eval()
        result = validation(option, rank, epoch, task_id, new_model, val_loader)

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
        result = validation(option, rank, epoch, task_id, new_model, val_loader)
        early.result = result

    # Remove Hook
    if task_id > 0:
        handle_cur_features.remove()
        handle_ref_features.remove()

    return early, save_module, option
















