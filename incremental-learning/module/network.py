import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def split_model(model):
    pass


class Identity_Layer(nn.Module):
    def __init__(self):
        super(Identity_Layer, self).__init__()

    def forward(self, x):
        return x


class Incremental_Wrapper(nn.Module):
    def __init__(self, option, model_enc, model_fc, old_class, new_class, device):
        super(Incremental_Wrapper, self).__init__()
        self.option = option
        self.model_enc = model_enc
        self.model_fc = model_fc

        self.exemplar_list = []
        self.exemplar_aug_list = []

        self.new_class = new_class
        self.old_class = old_class

        self.device = device

    def forward(self, image):
        x = self.model_enc(image)
        out = self.model_fc(x)
        return out

    def update_datasize(self, data_size):
        self.data_size = data_size


    def feature_extractor(self, image):
        out1 = self.model_enc(image)
        return out1

    def exemple_loader(self, data_list):
        class exemple_dataset(Dataset):
            def __init__(self, data_list):
                self.data_list = data_list

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, index):
                data, label = self.data_list[index]
                return data, label

        ex_ds = exemple_dataset(data_list)
        ex_loader = DataLoader(ex_ds, shuffle=False, batch_size=self.option.result['train']['batch_size'])
        return ex_loader


    def get_new_exemplar(self, data, m, rank):
        features = []
        logits = []
        labels = []

        ex_loader_imp = self.exemple_loader(data)

        self.model_enc.eval()

        # Calculate the centers
        if self.option.result['exemplar']['sampling_type'] == 'herding':
            for img, label in ex_loader_imp:
                x = img.to(rank)
                with torch.no_grad():
                    feature_imp = self.model_enc(x)
                    logit =  self.model_fc(feature_imp).detach()
                    feature = F.normalize(feature_imp).detach().cpu().numpy()
                features.append(feature)
                logits.append(logit.cpu().numpy())
                labels.append(label.cpu().numpy())

            features = np.concatenate(features, axis=0)
            logits = np.concatenate(logits, axis=0)
            labels = np.concatenate(labels, axis=0)
            class_mean = np.mean(features, axis=0)

            # Find the optimal exemplar set
            exemplar_set = []
            feature_dim = features.shape[1]
            now_class_mean = np.zeros((1, feature_dim))

            for i in range(m):
                # shape：batch_size*512
                x = class_mean - (now_class_mean + features) / (i + 1)
                # shape：batch_size
                x = np.linalg.norm(x, axis=1)
                index = np.argmin(x)
                now_class_mean += features[index]

                if self.option.result['exemplar']['exemplar_type'] == 'data':
                    exemplar_set.append(data[index])
                elif self.option.result['exemplar']['exemplar_type'] == 'logit':
                    exemplar_set.append((torch.tensor(logits[index]).float(), torch.tensor([labels[index]]).long().item()))
                else:
                    raise('Select Proper exemplar type')

        else:
            raise('Select Proper Exemplar Sampling Type')

        self.exemplar_list.append(exemplar_set)

    def get_aug_exemplar(self):
        self.exemplar_aug_list = self.exemplar_list

    def update_center(self, rank):
        self.exemplar_means = []

        self.model_enc.eval()

        for index in range(len(self.exemplar_aug_list)):
            print("compute the class mean of %s"%(str(index)))
            exemplar = self.exemplar_aug_list[index]

            # Calculate the centers
            features = []
            ex_loader_imp = self.exemple_loader(exemplar)
            for img, label in ex_loader_imp:
                x = img.to(rank)
                with torch.no_grad():
                    feature = F.normalize(self.model_enc(x).detach()).cpu().numpy()
                features.append(feature)

            features = np.concatenate(features, axis=0)
            class_mean = np.mean(features, axis=0)
            class_mean= class_mean / np.linalg.norm(class_mean)
            self.exemplar_means.append(class_mean)

    def icarl_classify(self, x):
        result = []

        self.model_enc.eval()

        x_out = F.normalize(self.model_enc(x).detach()).cpu().numpy()
        exemplar_means = np.array(self.exemplar_means)
        for target in x_out:
            x = target - exemplar_means
            x = np.linalg.norm(x, ord=2, axis=1)
            result.append(-x)
        return torch.tensor(result)


    def reduce_old_exemplar(self, m):
        if self.option.result['exemplar']['exemplar_type'] == 'data':
            for ix in range(len(self.exemplar_list)):
                self.exemplar_list[ix] = self.exemplar_list[ix][:m]
        else:
            raise('Select Proper Exemplar Type')


    def my_hook(self, grad):
        grad_clone = grad.clone()
        grad_clone[:self.old_class] = 0
        return grad_clone

    def register_hook(self, n_old_class):
        self.old_class = n_old_class
        for param in self.model_fc.parameters():
            param.register_hook(self.my_hook)


## Augmentation
class Augment_Wrapper(Incremental_Wrapper):
    def __init__(self, option, model_enc, model_fc, old_class, new_class, device, transform):
        super(Augment_Wrapper, self).__init__(option, model_enc, model_fc, old_class, new_class, device)
        self.option = option
        self.transform = transform

    def zero_grad(self):
        self.model_enc.zero_grad()
        self.model_fc.zero_grad()

    def eval(self):
        self.model_enc.eval()
        self.model_fc.eval()

    def augment_adversarial(self):
        if self.option.result['exemplar']['exemplar_type'] == 'logit' and self.option.result['exemplar']['num_exemplary'] > 0:
            C, H, W = self.data_size
            criterion = nn.MSELoss()
            iters = 200

            self.eval()

            exemplar_total_list = []
            print('Generate Adversarial Samples')
            for exemplar_ix in tqdm(self.exemplar_list):
                exemplar_list = []
                for exemplar in exemplar_ix:
                    logit, label = exemplar
                    logit, label = torch.unsqueeze(logit, dim=0), torch.tensor([label]).long()
                    logit, label = logit.float().to(self.device), label.to(self.device)

                    # Perturbation
                    perturbation = torch.empty([1, C, H, W]).uniform_(0, 1)
                    perturbation = perturbation.to(self.device)

                    for _ in range(iters):
                        perturbation.requires_grad = True
                        current = self.transform.normalize(perturbation)

                        out_perturb = self.forward(current)

                        self.zero_grad()
                        cost = criterion(out_perturb[:, :self.old_class], logit[:, :self.old_class])
                        cost.backward()

                        diff = -0.01 * perturbation.grad.sign()
                        perturbation = perturbation + diff
                        perturbation = torch.clamp(perturbation, min=0, max=1).detach_()

                    perturbation = torch.squeeze(self.transform.normalize(perturbation))

                    # Save the output perturbation for each class
                    exemplar_list.append((perturbation.cpu(), label.cpu().item()))

                # Save the output perturbation for all classes
                exemplar_total_list.append(exemplar_list)
            return exemplar_total_list

        elif self.option.result['exemplar']['num_exemplary'] == 0:
            pass

        else:
            raise('Select Proper Exemplar Type')


    def get_aug_exemplar(self):
        aug_option = self.option.result['exemplar']['augment']
        if aug_option is None:
            self.exemplar_aug_list = self.exemplar_list
        elif aug_option == 'adversarial':
            self.exemplar_aug_list = self.augment_adversarial()


    def reduce_old_exemplar(self, m):
        if self.option.result['exemplar']['exemplar_type'] == 'data':
            for ix in range(len(self.exemplar_list)):
                self.exemplar_list[ix] = self.exemplar_list[ix][:m]

        elif self.option.result['exemplar']['exemplar_type'] == 'logit':
            self.eval()
            for ix in range(len(self.exemplar_aug_list)):
                logit_list = []
                label_list = []
                ex_loader = self.exemple_loader(self.exemplar_aug_list[ix][:m])
                for img, label in ex_loader:
                    img = img.to(self.device)

                    with torch.no_grad():
                        logit = self.forward(img)

                    logit = logit.detach().cpu()
                    logit_list.append(logit)
                    label_list.append(label.cpu())

                logit_list = torch.cat(logit_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                self.exemplar_list[ix] = [(logit_imp, label_imp.item()) for logit_imp, label_imp in zip(logit_list, label_list)]
        else:
            raise('Select Proper Exemplar Type')
