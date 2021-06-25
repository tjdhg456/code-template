import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.parameter import Parameter

def split_model(model):
    pass


class Identity_Layer(nn.Module):
    def __init__(self):
        super(Identity_Layer, self).__init__()

    def forward(self, x):
        return x

#### ICaRL Wrapper (Basis)
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
        grad_clone[:self.old_class] = 0.0
        return grad_clone

    def register_hook(self):
        self.hook = self.model_fc.weight.register_hook(self.my_hook)

    def remove_hook(self):
        self.hook.remove()


## Dynamically Expandable Representations
class super_feature_extractor(nn.Module):
    def __init__(self, option, task_id, old_class, new_class, device):
        super(super_feature_extractor, self).__init__()
        self.option = option
        self.task_id = task_id

    def concat(self, old):
        pass

class gate(nn.Module):
    def __init__(self, mask_size):
        super(gate, self).__init__()
        self.mask_param = Parameter(torch.ones([mask_size]).float(), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, s):
        out = self.sigmoid(self.mask_param * s)
        return out

    def get_scale(self, s):
        scale = (self.sigmoid(self.mask_param) * (1 - self.sigmoid(self.mask_param))) / ((s * self.sigmoid(s * self.mask_param) * (1 - self.sigmoid(s * self.mask_param))) + 1e-5)
        return scale




