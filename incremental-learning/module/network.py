import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def split_model(model):
    pass


class Identity_Layer(nn.Module):
    def __init__(self):
        super(Identity_Layer, self).__init__()

    def forward(self, x):
        return x


class Incremental_Wrapper(nn.Module):
    def __init__(self, option, model_enc, model_fc):
        super(Incremental_Wrapper, self).__init__()
        self.option = option
        self.model_enc = model_enc
        self.model_fc = model_fc

        self.exemplar_list = []

    def forward(self, image):
        out1 = self.model_enc(image)
        out2 = self.model_fc(out1)
        return out2


class Icarl_Wrapper(Incremental_Wrapper):
    def __init__(self, option, model_enc, model_fc):
        super(Icarl_Wrapper, self).__init__(option, model_enc, model_fc)
        self.option = option
        self.model_enc = model_enc
        self.model_fc = model_fc

        self.exemplar_list = []

    def forward(self, image):
        x = self.model_enc(image)
        out = self.model_fc(x)
        return out

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

        ex_loader_imp = self.exemple_loader(data)

        self.model_enc.eval()

        # Calculate the centers
        for img, label in ex_loader_imp:
            x = img.to(rank)
            with torch.no_grad():
                feature = F.normalize(self.model_enc(x).detach()).cpu().numpy()
            features.append(feature)

        features = np.concatenate(features, axis=0)
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
            exemplar_set.append(data[index])

        self.exemplar_list.append(exemplar_set)


    def update_center(self, rank):
        self.exemplar_means = []

        self.model_enc.eval()

        for index in range(len(self.exemplar_list)):
            print("compute the class mean of %s"%(str(index)))
            exemplar = self.exemplar_list[index]

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
        for ix in range(len(self.exemplar_list)):
            self.exemplar_list[ix] = self.exemplar_list[ix][:m]

    def my_hook(self, grad):
        grad_clone = grad.clone()
        grad_clone[:self.old_class] = 0
        return grad_clone

    def register_hook(self, n_old_class):
        self.old_class = n_old_class
        for param in self.model_fc.parameters():
            param.register_hook(self.my_hook)

