import torch
import torch.nn as nn
import numpy as np


def split_model(model):
    pass


class Incremental_Wrapper(nn.Module):
    def __init__(self, option, model_cls):
        super(Incremental_Wrapper, self).__init__()
        self.option = option
        self.model_cls = model_cls

        self.exemplar_list = []

    def forward(self, image):
        out = self.model_cls(image)
        return out

    def get_new_exemplar(self, data, m, rank):
        # if self.option.result['train']['train_type'] == 'icarl':
        # Compute and cache features for each example
        features = []

        self.model_cls.eval()
        for img, label in data:
            x = img.to(rank)
            x.requires_grad = False
            with torch.no_grad():
                feature = self.model_cls.to(rank)(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)  # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)  # Normalize

        exemplar_set = []
        exemplar_features = []  # list of Variables of shape (feature_size,)
        for k in range(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0 / (k + 1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))

            exemplar_set.append(data[i])
            exemplar_features.append(features[i])

        self.exemplar_list.append(exemplar_set)


    def reduce_old_exemplar(self, m):
        for ix in range(len(self.exemplar_list)):
            self.exemplar_list[ix] = self.exemplar_list[ix][:m]

