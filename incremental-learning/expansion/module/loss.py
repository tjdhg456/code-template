import torch.nn as nn
import torch
import torch.nn.functional as F

class icarl_loss(nn.Module):
    def __init__(self, old_class, new_class):
        super(icarl_loss, self).__init__()
        self.old_class = old_class
        self.new_class = new_class

    def get_one_hot(self, target, num_class):
        one_hot = torch.zeros(target.shape[0], num_class).to(target.device)
        one_hot = one_hot.scatter(dim=1, index=target.long().view(-1, 1), value=1.)
        return one_hot

    def forward(self, output_new, target, output_old=None):
        target = self.get_one_hot(target, self.new_class)

        if output_old is not None:
            output_old = torch.sigmoid(output_old)
            target[..., :self.old_class] = output_old

        loss_cls = F.binary_cross_entropy_with_logits(output_new, target)
        return loss_cls