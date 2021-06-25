from module.resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import torch.nn as nn
import torch
from module.network import gate

model_enc = ResNet18()

## 1. Register Hook for Check Mask Size
class mask_hook(object):
    def __init__(self):
        self.mask_size = []
        self.handler = []
        self.ix = 0
        self.s = 1

    def mask_size_hook(self, _, input, output):
        self.mask_size.append(output.size()[1])

    def mask_forward_hook(self, _, input, output):
        _, C, _, _ = output.size()
        getattr(self, 'mask_%d' % self.ix).register_backward_hook(self.mask_backward_hook(self.ix))
        output = output * getattr(self, 'mask_%d' %self.ix).to(output.device)(self.s).reshape(1, C, 1, 1)
        self.ix += 1
        return output

    def mask_backward_hook(self, index):
        def backward_hook(_, grad_input, grad_output):
            device = grad_input[0].device
            scale = getattr(self, 'mask_%d' %index).get_scale(self.s).to(device)
            grad_input = (grad_input[0] * scale, )
            return grad_input
        return backward_hook

    def register_hook(self, model, check_size=True):
        if check_size:
            hook_fn = self.mask_size_hook
        else:
            hook_fn = self.mask_forward_hook

        for name, layer in model._modules.items():
            if isinstance(layer, nn.Sequential):
                self.register_hook(layer, check_size)
            else:
                if len(list(layer._modules)) == 0:
                    if 'conv' in name:
                        handle = layer.register_forward_hook(hook_fn)
                        self.handler.append(handle)
                else:
                    for name2, param in layer._modules.items():
                        if 'conv' in name2:
                            handle = layer.register_forward_hook(hook_fn)
                            self.handler.append(handle)

    def remove_hook(self):
        for handler in self.handler:
            handler.remove()

        self.handler = []
        self.ix = 0

    def update_mask_param(self):
        assert len(self.mask_size) > 0

        for ix, m_size in enumerate(self.mask_size):
            setattr(self, 'mask_%d' %ix, gate(m_size))


# Check Mask Size
hook = mask_hook()
hook.register_hook(model_enc, check_size=True)

x = torch.ones([1,3,128,128])
model_enc(x)

print(hook.mask_size)
hook.remove_hook()

# Update Mask Parameters
hook.update_mask_param()

# Forward with Mask
hook.register_hook(model_enc, check_size=False)

x = torch.ones([1,3,128,128])

out = model_enc(x)
torch.sum(out).backward()

hook.remove_hook()





