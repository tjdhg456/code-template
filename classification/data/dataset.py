from torchvision.transforms import transforms
import torchvision

def load_cifar():
    transform = transforms.Compose(
        [transforms.Resize((128,128)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    tr_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return tr_dataset, val_dataset

def load_data(option, data_type='train'):
    tr_d, val_d = load_cifar()
    if data_type == 'train':
        return tr_d
    else:
        return val_d
