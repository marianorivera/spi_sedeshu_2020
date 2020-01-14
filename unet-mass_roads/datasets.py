import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Mass_Roads(Dataset):
    def __init__(self, transform=transforms.ToTensor(), train=True):
        if train:
            path = '../data/'
        else:
            path = '../data/'

        self.inputs = torchvision.datasets.ImageFolder(
            root=path + 'input', transform=transform)

        self.targets = torchvision.datasets.ImageFolder(
            root=path + 'target', transform=transform)

    def __len__(self):
        return min(len(self.inputs),
                   len(self.targets))

    def __getitem__(self, idx):
        inpt, _ = self.inputs[idx]
        trgt, _ = self.targets[idx]

        return inpt, trgt
