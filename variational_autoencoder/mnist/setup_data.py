import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

# for loading and batching MNIST dataset
def setup_data_loaders(batch_size = 128):
    root = '/Users/ricard/test/pyro/files'
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader