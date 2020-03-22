"""
http://pyro.ai/examples/vae.html
"""

import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
assert pyro.__version__.startswith('1.3.0')

from vae import *

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

# seed
pyro.set_rng_seed(0)

#####################
## Hyperparameters ##
#####################

# Run options
LEARNING_RATE = 1.0e-3

# Run only for a single iteration for testing
NUM_EPOCHS = 5
TEST_FREQUENCY = 1

###############
## Load data ##
###############

# for loading and batching MNIST dataset
def setup_data_loaders(batch_size = 128):
    root = '/Users/ricard/test/pyro/files'
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

##################
## Do inference ##
##################

# Load data
train_loader, test_loader = setup_data_loaders(batch_size=256)

# clear param store
pyro.clear_param_store()

# setup the VAE
vae = VAE()

# setup the inference algorithm
svi = SVI(vae.model, vae.guide, optim = Adam({"lr": LEARNING_RATE}), loss = Trace_ELBO())


# training loop
train_elbo = []
test_elbo = []
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, train_loader)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % TEST_FREQUENCY == 0:
        total_epoch_loss_test = evaluate(svi, test_loader)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))