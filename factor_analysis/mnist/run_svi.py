"""
"""

import torch
import pyro
import numpy as np

from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoNormal, AutoDiagonalNormal

from setup_data import *
# from plot_utils import *
from fa import *

assert pyro.__version__.startswith('1.3')
pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

# seed
pyro.set_rng_seed(0)

def sigmoid(X): return np.divide(1.,1.+np.exp(-X))

###########################################
## Functions to train and test the model ##
###########################################

# Train
def train(svi, train_loader):
    epoch_loss = 0.
    for x, _ in train_loader:
        epoch_loss += svi.step(x)

    # step() returns a noisy estimate of the loss. This estimate is not normalized, and it scales with the size of the mini-batch
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train


# Test
def evaluate(svi, test_loader):
    test_loss = 0.
    for x, _ in test_loader:
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


#####################
## Hyperparameters ##
#####################

# Run options
LEARNING_RATE = 1.0e-3

# Run only for a single iteration for testing
NUM_EPOCHS = 2
TEST_FREQUENCY = 1


##################
## Do inference ##
##################

# Load full unnormalised data
# X_train, y_train = torch.load('/Users/ricard/test/pyro/files/MNIST/processed/training.pt')
# X_test, y_test = torch.load('/Users/ricard/test/pyro/files/MNIST/processed/test.pt')


# Load data using torch.utils.data.DataLoader
train_loader, test_loader = setup_data_loaders(batch_size=512, subset=True)

# clear param store
pyro.clear_param_store()

# setup the Factor Analysis model
fa = FA()

# setup the inference algorithm
guide = AutoNormal(fa)
# guide = fa.guide
optim = Adam({"lr": LEARNING_RATE})
svi = SVI(fa.forward, guide = guide, optim = optim, loss = Trace_ELBO())


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

guide.requires_grad_(False)


#########################
## Plot training curve ##
#########################

# file = "/Users/ricard/test/pyro/factor_analysis/mnist/outdir/elbo.png"
# plot_elbo(np.array(train_elbo), np.array(test_elbo), file)

######################
## Reconstruct data ##
######################

# Y = test_loader.dataset.data.float().reshape(-1, 784)



# Ypred_1 = fa.forward(X_train)

# (Q) Do we get an expectation of the posterior or the sample from the last batch?
# (ANSWER) I THINK JUST THE SAMPLE FROM THE LAST BATCH
param_dic = pyro.get_param_store()


# Calculate expectation of posteriors

# W = param_dic.get_param("AutoNormal.locs.W").numpy()
# W = dist.Normal(
#     param_dic.get_param("AutoNormal.locs.W"), 
#     param_dic.get_param("AutoNormal.scales.W")
# ).sample((1000,)).mean(axis=0).numpy()

# Z = param_dic.get_param("AutoNormal.locs.Z").numpy()
Z = dist.Normal(
    param_dic.get_param("AutoNormal.locs.Z"), 
    param_dic.get_param("AutoNormal.scales.Z")
).sample((1000,)).mean(axis=0).numpy()

# Ypred_2 = sigmoid(Z.dot(W.T))


# np.unique(Y)
# np.unique(Ypred)
# np.unique(X)
# Calculate MSE

###############################
## Plot reconstructed digits ##
###############################

###############################
## t-SNE on the latent space ##
###############################


X_train, y_train, X_test, y_test = setup_data_loaders(batch_size="full")

# (Q) How to get the expectation of the posterior for the full data set?
Z,_ = fa.forward(X_test)


# import pdb; pdb.set_trace()

outdir = "/Users/ricard/test/pyro/factor_analysis/mnist/outdir"
plot_tsne(Z, y_test, outdir)

######################
## Store parameters ##
######################

# param_dic = pyro.get_param_store()
# param_dic.get_all_param_names() # dict_keys(['AutoNormal.locs.W', 'AutoNormal.scales.W', 'AutoNormal.locs.Z', 'AutoNormal.scales.Z'])

# param_dic.save()

