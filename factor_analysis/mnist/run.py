"""
"""

import torch

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
assert pyro.__version__.startswith('1.3.0')

from setup_data import *
from fa import *

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

# seed
pyro.set_rng_seed(0)

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
NUM_EPOCHS = 5
TEST_FREQUENCY = 1


##################
## Do inference ##
##################

# Load data
train_loader, test_loader = setup_data_loaders(batch_size=256)

# clear param store
pyro.clear_param_store()

# setup the Factor Analysis model
fa = FA()

# setup the inference algorithm
svi = SVI(fa.model, fa.guide, optim = Adam({"lr": LEARNING_RATE}), loss = Trace_ELBO())


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