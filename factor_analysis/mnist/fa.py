"""
"""

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
assert pyro.__version__.startswith('1.3.0')

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

# seed
pyro.set_rng_seed(42)


######################
## Define the model ##
######################
	
class FA(nn.Module):
    def __init__(self, nfactors = 50):
        super().__init__()

        self.nfactors = nfactors

    # define the stochastic model p(x|z)p(z)
    def model(self, x):
        """ x is a torch.Tensor of size (batch_size,784) """

        # Flatten x so that all the pixels are in the rightmost dimension.
        # - x before reshape has shape [batch_size, 1, 28, 28]
        # - x after reshape has shape [batch_size, 784]
        x = x.reshape(-1, 784)

        # sample W from prior q(w) = N(0,1)
        w_loc = torch.zeros([x.shape[1],self.nfactors])
        w_scale = torch.ones([x.shape[1],self.nfactors])
        W = pyro.sample("W", dist.Normal(w_loc,w_scale))
        # W = pyro.sample("W", dist.Normal(w_loc,w_scale).to_event(1))

        # plate to indicate independent samples
        with pyro.plate("samples", x.shape[0]):

            # sample Z from prior q(z) = N(0,1)
            z_loc = torch.zeros([x.shape[0], self.nfactors])
            z_scale = torch.ones([x.shape[0], self.nfactors])
            # Z = pyro.sample("Z", dist.Normal(z_loc, z_scale))
            Z = pyro.sample("Z", dist.Normal(z_loc, z_scale).to_event(1))

            # sample images using the Bernoulli distributions and score against the actual images
            # pyro.sample("X", dist.Bernoulli(torch.matmul(Z,W.T)), obs=x)
            pyro.sample("X", dist.Bernoulli(torch.matmul(Z,W.T)).to_event(1), obs=x)


    # # define the guide: variational distributions q(z|x) for each unobserved variable
    # def guide(self, x):
    #     """ x is a torch.Tensor of size (batch_size,784) """

    #     x = x.reshape(-1, 784)

    #     qw_loc = pyro.param("w_loc", torch.zeros([x.shape[1], self.nfactors]))
    #     qw_scale = pyro.param("w_scale", torch.ones([x.shape[1], self.nfactors]))
    #     pyro.sample("W", pyro.distributions.Normal(qw_loc,qw_scale))
    #     # pyro.sample("W", pyro.distributions.Normal(qw_loc,qw_scale).to_event(1))

    #     # plate to indicate independent samples
    #     with pyro.plate("samples", x.shape[0]):
    #         qz_loc = pyro.param("z_loc", torch.zeros([x.shape[0], self.nfactors]))
    #         qz_scale = pyro.param("z_scale", torch.ones([x.shape[0], self.nfactors]))
    #         # pyro.sample("Z", pyro.distributions.Normal(qz_loc,qz_scale))
    #         pyro.sample("Z", pyro.distributions.Normal(qz_loc,qz_scale).to_event(1))


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
