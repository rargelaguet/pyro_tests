"""
"""

####################
## Load libraries ##
####################

import pandas as pd

import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms
import logging

import pyro
import pyro.distributions as dist
from pyro.infer import *
from pyro.infer.mcmc.util import summary
from pyro.optim import Adam

assert pyro.__version__.startswith('1.3')

from fa import *

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

# seed
pyro.set_rng_seed(0)

#####################
## Hyperparameters ##
#####################

NUM_CHAINS = 1
WARMUP_STEPS = 10
NUM_SAMPLES = 10

###############
## Load data ##
###############

# for loading and batching MNIST dataset
root = '/Users/ricard/test/pyro/files'
trans = transforms.ToTensor()
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
# test_set = dset.MNIST(root=root, train=False, transform=trans)

# train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

X_train = train_set.data.to(dtype=torch.float64).float()

# test
X_train = X_train[:100,:,:] 

##################
## Do inference ##
##################

# clear param store
pyro.clear_param_store()

# setup the Factor Analysis model
fa = FA(nfactors=10)

# Load NUTS algorithm (Hamiltonian Monte Carlo)
nuts_kernel = pyro.infer.mcmc.NUTS(fa.forward, adapt_step_size=True)

# Run MCMC
model = pyro.infer.mcmc.MCMC(kernel = nuts_kernel, num_samples = NUM_SAMPLES, warmup_steps = WARMUP_STEPS, num_chains = NUM_CHAINS)
model.run(x=X_train)


###################################
## Query posterior distributions ##
###################################

# Get posterior samples for the variable "W" and take the average
W = model.get_samples(50)['W'].mean(0)

# Get posterior samples for all variables
posterior_samples = model.get_samples(group_by_chain=False, num_samples=100)

# Get summary statistics for the variable "W"
# ['mean', 'std', 'median', '25.0%', '75.0%', 'n_eff', 'r_hat']
# - diagnostics: n_eff, r_hat
variable_summary = summary({"W": posterior_samples["W"]}, prob=0.5, group_by_chain=False)["W"]

##############################
## Predictive distributions ##
##############################

# (NOT WORKING....) Calculate predictive distributions
# samples = model.get_samples()
# foo = Predictive(model, posterior_samples=samples)
# foo(X_train)
# foo.forward(X_train)

####################################
## Data reconstruction statistics ##
####################################
