"""
https://pyro.ai/examples/bayesian_regression.html
We would like to explore the relationship between topographic heterogeneity of a nation as measured by the Terrain Ruggedness Index 
(variable rugged in the dataset) and its GDP per capita.

three features from the dataset: 
	- rugged: quantifies the Terrain Ruggedness Index 
	- cont_africa: whether the given nation is in Africa 
	- rgdppc_2000: Real GDP per capita for the year 2000
"""

import os
# from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from pyro.nn import PyroModule

import seaborn as sns
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

# for CI testing
# smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.3.0')
# pyro.enable_validation(True)
# pyro.set_rng_seed(1)
# pyro.enable_validation(True)

#####################
## Hyperparameters ##
#####################

num_iterations = 1000

learning_rate = 0.01

###############
## Load data ##
###############

DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

# Add a feature to capture the interaction between "cont_africa" and "rugged"
df["cont_africa_x_rugged"] = df["cont_africa"] * df["rugged"]

###############
## Plot data ##
###############

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
african_nations = df[df["cont_africa"] == 1]
non_african_nations = df[df["cont_africa"] == 0]
sns.scatterplot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], ax=ax[0])
ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")
sns.scatterplot(african_nations["rugged"], african_nations["rgdppc_2000"], ax=ax[1])
ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations");

###################################################
## Standard (non-probabilistic) regression model ##
###################################################

from pyro.nn import PyroSample


class BayesianRegression(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        mean = self.linear(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


#######################
## Training settings ##
#######################


from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO

# Initialise model
model = BayesianRegression(3, 1)

# We will use the AutoDiagonalNormal guide that models the distribution of unobserved parameters in the model as a 
# Gaussian with diagonal covariance, i.e. it assumes that there is no correlation amongst the latent variables.
guide = AutoDiagonalNormal(model)

# Define optimiser
optim = pyro.optim.Adam({"lr": learning_rate})

# Stochastic Variational Inference (loss is defined as the ELBO)
svi = SVI(model, guide, optim, loss=Trace_ELBO())


#############
## Iterate ##
#############

# Create Torch tensor
data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values, dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]

pyro.clear_param_store()
for j in range(num_iterations):
    loss = svi.step(x_data, y_data)
    print("[iteration %04d] ELBO: %.4f" % (j + 1, loss / len(data)))
        

# Inspect learned parameters
print("Learned parameters:")
guide.requires_grad_(False)
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))



#####################################
## Inspect posterior distributions ##
#####################################

# instead of just point estimates, we now have uncertainty estimates (AutoDiagonalNormal.scale) for our learned parameters