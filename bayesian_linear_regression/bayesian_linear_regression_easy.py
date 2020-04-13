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
import torch
import pyro
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
from pyro.nn import PyroSample

pyro.enable_validation(True)

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

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
# african_nations = df[df["cont_africa"] == 1]
# non_african_nations = df[df["cont_africa"] == 0]
# sns.scatterplot(non_african_nations["rugged"], non_african_nations["rgdppc_2000"], ax=ax[0])
# ax[0].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="Non African Nations")
# sns.scatterplot(african_nations["rugged"], african_nations["rgdppc_2000"], ax=ax[1])
# ax[1].set(xlabel="Terrain Ruggedness Index", ylabel="log GDP (2000)", title="African Nations");

###################################################
## Standard (non-probabilistic) regression model ##
###################################################

# PyroModule is very similar to PyTorch's nn.Module, but additionally supports Pyro primitives as attributes.
# We will create a trivial class called PyroModule[nn.Linear] that subclasses PyroModule and torch.nn.Linear.
assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

# (Q) why expand ??
# - Use .expand() to draw a batch of samples

# (Q) .to_event() ??
# - Use my_dist.to_event(1) to declare a dimension as dependent.

class BayesianRegression(PyroModule):
    def __init__(self, nfeatures):
        super().__init__()

        # PyroModule[nn.linear] has two main attributes: weight and bias
        self.linear = PyroModule[nn.Linear](nfeatures, 1) 

        # weight: N(0,1), 
        # .expand() yields batch_shape = [1,3], event_shape = []
        # ??? to._event()
        # foo = dist.Normal(0.,1.).expand([1, nfeatures]) 
        foo = dist.Normal(0.,1.).expand([1, nfeatures]).to_event(2) # Independent ???
        self.linear.weight = PyroSample(foo)

        # bias: N(0,10)
        # For the bias component, we set a reasonably wide prior since it is likely to be substantially above 0.
        # .expand() yields batch_shape = [1,], event_shape = []
        self.linear.bias = PyroSample(dist.Normal(0.,10.).expand([1]).to_event(1))

    def forward(self, x, y=None):
        # x has shape torch.Size([Nsamples,Nfeatures])
        # y has shape torch.Size([Nsamples])

        # Sample noise
        # (Q) Is sigma also infered? It has no prior??
        # (Q) Difference between PyroSample and pyro.sample ??
        # (Q) Why no need to expand or .to_event()
        sigma = pyro.sample("sigma", dist.Uniform(0.,10.))

        # (Q) Why do "self.linear.weight" and "self.linear.bias" not intervene here???
        # - Because the .linear() module samples the weight and bias parameters from the prior and returns a value for the mean response.
        mean = self.linear(x).squeeze(-1) # shape: torch.Size([Nsamples])

        # create plate for the number of samples
        with pyro.plate("data", x.shape[0]):
            # we use the obs= argument to the pyro.sample statement to condition on the observed data. 
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

        # The model returns the regression line given by the variable mean.
        return mean


#######################
## Training settings ##
#######################

from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO

# Initialise model
model = BayesianRegression(nfeatures=3)

# Define guide: mean-field
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

# (Q) ???
pyro.clear_param_store()

for j in range(num_iterations):
    loss = svi.step(x_data, y_data)
    print("[iteration %04d] ELBO: %.4f" % (j + 1, loss / len(data)))
        


#####################################
## Inspect posterior distributions ##
#####################################

print("Learned parameters:")

# (Q) ???
guide.requires_grad_(False)

# Autoguide packs the latent variables into a single tensor, in this case, one entry per variable sampled in our model
# Both the loc and scale parameters have size (5,), one for each of the latent variables in the model (bias=1, weight=3, sigma=1)
# - AutoDiagonalNormal.loc: tensor([-2.2371, -1.8097, -0.1691,  0.3791,  9.1823])
# - AutoDiagonalNormal.scale: tensor([0.0551, 0.1142, 0.0387, 0.0769, 0.0702])
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name))

