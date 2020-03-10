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

assert issubclass(PyroModule[nn.Linear], nn.Linear)
assert issubclass(PyroModule[nn.Linear], PyroModule)

# Regression model
linear_reg_model = PyroModule[nn.Linear](3, 1)

#######################
## Training settings ##
#######################

# Create Torch tensor
data = torch.tensor(df[["cont_africa", "rugged", "cont_africa_x_rugged", "rgdppc_2000"]].values, dtype=torch.float)
x_data, y_data = data[:, :-1], data[:, -1]

# Define loss
loss_fn = torch.nn.MSELoss(reduction='sum')

# Define optimiser
optim = torch.optim.Adam(linear_reg_model.parameters(), lr=learning_rate)


def train():
    optim.zero_grad()                               # initialize gradients to zero
    y_pred = linear_reg_model(x_data).squeeze(-1)   # run the model forward on the data
    loss = loss_fn(y_pred, y_data)                  # calculate the mse loss
    loss.backward()                                 # backpropagate
    optim.step()                                    # update parameters
    return loss

#############
## Iterate ##
#############

for j in range(num_iterations):
    loss = train()
    print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))


# Inspect learned parameters
print("Learned parameters:")
for name, param in linear_reg_model.named_parameters():
    print(name, param.data.numpy())

# Suggested values:
# [iteration 1500] loss: 147.8815
# Learned parameters: weight [[-1.9478593  -0.20278624  0.39330274]], bias [9.22308]