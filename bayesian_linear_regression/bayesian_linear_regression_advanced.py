"""
https://pyro.ai/examples/bayesian_regression_ii.html
We would like to explore the relationship between topographic heterogeneity of a nation as measured by the Terrain Ruggedness Index 
(variable rugged in the dataset) and its GDP per capita.

three features from the dataset: 
	- rugged: quantifies the Terrain Ruggedness Index 
	- cont_africa: whether the given nation is in Africa 
	- rgdppc_2000: Real GDP per capita for the year 2000

We will write out the model again, similar to that in Part I, but explicitly without the use of PyroModule
"""

import os
import torch
import pyro
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import pyro.distributions as dist
import torch.distributions.constraints as constraints

pyro.enable_validation(True)

#####################
## Hyperparameters ##
#####################

num_iterations = 10

learning_rate = 0.001

###############
## Load data ##
###############

DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")

df = data[["cont_africa", "rugged", "rgdppc_2000"]]
df = df[np.isfinite(df.rgdppc_2000)]
df["rgdppc_2000"] = np.log(df["rgdppc_2000"])

# Prepare training data
train = torch.tensor(df.values, dtype=torch.float)


###########
## Model ##
###########

def model(x_1, x_2, y):

    # bias
    a = pyro.sample("a", dist.Normal(0., 10.))

    # weights
    b_x1 = pyro.sample("b1", dist.Normal(0., 1.))
    b_x2 = pyro.sample("b2", dist.Normal(0., 1.))
    b_x12 = pyro.sample("b12", dist.Normal(0., 1.))

    # sigma
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    # prediction
    mean = a + b_x1*x_1 + b_x2*x_2 + b_x12*x_1*x_2

    # plate for the number of samples
    with pyro.plate("data", len(x_2)):
        pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

###########
## Guide ##
###########

# def guide(x_1, x_2, log_gdp):
def guide(x_1, x_2, y):
    # Define parameters, with dimensionalities and constraint
    a_loc = pyro.param('a_loc', torch.tensor(0.))
    a_scale = pyro.param('a_scale', torch.tensor(1.), constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', torch.tensor(1.), constraint=constraints.positive)
    weights_loc = pyro.param('weights_loc', torch.randn(3))
    weights_scale = pyro.param('weights_scale', torch.ones(3), constraint=constraints.positive)

    # Define sampling distribution for each parameter
    a = pyro.sample("a", dist.Normal(a_loc, a_scale))
    b_x1 = pyro.sample("b1", dist.Normal(weights_loc[0], weights_scale[0]))
    b_x2 = pyro.sample("b2", dist.Normal(weights_loc[1], weights_scale[1]))
    b_x12 = pyro.sample("b12", dist.Normal(weights_loc[2], weights_scale[2]))
    sigma = pyro.sample("sigma", dist.Normal(sigma_loc, torch.tensor(0.05)))

    # Why ???
    mean = a + b_x1*x_1 + b_x2*x_2 + b_x12*x_1*x_2

#######################
## Training settings ##
#######################

from pyro.infer import SVI, Trace_ELBO

# Define optimiser
optim = pyro.optim.Adam({"lr": learning_rate})

# Stochastic Variational Inference (loss is defined as the ELBO)
svi = SVI(model, guide, optim, loss=Trace_ELBO())


#############
## Iterate ##
#############

# (Q) ???
pyro.clear_param_store()

is_cont_africa, ruggedness, log_gdp = train[:, 0], train[:, 1], train[:, 2]

# Elbo loss: 5795.467590510845
# Elbo loss: 415.8169444799423
# Elbo loss: 250.71916329860687
# Elbo loss: 247.19457268714905
# Elbo loss: 249.2004036307335
# Elbo loss: 250.96484470367432
# Elbo loss: 249.35092514753342
# Elbo loss: 248.7831552028656
# Elbo loss: 248.62140649557114
# Elbo loss: 250.4274433851242

for j in range(num_iterations):
    elbo = svi.step(is_cont_africa, ruggedness, log_gdp)
    print("[iteration %04d] ELBO: %.4f" % (j + 1, elbo))
        
