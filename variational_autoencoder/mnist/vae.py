"""
http://pyro.ai/examples/vae.html
"""

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
assert pyro.__version__.startswith('1.3.0')

# pyro.enable_validation(True)
# pyro.distributions.enable_validation(False)

# seed
# pyro.set_rng_seed(0)

################################################################
## Define the Neural network architecture (not probabilistic) ##
################################################################

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim) # Input layer
        self.fc21 = nn.Linear(hidden_dim, 784)  # Output layer 
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    # define the forward computation on the latent z
    def forward(self, z):

        # (1) compute the hidden units
        # hidden has shape [batch_size, hidden_dim]
        hidden = self.softplus(self.fc1(z))

        # (2) return the parameter for the output Bernoulli
        # loc_img has shape [batch_size, 784]. Each pixel has a Bernoulli distribution
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)    # input layer
        self.fc21 = nn.Linear(hidden_dim, z_dim) # hidden layer 
        self.fc22 = nn.Linear(hidden_dim, z_dim) # output layer  
        self.softplus = nn.Softplus()            # setup the non-linearities

    # define the forward computation on an image
    def forward(self, x):

        # x before reshape has shape [batch_size, 1, 28, 28]
        # x after reshape has shape [batch_size, 784]
        x = x.reshape(-1, 784)

        # (1) compute the hidden units
        # hidden has shape [batch_size, hidden_dim]
        hidden = self.softplus(self.fc1(x))

        # (2) return a mean vector and a (positive) square root variance
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))  # (Q) Why EXP??

        return z_loc, z_scale


####################
## Define the VAE ##
####################
	
class VAE(nn.Module):
    def __init__(self, z_dim = 50, hidden_dim = 400):
        super().__init__()

        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        self.z_dim = z_dim

    # define the stochastic model p(x|z)p(z)
    def model(self, x):
        """ x is a torch.Tensor of size (batch_size,784) """

        # Flatten x so that all the pixels are in the rightmost dimension.
        # - x before reshape has shape [batch_size, 1, 28, 28]
        # - x after reshape has shape [batch_size, 784]
        x = x.reshape(-1, 784)

        # register PyTorch module "decoder" with Pyro
        pyro.module("decoder", self.decoder)

        # plate to indicate independent samples
        with pyro.plate("data", x.shape[0]):

            # setup hyperparameters for prior p(z)
            # - z_loc and z_scale have dimensions (batch_size, z_dim)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))

            # sample z from prior q(z) = N(0,1)
            # - z has dimensions (batch_size, z_dim)
            # (Q) Why .to_event(1)???
            #   .to_event(1) ensures that instead of treating our sample as being generated from a univariate normal 
            #   with batch_size = z_dim, we treat them as being generated from a multivariate normal distribution with diagonal covariance.
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            # decode the latent z
            # - loc_img has dimensions (batch_size, 784)
            # - loc_img contains rates of Bernoulli distributions
            loc_img = self.decoder.forward(z)

            # sample images using the Bernoulli distributions and score against the actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x)

    # define the guide: variational distributions q(z|x) for each unobserved variable
    def guide(self, x):
        """ x is a torch.Tensor of size (batch_size,784) """

        # register PyTorch module "encoder" with Pyro
        pyro.module("encoder", self.encoder)

        # plate to indicate independent samples
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc,z_scale).to_event(1))

    # define a helper function for reconstructing images
    # def reconstruct_img(self, x):
    #     # encode image x
    #     z_loc, z_scale = self.encoder(x)
    #     # sample in latent space
    #     z = dist.Normal(z_loc, z_scale).sample()
    #     # decode the image (note we don't sample in image space)
    #     loc_img = self.decoder(z)
    #     return loc_img

