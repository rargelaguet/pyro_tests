import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
assert pyro.__version__.startswith('1.3')

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
        self.sigmoid = nn.Sigmoid()

    # define the stochastic model p(x|z)p(z)
    def forward(self, x):
        """ x is a torch.Tensor of size (batch_size,784) """

        # Flatten x so that all the pixels are in the rightmost dimension.
        # - x before reshape has shape [batch_size, 1, 28, 28]
        # - x after reshape has shape [batch_size, 784]
        x = x.reshape(-1, 784)

        # sample W from prior q(w) = N(0,1)
        w_loc = torch.zeros([x.shape[1],self.nfactors])
        w_scale = torch.ones([x.shape[1],self.nfactors])

        # (Q) Why is .to_event(2) required? We want univariate distributions, not multivariate!
        # (ANSWER) Because we are using a multivariate normal as a guide!
        W = pyro.sample("W", dist.Normal(w_loc,w_scale).to_event(2))

        # plate to indicate independent samples
        with pyro.plate("samples", x.shape[0]):

            # sample Z from prior q(z) = N(0,1)
            z_loc = torch.zeros([self.nfactors])
            z_scale = torch.ones([self.nfactors])

            # (Q) Why is .to_event(1) required?
            # (ANSWER) Latent dimensions must be modelled with a Multivariate Normal
            Z = pyro.sample("Z", dist.Normal(z_loc, z_scale).to_event(1))
            
            # sample images using the Bernoulli distributions and score against the actual images
            # (Q) Why is .to_event(1) required?
            # (ANSWER) Batch_shape = [256] (samples), event_shape = 784 (features)
            ZW = self.sigmoid(torch.matmul(Z,W.T))
            X = pyro.sample("X", dist.Bernoulli(ZW).to_event(1), obs=x)

        return Z,W

    # define the guide: variational distributions q(z|x) for each unobserved variable
    def guide(self, x):
        """ x is a torch.Tensor of size (batch_size,784) """

        x = x.reshape(-1, 784)

        qw_loc = pyro.param("w_loc", torch.zeros([x.shape[1], self.nfactors]))
        qw_scale = pyro.param("w_scale", torch.ones([x.shape[1], self.nfactors]))
        pyro.sample("W", pyro.distributions.Normal(qw_loc,qw_scale))
        # pyro.sample("W", pyro.distributions.Normal(qw_loc,qw_scale).to_event(1))

        # plate to indicate independent samples
        with pyro.plate("samples", x.shape[0]):
            qz_loc = pyro.param("z_loc", torch.zeros([x.shape[0], self.nfactors]))
            qz_scale = pyro.param("z_scale", torch.ones([x.shape[0], self.nfactors]))
            # pyro.sample("Z", pyro.distributions.Normal(qz_loc,qz_scale))
            pyro.sample("Z", pyro.distributions.Normal(qz_loc,qz_scale).to_event(1))


    # def reconstruct_img(self, x):
    #     W,Z = self.forward(x)
    #     # decode the image (note we don't sample in image space)
    #     loc_img = self.decoder(z)
    #     return loc_img
