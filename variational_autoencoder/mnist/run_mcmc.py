"""
http://pyro.ai/examples/vae.html
"""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

import pyro
import pyro.infer.mcmc

import argparse

from vae import *
# from setup_data import *

pyro.enable_validation(True)
pyro.distributions.enable_validation(False)

# seed
pyro.set_rng_seed(0)

# clear param store
pyro.clear_param_store()


def main(args):

    # Load data
    root = '/Users/ricard/test/pyro/files'
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    # X_train = train_set.data.to(dtype=torch.float64)
    X_train = train_set.data.to(dtype=torch.float64).float()

    # setup the VAE
    vae = VAE()

    # Load NUTS algorithm
    nuts_kernel = pyro.infer.mcmc.NUTS(vae.model, adapt_step_size=True)

    # Run MCMC
    mcmc = pyro.infer.mcmc.MCMC(nuts_kernel, 
        num_samples = args.num_samples, 
        warmup_steps = args.warmup_steps, 
        num_chains = args.num_chains
    ).run(x=X_train)



    # Summary
    # mcmc.summary(prob=0.5)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Eight Schools MCMC')
    parser.add_argument('--num-samples', type=int, default=1000, help='number of MCMC samples (default: 1000)')
    parser.add_argument('--num-chains', type=int, default=1, help='number of parallel MCMC chains (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='number of MCMC samples for warmup (default: 1000)')
    args = parser.parse_args()

    main(args)


