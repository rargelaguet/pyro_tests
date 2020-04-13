import argparse
import logging

import pandas as pd
import torch

import pyro
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model

from models import *
from utils import *

assert pyro.__version__.startswith('1.3.1')

"""
It demonstrates how to do Bayesian inference using NUTS (or, HMC) in Pyro, and use of some common inference utilities.

As in the Stan tutorial, this uses the small baseball dataset to estimate players' batting average

The dataset separates the initial 45 at-bats statistics from the remaining season.
We use the hits data from the initial 45 at-bats to estimate the batting average
for each player. We then use the remaining season's data to validate the predictions from our models.

Three models are evaluated:
 - Complete pooling model: The success probability of scoring a hit is shared
     amongst all players.
 - No pooling model: Each individual player's success probability is distinct and
     there is no data sharing amongst players.
 - Partial pooling model: A hierarchical model with partial data sharing.


We recommend Radford Neal's tutorial on HMC ([3]) to users who would like to get a
more comprehensive understanding of HMC and its variants, and to [4] for details on
the No U-Turn Sampler, which provides an efficient and automated way (i.e. limited
hyper-parameters) of running HMC on different problems.

[1] Carpenter B. (2016), ["Hierarchical Partial Pooling for Repeated Binary Trials"]
    (http://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html).
[2] Efron B., Morris C. (1975), "Data analysis using Stein's estimator and its
    generalizations", J. Amer. Statist. Assoc., 70, 311-319.
[3] Neal, R. (2012), "MCMC using Hamiltonian Dynamics",
    (https://arxiv.org/pdf/1206.1901.pdf)
[4] Hoffman, M. D. and Gelman, A. (2014), "The No-U-turn sampler: Adaptively setting
    path lengths in Hamiltonian Monte Carlo", (https://arxiv.org/abs/1111.4246)
"""

logging.basicConfig(format='%(message)s', level=logging.INFO)
DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/EfronMorrisBB.txt"


def main(args):
    baseball_dataset = pd.read_csv(DATA_URL, "\t")
    train, _, player_names = train_test_split(baseball_dataset)
    at_bats, hits = train[:, 0], train[:, 1]
    logging.info("Original Dataset:")
    logging.info(baseball_dataset)

    # (1) Full Pooling Model
    # In this model, we illustrate how to use MCMC with general potential_fn.
    init_params, potential_fn, transforms, _ = initialize_model(
        fully_pooled, model_args=(at_bats, hits), num_chains=args.num_chains,
        jit_compile=args.jit, skip_jit_warnings=True)
    nuts_kernel = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains,
                initial_params=init_params,
                transforms=transforms)
    mcmc.run(at_bats, hits)
    samples_fully_pooled = mcmc.get_samples()
    logging.info("\nModel: Fully Pooled")
    logging.info("===================")
    logging.info("\nphi:")
    logging.info(get_summary_table(mcmc.get_samples(group_by_chain=True),
                                   sites=["phi"],
                                   player_names=player_names,
                                   diagnostics=True,
                                   group_by_chain=True)["phi"])
    num_divergences = sum(map(len, mcmc.diagnostics()["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(fully_pooled, samples_fully_pooled, baseball_dataset)
    evaluate_pointwise_pred_density(fully_pooled, samples_fully_pooled, baseball_dataset)

    # (2) No Pooling Model
    nuts_kernel = NUTS(not_pooled, jit_compile=args.jit, ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains)
    mcmc.run(at_bats, hits)
    samples_not_pooled = mcmc.get_samples()
    logging.info("\nModel: Not Pooled")
    logging.info("=================")
    logging.info("\nphi:")
    logging.info(get_summary_table(mcmc.get_samples(group_by_chain=True),
                                   sites=["phi"],
                                   player_names=player_names,
                                   diagnostics=True,
                                   group_by_chain=True)["phi"])
    num_divergences = sum(map(len, mcmc.diagnostics()["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(not_pooled, samples_not_pooled, baseball_dataset)
    evaluate_pointwise_pred_density(not_pooled, samples_not_pooled, baseball_dataset)

    # (3) Partially Pooled Model
    nuts_kernel = NUTS(partially_pooled, jit_compile=args.jit, ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains)
    mcmc.run(at_bats, hits)
    samples_partially_pooled = mcmc.get_samples()
    logging.info("\nModel: Partially Pooled")
    logging.info("=======================")
    logging.info("\nphi:")
    logging.info(get_summary_table(mcmc.get_samples(group_by_chain=True),
                                   sites=["phi"],
                                   player_names=player_names,
                                   diagnostics=True,
                                   group_by_chain=True)["phi"])
    num_divergences = sum(map(len, mcmc.diagnostics()["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(partially_pooled, samples_partially_pooled, baseball_dataset)
    evaluate_pointwise_pred_density(partially_pooled, samples_partially_pooled, baseball_dataset)

    # (4) Partially Pooled with Logit Model
    nuts_kernel = NUTS(partially_pooled_with_logit, jit_compile=args.jit, ignore_jit_warnings=True)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains)
    mcmc.run(at_bats, hits)
    samples_partially_pooled_logit = mcmc.get_samples()
    logging.info("\nModel: Partially Pooled with Logit")
    logging.info("==================================")
    logging.info("\nSigmoid(alpha):")
    logging.info(get_summary_table(mcmc.get_samples(group_by_chain=True),
                                   sites=["alpha"],
                                   player_names=player_names,
                                   transforms={"alpha": torch.sigmoid},
                                   diagnostics=True,
                                   group_by_chain=True)["alpha"])
    num_divergences = sum(map(len, mcmc.diagnostics()["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(partially_pooled_with_logit, samples_partially_pooled_logit, baseball_dataset)
    evaluate_pointwise_pred_density(partially_pooled_with_logit, samples_partially_pooled_logit, baseball_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseball batting average using HMC")
    parser.add_argument("-n", "--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument("--warmup-steps", nargs='?', default=100, type=int)
    parser.add_argument("--rng_seed", nargs='?', default=0, type=int)
    parser.add_argument("--jit", action="store_true", default=False, help="use PyTorch jit")
    args = parser.parse_args()

    pyro.set_rng_seed(args.rng_seed)
    # Enable validation checks
    pyro.enable_validation(__debug__)

    # work around with the error "RuntimeError: received 0 items of ancdata"
    # see https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823
    torch.multiprocessing.set_sharing_strategy("file_system")

    main(args)