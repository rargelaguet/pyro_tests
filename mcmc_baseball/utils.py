import torch
import math
import logging
import pandas as pd

from pyro.infer.mcmc.util import summary
from pyro.util import ignore_experimental_warning

from pyro.infer import MCMC, NUTS, Predictive

# ===================================
#        DATA SUMMARIZE UTILS
# ===================================


def get_summary_table(posterior, sites, player_names, transforms={}, diagnostics=False, group_by_chain=False):
    """
    Return summarized statistics for each of the ``sites`` in the
    traces corresponding to the approximate posterior.
    """
    site_stats = {}

    for site_name in sites:
        marginal_site = posterior[site_name]#.cpu()

        if site_name in transforms:
            marginal_site = transforms[site_name](marginal_site)

        site_summary = summary({site_name: marginal_site}, prob=0.5, group_by_chain=group_by_chain)[site_name]
        if site_summary["mean"].shape:
            site_df = pd.DataFrame(site_summary, index=player_names)
        else:
            site_df = pd.DataFrame(site_summary, index=[0])
        if not diagnostics:
            site_df = site_df.drop(["n_eff", "r_hat"], axis=1)
        site_stats[site_name] = site_df.astype(float).round(2)

    return site_stats


def train_test_split(pd_dataframe):
    """
    Training data - 45 initial at-bats and hits for each player.
    Validation data - Full season at-bats and hits for each player.
    """
    device = torch.Tensor().device
    train_data = torch.tensor(pd_dataframe[["At-Bats", "Hits"]].values, dtype=torch.float, device=device)
    test_data = torch.tensor(pd_dataframe[["SeasonAt-Bats", "SeasonHits"]].values, dtype=torch.float, device=device)
    first_name = pd_dataframe["FirstName"].values
    last_name = pd_dataframe["LastName"].values
    player_names = [" ".join([first, last]) for first, last in zip(first_name, last_name)]
    return train_data, test_data, player_names


# ===================================
#       MODEL EVALUATION UTILS
# ===================================


def sample_posterior_predictive(model, posterior_samples, baseball_dataset):
    """
    Generate samples from posterior predictive distribution.
    """
    train, test, player_names = train_test_split(baseball_dataset)
    at_bats = train[:,0]        # <class 'torch.Tensor'> torch.Size([18])
    at_bats_season = test[:,0]  # <class 'torch.Tensor'> torch.Size([18])
    logging.Formatter("%(message)s")
    logging.info("\nPosterior Predictive:")
    logging.info("Hit Rate - Initial 45 At Bats")
    logging.info("-----------------------------")
    # set hits=None to convert it from observation node to sample node
    train_predict = Predictive(model, posterior_samples)(at_bats, None)
    train_summary = get_summary_table(train_predict, sites=["obs"], player_names=player_names)["obs"]
    train_summary = train_summary.assign(ActualHits=baseball_dataset[["Hits"]].values)
    logging.info(train_summary)
    logging.info("\nHit Rate - Season Predictions")
    logging.info("-----------------------------")
    with ignore_experimental_warning():
        test_predict = Predictive(model, posterior_samples)(at_bats_season, None)
    test_summary = get_summary_table(test_predict,
                                     sites=["obs"],
                                     player_names=player_names)["obs"]
    test_summary = test_summary.assign(ActualHits=baseball_dataset[["SeasonHits"]].values)
    logging.info(test_summary)


def evaluate_pointwise_pred_density(model, posterior_samples, baseball_dataset):
    """
    Evaluate the log probability density of observing the unseen data (season hits)
    given a model and posterior distribution over the parameters.
    """
    _, test, player_names = train_test_split(baseball_dataset)
    at_bats_season, hits_season = test[:, 0], test[:, 1]
    trace = Predictive(model, posterior_samples).get_vectorized_trace(at_bats_season, hits_season)
    # Use LogSumExp trick to evaluate $log(1/num_samples \sum_i p(new_data | \theta^{i})) $,
    # where $\theta^{i}$ are parameter samples from the model's posterior.
    trace.compute_log_prob()
    post_loglik = trace.nodes["obs"]["log_prob"]
    # computes expected log predictive density at each data point
    exp_log_density = (post_loglik.logsumexp(0) - math.log(post_loglik.shape[0])).sum()
    logging.info("\nLog pointwise predictive density")
    logging.info("--------------------------------")
    logging.info("{:.4f}\n".format(exp_log_density))
