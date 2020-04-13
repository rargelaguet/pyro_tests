import torch
import pyro

from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform
from pyro.distributions.util import scalar_like

# ===================================
#               MODELS
# ===================================


def fully_pooled(at_bats, hits):
    r"""
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with a common probability of success, $\phi$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    phi_prior = Uniform(scalar_like(at_bats, 0), scalar_like(at_bats, 1))
    phi = pyro.sample("phi", phi_prior)
    num_players = at_bats.shape[0]
    with pyro.plate("num_players", num_players):
        return pyro.sample("obs", Binomial(at_bats, phi), obs=hits)


def not_pooled(at_bats, hits):
    r"""
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with independent probability of success, $\phi_i$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    with pyro.plate("num_players", num_players):
        phi_prior = Uniform(scalar_like(at_bats, 0), scalar_like(at_bats, 1))
        phi = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", Binomial(at_bats, phi), obs=hits)


def partially_pooled(at_bats, hits):
    r"""
    Number of hits has a Binomial distribution with independent
    probability of success, $\phi_i$. Each $\phi_i$ follows a Beta
    distribution with concentration parameters $c_1$ and $c_2$, where
    $c_1 = m * kappa$, $c_2 = (1 - m) * kappa$, $m ~ Uniform(0, 1)$,
    and $kappa ~ Pareto(1, 1.5)$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    m = pyro.sample("m", Uniform(scalar_like(at_bats, 0), scalar_like(at_bats, 1)))
    kappa = pyro.sample("kappa", Pareto(scalar_like(at_bats, 1), scalar_like(at_bats, 1.5)))
    with pyro.plate("num_players", num_players):
        phi_prior = Beta(m * kappa, (1 - m) * kappa)
        phi = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", Binomial(at_bats, phi), obs=hits)


def partially_pooled_with_logit(at_bats, hits):
    r"""
    Number of hits has a Binomial distribution with a logit link function.
    The logits $\alpha$ for each player is normally distributed with the
    mean and scale parameters sharing a common prior.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    loc = pyro.sample("loc", Normal(scalar_like(at_bats, -1), scalar_like(at_bats, 1)))
    scale = pyro.sample("scale", HalfCauchy(scale=scalar_like(at_bats, 1)))
    with pyro.plate("num_players", num_players):
        alpha = pyro.sample("alpha", Normal(loc, scale))
        return pyro.sample("obs", Binomial(at_bats, logits=alpha), obs=hits)

