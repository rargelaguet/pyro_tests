"""
Sample shape: 
- iid (example: Montecarlo draws for the n-th sample and the k-th parameter)
Batch shape
- independent, but not identical (different samples)
Event shape:
- shape of the event space (multivariate or univariate distributions)
"""

# (Q) Difference between [] and [1] in shapes??

import torch
from pyro.distributions import *


# sampling two MC samples for a single individual using a univariate normal distribution
# - sample_shape: [2]
# - batch_shape: []
# - event_shape: []
dist = Normal(0.0, 1.0)
sample_shape = torch.Size([])
dist.sample(sample_shape) # tensor([ 0.2786, -1.4113])
(sample_shape, dist.batch_shape, dist.event_shape) # (torch.Size([2]), torch.Size([]), torch.Size([]))


# sampling one MC sample for two individuals using a univariate gaussian
# - sample_shape: []
# - batch_shape: [2]
# - event_shape: []
dist = Normal(torch.zeros(2), torch.ones(2))
sample_shape = torch.Size([])
dist.sample(sample_shape) # tensor([0.0101, 0.6976])
(sample_shape, dist.batch_shape, dist.event_shape) # (torch.Size([]), torch.Size([2]), torch.Size([]))


# sampling three MC sample for five individuals using a multivariate gaussian
# - sample_shape: [2]
# - batch_shape: [2]
# - event_shape: [2]
dist = MultivariateNormal(torch.zeros(2, 2), torch.eye(2))
sample_shape = torch.Size([2])
(sample_shape, dist.batch_shape, dist.event_shape) # (torch.Size([2]), torch.Size([2]), torch.Size([2]))


# sampling three MC sample for five individuals using a multivariate gaussian
# - sample_shape: [3]
# - batch_shape: [5]
# - event_shape: [2]
dist = MultivariateNormal(torch.zeros(2), torch.eye(2)).expand(torch.Size([5]))
sample_shape = torch.Size([3])
dist.sample(sample_shape)
(sample_shape, dist.batch_shape, dist.event_shape) # (torch.Size([3]), torch.Size([5]), torch.Size([2]))


