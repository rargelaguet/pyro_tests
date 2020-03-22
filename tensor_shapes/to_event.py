import torch
import pyro
from pyro.distributions import *

"""
.to_event(reinterpreted_batch_ndims=None)
Reinterprets the n rightmost dimensions of this distributions batch_shape as event dims, adding them to the left side of event_shape.

return a reshaped version of this distribution, of type pyro.distributions.torch.Independent

In Pyro you can treat a univariate distribution as multivariate by calling the .to_event(n) 
property where n is the number of batch dimensions (from the right) to declare as dependent.

Use my_dist.to_event(1) to declare a dimension as dependent
"""


###############
## Example 1 ##
###############

# You go from a normal distribution with 10 independent samples to a multivariate distribution of dimensionality D
# (Q) How is the covariance matrix defined ??
d = Normal(0,1).expand([10]).to_event(1) 
x = pyro.sample("x",d)  # 
print([d.batch_shape, d.event_shape]) # [torch.Size([]), torch.Size([10])]
print(x.shape) # torch.Size([10]), samples have shape (batch_shape + event_shape)

# Equivalent code using plate
with pyro.plate("x_plate", 10):
	d = dist.Normal(0,1)
    x = pyro.sample("x",d)  # .expand([10]) is automatic

# The difference between these two versions is that the second version with plate informs Pyro 
# that it can make use of conditional independence information when estimating gradients


###############
## Example 2 ##
###############

d1 = MultivariateNormal(torch.zeros(2,3), torch.eye(3))
print([d1.batch_shape, d1.event_shape]) # [torch.Size([2]), torch.Size([3])]

d2 = d1.to_event(1)
print([d2.batch_shape, d2.event_shape]) # [torch.Size([]), torch.Size([2,3])]





