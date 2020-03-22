"""
Pyro models can use the context manager pyro.plate to declare that certain batch dimensions are independent. 
- An example of an independent dimension is the index over data in a minibatch: each datum should be independent of all others.

# Note that we always count from the right by using negative indices like -2, -1.
with pyro.plate("x_axis", 320):
    # within this context, batch dimension -1 is independent
    with pyro.plate("y_axis", 200):
        # within this context, batch dimensions -2 and -1 are independent
"""

import torch
import pyro
from pyro.distributions import *


# a: batch_shape = (), event_shape = ()
a = pyro.sample("a", Normal(0,1))

# b (2,) resulting from batch_shape = (), event_shape = (2)
b = pyro.sample("b", Normal(torch.zeros(2),1).to_event(1))

# c: (2,) resulting from batch_shape = (2), event_shape = ()
with pyro.plate("c_plate", 2):
    # (Q) if batch_size is already (2), why specify torch.zeros(2) ???
    # c = pyro.sample("c", Normal(torch.zeros(2), 1))
    c = pyro.sample("c", Normal(0, 1))

# d: (3,4,5) resulting from batch_shape = (3), event_shape = (4,5)
# (Q) How do you interpret a normal distribution with event_shape (4,5) ???
with pyro.plate("d_plate", 3):
    d = pyro.sample("d", Normal(torch.zeros(3,4,5), 1).to_event(2))


# x_axis = pyro.plate("x_axis", 3, dim=-2)
# y_axis = pyro.plate("y_axis", 2, dim=-3)
# with x_axis:
#     x = pyro.sample("x", Normal(0, 1))
# with y_axis:
#     y = pyro.sample("y", Normal(0, 1))
# with x_axis, y_axis:
#     xy = pyro.sample("xy", Normal(0, 1))
#     z = pyro.sample("z", Normal(0, 1).expand([5]).to_event(1))
# assert x.shape == (3, 1)        # batch_shape == (3,1)     event_shape == ()
# assert y.shape == (2, 1, 1)     # batch_shape == (2,1,1)   event_shape == ()
# assert xy.shape == (2, 3, 1)    # batch_shape == (2,3,1)   event_shape == ()
# assert z.shape == (2, 3, 1, 5)  # batch_shape == (2,3,1)   event_shape == (5,)
