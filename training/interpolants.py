"""
Interpolants are defined by a path from the noise distribution to the data:

    x_t = alpha(t) * x_0 + beta(t) * x_1,    t in [t_min, t_max]

with the corresponding velocity

    v_t = alpha_dot(t) * x_0 + beta_dot(t) * x_1.
"""

import numpy as np
import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Linear:  x_t = (1 - t) * x_0 + t * x_1,  t in [0, 1].

@persistence.persistent_class
class LinearInterpolant:
    def __init__(self):
        self.t_min = 0.0
        self.t_max = 1.0

    def alpha(self, t):     return 1.0 - t
    def beta(self, t):      return t
    def alpha_dot(self, t): return -torch.ones_like(t)
    def beta_dot(self, t):  return torch.ones_like(t)

#----------------------------------------------------------------------------
# Trigonometric:  x_t = cos(t) * x_0 + sin(t) * x_1,  t in [0, pi/2].

@persistence.persistent_class
class TrigInterpolant:
    def __init__(self):
        self.t_min = 0.0
        self.t_max = float(np.pi / 2)

    def alpha(self, t):     return torch.cos(t)
    def beta(self, t):      return torch.sin(t)
    def alpha_dot(self, t): return -torch.sin(t)
    def beta_dot(self, t):  return torch.cos(t)

#----------------------------------------------------------------------------
