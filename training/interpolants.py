"""Flow interpolants used by FlowMatchingModel / FlowMatchingLoss.

An interpolant defines a path between data and noise:

    x_t = data_coef(t) * x_data + noise_coef(t) * x_noise
    v_t = data_coef_dot(t) * x_data + noise_coef_dot(t) * x_noise

It also owns three things that are entirely about the time axis:

    - the time range [t_min, t_max] and on which side the data lives
      (`data_side in {'min', 'max'}`); the noise lives on the other side.
    - the training-time distribution of t, configured via `t_dist_kwargs`
      (uniform by default).
    - the integration grid used at sampling time, returned by
      `sample_steps(n_steps, device)`. The grid is ordered from
      `t_noise` to `t_data`, so the sampler always integrates "noise -> data"
      regardless of the data side.

`t` is always a tensor; coefficient functions are element-wise.
"""

import numpy as np
import torch
from torch_utils import persistence
import dnnlib

#----------------------------------------------------------------------------
# Training-time distributions over t. Each samples t in [t_min, t_max].

@persistence.persistent_class
class UniformDist:
    """t ~ U[t_min, t_max]."""
    def sample(self, batch_size, device, t_min, t_max):
        return torch.rand(batch_size, device=device) * (t_max - t_min) + t_min


@persistence.persistent_class
class LogitNormalDist:
    """t = t_min + (t_max - t_min) * sigmoid(loc + scale * eps), eps ~ N(0, 1).
    Default loc=0, scale=1 matches the schedule used in Stable Diffusion 3."""
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = float(loc)
        self.scale = float(scale)

    def sample(self, batch_size, device, t_min, t_max):
        z = torch.randn(batch_size, device=device) * self.scale + self.loc
        return torch.sigmoid(z) * (t_max - t_min) + t_min

#----------------------------------------------------------------------------
# Interpolant base class.

@persistence.persistent_class
class Interpolant:
    """Subclasses set t_min/t_max/data_side and override the four coefficients."""

    def __init__(self, t_min, t_max, data_side, t_dist_kwargs=None):
        assert data_side in ('min', 'max')
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.data_side = data_side
        if t_dist_kwargs is None:
            t_dist_kwargs = dict(class_name='training.interpolants.UniformDist')
        self.t_dist = dnnlib.util.construct_class_by_name(**t_dist_kwargs)

    @property
    def t_noise(self):
        return self.t_max if self.data_side == 'min' else self.t_min

    @property
    def t_data(self):
        return self.t_min if self.data_side == 'min' else self.t_max

    def sample_t(self, batch_size, device):
        return self.t_dist.sample(batch_size, device, self.t_min, self.t_max)

    def sample_steps(self, n_steps, device):
        # Integration grid: n_steps + 1 points from t_noise to t_data.
        return torch.linspace(self.t_noise, self.t_data, n_steps + 1, device=device)

    def data_coef(self, t):      raise NotImplementedError
    def noise_coef(self, t):     raise NotImplementedError
    def data_coef_dot(self, t):  raise NotImplementedError
    def noise_coef_dot(self, t): raise NotImplementedError

#----------------------------------------------------------------------------
# Linear:  x_t = (1 - t) * x_noise + t * x_data,  t in [0, 1]. Data at t_max.

@persistence.persistent_class
class LinearInterpolant(Interpolant):
    def __init__(self, t_dist_kwargs=None):
        super().__init__(t_min=0.0, t_max=1.0, data_side='max', t_dist_kwargs=t_dist_kwargs)

    def data_coef(self, t):      return t
    def noise_coef(self, t):     return 1.0 - t
    def data_coef_dot(self, t):  return torch.ones_like(t)
    def noise_coef_dot(self, t): return -torch.ones_like(t)

#----------------------------------------------------------------------------
# Trigonometric:  x_t = cos(t) * x_data + sin(t) * x_noise,  t in [0, pi/2].
# Data at t_min.

@persistence.persistent_class
class TrigInterpolant(Interpolant):
    def __init__(self, t_dist_kwargs=None):
        super().__init__(t_min=0.0, t_max=float(np.pi / 2), data_side='min', t_dist_kwargs=t_dist_kwargs)

    def data_coef(self, t):      return torch.cos(t)
    def noise_coef(self, t):     return torch.sin(t)
    def data_coef_dot(self, t):  return -torch.sin(t)
    def noise_coef_dot(self, t): return torch.cos(t)

#----------------------------------------------------------------------------
