import torch
from torch_utils import persistence

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingLoss:
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def __call__(self, net, images, labels=None):
        t = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        x0 = torch.randn_like(images)
        xt = (1 - t) * x0 + t * images

        v_target = images - x0
        v_pred = net(xt, t, labels)

        loss = (v_pred - v_target) ** 2
        return loss

#----------------------------------------------------------------------------