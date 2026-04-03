import torch
from torch_utils import persistence

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingLoss:
    def __init__(self):
        pass

    def __call__(self, model, images, labels=None):
        t = torch.rand(images.shape[0], device=images.device)
        t_expanded = t.view(-1, *([1] * (images.ndim - 1)))
        x0 = torch.randn_like(images) * model.module.sigma_data
        xt = (1 - t_expanded) * x0 + t_expanded * images

        v_target = images - x0
        v_pred = model(xt, t, labels)

        loss = (v_pred - v_target).square().mean()
        return loss

#----------------------------------------------------------------------------
