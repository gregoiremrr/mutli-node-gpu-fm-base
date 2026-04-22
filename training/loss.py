import torch
from torch_utils import persistence

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingLoss:
    def __init__(self, p_uncond=0.1):
        """
        Args:
            p_uncond: The probability of dropping the class label to train the unconditional branch.
        """
        self.p_uncond = p_uncond

    def __call__(self, model, images, labels=None):

        if labels is not None and self.p_uncond > 0.0:
            drop_mask = torch.rand(labels.shape[0], 1, device=labels.device) < self.p_uncond
            labels = torch.where(drop_mask, torch.zeros_like(labels), labels)

        t = torch.rand(images.shape[0], device=images.device)
        t_expanded = t.view(-1, *([1] * (images.ndim - 1)))

        # We assume that the model is wrapped in DDP
        x0 = torch.randn_like(images) * model.module.sigma_data
        xt = (1 - t_expanded) * x0 + t_expanded * images

        v_target = images - x0
        v_pred = model(xt, t, labels)

        loss = (v_pred - v_target).square().mean()
        return loss

#----------------------------------------------------------------------------
