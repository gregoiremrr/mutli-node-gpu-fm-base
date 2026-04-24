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

        # We assume that the model is wrapped in DDP.
        interp = model.module.interpolant
        sigma_data = model.module.sigma_data

        # Sample t uniformly in the interpolant's time range.
        t = torch.rand(images.shape[0], device=images.device)
        t = t * (interp.t_max - interp.t_min) + interp.t_min
        t_expanded = t.view(-1, *([1] * (images.ndim - 1)))

        x0 = torch.randn_like(images) * sigma_data
        xt = interp.alpha(t_expanded) * x0 + interp.beta(t_expanded) * images
        v_target = interp.alpha_dot(t_expanded) * x0 + interp.beta_dot(t_expanded) * images

        v_pred = model(xt, t, labels)

        loss = (v_pred - v_target).square().mean()
        return loss

#----------------------------------------------------------------------------
