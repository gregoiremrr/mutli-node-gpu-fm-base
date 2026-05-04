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

        # Sample t from the interpolant's training distribution.
        t = interp.sample_t(images.shape[0], images.device)
        t_expanded = t.view(-1, *([1] * (images.ndim - 1)))

        x_noise = torch.randn_like(images) * sigma_data
        xt = interp.data_coef(t_expanded) * images + interp.noise_coef(t_expanded) * x_noise
        v_target = interp.data_coef_dot(t_expanded) * images + interp.noise_coef_dot(t_expanded) * x_noise

        v_pred, logvar = model(xt, t, labels, return_logvar=True)

        # EDM2-style adaptive weighting (Karras et al., 2024).
        residual_sq = (v_pred - v_target).square()
        weighted_loss = (torch.exp(-logvar) * residual_sq + logvar).mean()

        # Side stats.
        stats = dict(
            mse=residual_sq.mean().detach(),
            logvar=logvar.mean().detach(),
        )
        return weighted_loss, stats

#----------------------------------------------------------------------------
