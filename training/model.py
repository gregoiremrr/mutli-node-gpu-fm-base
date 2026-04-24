import torch
from torch_utils import persistence
import dnnlib

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingModel(torch.nn.Module):
    def __init__(
        self,
        pred,
        img_resolution,
        img_channels,
        sigma_data,
        label_dim=0,
        t_scale=1000,
        eps=0.05,
        use_fp16=False,
        net_kwargs=None,
        interpolant_kwargs=None,
    ):
        assert pred in ["x", "v"]
        super().__init__()
        self.sigma_data = sigma_data
        self.eps = eps
        self.use_fp16 = use_fp16
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.t_scale = t_scale
        if label_dim > 0:
            self.register_buffer('uncond_label', torch.zeros([1, label_dim]))
        else:
            self.uncond_label = None
        if interpolant_kwargs is None:
            interpolant_kwargs = dict(class_name='training.interpolants.LinearInterpolant')
        self.interpolant = dnnlib.util.construct_class_by_name(**interpolant_kwargs)
        self.net = dnnlib.util.construct_class_by_name(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **net_kwargs,
        )
        self.pred = pred

    def forward(self, xt, t, class_labels=None, force_fp32=False):
        xt = xt.to(torch.float32)
        t = t.to(torch.float32)
        dtype = (torch.float16
                 if (self.use_fp16 and not force_fp32 and xt.device.type == 'cuda')
                 else torch.float32)

        # Normalize t to [0, 1] before scaling so that t_scale has consistent
        # semantics across interpolants with different t ranges.
        t_min, t_max = self.interpolant.t_min, self.interpolant.t_max
        t_norm = (t - t_min) / (t_max - t_min)
        t_scaled = t_norm * self.t_scale

        # Net runs in FP16, but scaling is done in FP32.
        F_x = self.net(
            (xt / self.sigma_data).to(dtype),
            t_scaled,
            class_labels=class_labels,
        )
        pred = self.sigma_data * F_x.to(torch.float32)
        if self.pred == "v":
            return pred
        elif self.pred == "x":
            t_broad = t.reshape(-1, *([1] * (xt.ndim - 1)))
            alpha = self.interpolant.alpha(t_broad).clamp(min=self.eps)
            beta = self.interpolant.beta(t_broad)
            alpha_dot = self.interpolant.alpha_dot(t_broad)
            beta_dot = self.interpolant.beta_dot(t_broad)
            # v = (alpha_dot / alpha) * xt + (beta_dot - alpha_dot * beta / alpha) * x1_hat
            v_pred = (alpha_dot / alpha) * xt + (beta_dot - alpha_dot * beta / alpha) * pred
            return v_pred


def sample(model, labels, n_samples, n_steps, guidance=1.0, noise=None):
    """
    Sample from the model using a 2nd-order Heun solver with CFG support.
    Integrates the ODE dx/dt = v from t_min to t_max (defined by model.interpolant).
    """
    device = next(model.parameters()).device
    interp = model.interpolant
    t_min, t_max = interp.t_min, interp.t_max
    dt = (t_max - t_min) / n_steps

    if noise is None:
        x = torch.randn(
            n_samples, model.img_channels, model.img_resolution, model.img_resolution,
            device=device
        ) * model.sigma_data
    else:
        x = noise * model.sigma_data

    def get_guided_v(xt, t_cur, labels):
        if guidance == 1.0 or model.uncond_label is None:
            return model(xt, t_cur, class_labels=labels)

        # Batch the conditional and unconditional passes together for efficiency
        # Double the batch size: [conditional_batch, unconditional_batch]
        xt_combined = torch.cat([xt, xt], dim=0)
        t_combined = torch.cat([t_cur, t_cur], dim=0)
        l_combined = torch.cat([labels, model.uncond_label], dim=0)

        v_combined = model(xt_combined, t_combined, class_labels=l_combined)
        v_cond, v_uncond = v_combined.chunk(2)

        return torch.lerp(v_uncond, v_cond, guidance)

    with torch.no_grad():
        for i in range(n_steps):
            t = torch.full([n_samples], t_min + i * dt, device=device)

            # First evaluation (k1)
            k1 = get_guided_v(x, t, labels)

            # Avoid evaluating the model at t_max when predicting x (alpha->0 singularity)
            if i == n_steps - 1 and model.pred == "x":
                x = x + dt * k1
            else:
                # Standard Heun correction (2nd order)
                x_pred = x + dt * k1
                t_next = torch.full([n_samples], t_min + (i + 1) * dt, device=device)

                # Second evaluation (k2)
                k2 = get_guided_v(x_pred, t_next, labels)
                x = x + 0.5 * dt * (k1 + k2)

    return x

#----------------------------------------------------------------------------
