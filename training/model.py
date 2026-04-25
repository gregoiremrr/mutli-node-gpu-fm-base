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
            data_coef = self.interpolant.data_coef(t_broad)
            noise_coef = self.interpolant.noise_coef(t_broad).clamp(min=self.eps)
            data_dot = self.interpolant.data_coef_dot(t_broad)
            noise_dot = self.interpolant.noise_coef_dot(t_broad)
            # x_t = data_coef * x_data + noise_coef * x_noise  =>  x_noise = (x_t - data_coef * x_data) / noise_coef
            # v   = data_dot * x_data + noise_dot * x_noise
            #     = (noise_dot / noise_coef) * x_t + (data_dot - data_coef * noise_dot / noise_coef) * x_data
            v_pred = (noise_dot / noise_coef) * xt + (data_dot - data_coef * noise_dot / noise_coef) * pred
            return v_pred


def sample(model, labels, n_samples, n_steps, guidance=1.0, noise=None):
    """
    Sample from the model using a 2nd-order Heun solver with CFG support.
    Integrates dx/dt = v over the grid returned by `model.interpolant.sample_steps`,
    which goes from t_noise to t_data.
    """
    device = next(model.parameters()).device
    interp = model.interpolant
    schedule = interp.sample_steps(n_steps, device)  # [n_steps + 1]

    if noise is None:
        x = torch.randn(
            n_samples, model.img_channels, model.img_resolution, model.img_resolution,
            device=device,
        ) * model.sigma_data
    else:
        x = noise * model.sigma_data

    def get_guided_v(xt, t_cur, labels):
        if guidance == 1.0 or model.uncond_label is None:
            return model(xt, t_cur, class_labels=labels)

        # Batch the conditional and unconditional passes together for efficiency.
        xt_combined = torch.cat([xt, xt], dim=0)
        t_combined = torch.cat([t_cur, t_cur], dim=0)
        l_combined = torch.cat([labels, model.uncond_label], dim=0)

        v_combined = model(xt_combined, t_combined, class_labels=l_combined)
        v_cond, v_uncond = v_combined.chunk(2)

        return torch.lerp(v_uncond, v_cond, guidance)

    with torch.no_grad():
        for i in range(n_steps):
            t_cur = schedule[i].expand(n_samples)
            t_next = schedule[i + 1].expand(n_samples)
            dt = schedule[i + 1] - schedule[i]  # signed, depends on data_side

            # First evaluation (k1).
            k1 = get_guided_v(x, t_cur, labels)

            # Avoid evaluating the model at t_data when predicting x (noise_coef -> 0).
            if i == n_steps - 1 and model.pred == "x":
                x = x + dt * k1
            else:
                # Heun correction (2nd order).
                k2 = get_guided_v(x + dt * k1, t_next, labels)
                x = x + 0.5 * dt * (k1 + k2)

    return x

#----------------------------------------------------------------------------
