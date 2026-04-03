import torch
from torch_utils import persistence
import dnnlib

#----------------------------------------------------------------------------

@persistence.persistent_class
class FlowMatchingModel(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        sigma_data=0.5,
        eps=1e-5,
        use_fp16=False,
        net_kwargs=None,
        pred="v"
    ):
        assert pred in ["x", "v"]
        super().__init__()
        self.sigma_data = sigma_data
        self.eps = eps
        self.use_fp16 = use_fp16
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
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

        # Net runs in FP16, but scaling is done in FP32.
        F_x = self.net(
            (xt / self.sigma_data).to(dtype),
            t,
            class_labels=class_labels,
        )
        pred = self.sigma_data * F_x.to(torch.float32)
        if self.pred == "v":
            return pred
        elif self.pred == "x":
            t_broad = t.reshape(-1, *([1] * (xt.ndim - 1)))
            v_pred = (pred - xt) / (1 - t_broad).clamp(min=self.eps)
            return v_pred

    def sample(self, class_labels, n_samples, n_steps, device):
        dt = 1.0 / n_steps
        x = torch.randn(
            n_samples, self.img_channels, self.img_resolution, self.img_resolution,
            device=device
        ) * self.sigma_data

        with torch.no_grad():
            for i in range(n_steps):
                t = torch.full([n_samples], i * dt, device=device)
                k1 = self(x, t, class_labels=class_labels)
                x_pred = x + dt * k1
                t_next = torch.full([n_samples], (i + 1) * dt, device=device)
                k2 = self(x_pred, t_next, class_labels=class_labels)
                x = x + 0.5 * dt * (k1 + k2)

        return x

#----------------------------------------------------------------------------
