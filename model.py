"""LeNet adaptation from Z. Zhou's Assignment 2. MaxBlurPool follows Zhang (2019)."""
import torch
from torch import nn
from torch.nn import functional as F


class MaxBlurPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        b = torch.tensor([1., 2., 1.])
        self.register_buffer("kernel", (torch.outer(b, b) / 16).repeat(channels, 1, 1, 1))

    def forward(self, x):
        a = F.max_pool2d(x, 2, stride=1)
        # One pixel of edge-exclusive reflection; equivalent to F.pad(..., 'reflect').
        # Slices avoid the nondeterministic CUDA reflection-pad backward kernel.
        a = torch.cat((a[..., 1:2], a, a[..., -2:-1]), dim=-1)
        a = torch.cat((a[:, :, 1:2], a, a[:, :, -2:-1]), dim=-2)
        return F.conv2d(a, self.kernel, stride=2, groups=a.shape[1])


class LeNet(nn.Module):
    def __init__(self, blur=False, historical=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1 if historical else 3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400 if historical else 576, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool1 = MaxBlurPool(6) if blur else nn.MaxPool2d(2, 2)
        self.pool2 = MaxBlurPool(16) if blur else nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool1(self.conv1(x).tanh())
        x = self.pool2(self.conv2(x).tanh())
        x = self.fc1(x.flatten(1)).tanh()
        return self.fc3(self.fc2(x).tanh())


def shift(x, dx, dy, border="reflect"):
    """Positive dx/dy moves content right/down. Input is normalized for mean fill."""
    b, _, h, w = x.shape
    dx = torch.as_tensor(dx, device=x.device).reshape(-1, 1)
    dy = torch.as_tensor(dy, device=x.device).reshape(-1, 1)
    cols = torch.arange(w, device=x.device)[None, :] - dx
    rows = torch.arange(h, device=x.device)[None, :] - dy
    if border == "reflect":
        cols = (w - 1) - ((cols % (2*w - 2)) - (w - 1)).abs()
        rows = (h - 1) - ((rows % (2*h - 2)) - (h - 1)).abs()
    elif border != "mean":
        raise ValueError(border)
    good = ((cols >= 0) & (cols < w))[:, None, :] & ((rows >= 0) & (rows < h))[:, :, None]
    view = x.permute(0, 2, 3, 1)[torch.arange(b, device=x.device)[:, None, None],
                                rows.clamp(0, h-1)[:, :, None], cols.clamp(0, w-1)[:, None, :]]
    if border == "mean":
        view = view * good[..., None]
    return view.permute(0, 3, 1, 2)


def metrics(y, clean, views):
    return dict(flip_rate=float((views != clean[:, None]).mean()),
                clean_accuracy=float((clean == y).mean()),
                shifted_accuracy=float((views == y[:, None]).mean()))


def epoch_plan(seed, epoch, count):
    import numpy as np
    order = np.random.default_rng([seed, epoch, 1]).permutation(count)
    # Indexed by training-subset row, independent of the order stream.
    offsets = np.random.default_rng([seed, epoch, 2]).integers(-2, 3, (count, 2))
    return order, offsets
