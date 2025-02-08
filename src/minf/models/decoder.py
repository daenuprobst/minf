# Adapted from https://github.com/lucidrains/siren-pytorch

import torch
from torch import nn
import torch.nn.functional as F
from minf.models import SirenNet, Modulator
from minf.models.utils import get_grid


class Decoder(nn.Module):
    def __init__(self, net, output_shape, latent_dim=None, n_shapes=0):
        super().__init__()
        assert isinstance(net, SirenNet), "SirenWrapper must receive a Siren network"

        self.net = net
        output_shape = list(output_shape)
        self.output_shape = output_shape[:-1]
        self.output_channels = output_shape[-1]

        if latent_dim is not None and n_shapes > 0:
            self.latents = torch.zeros(n_shapes, latent_dim).normal_(0, 1.0)
            # self.latents = torch.nn.Parameter(self.latents)
            self.latents = self.latents.cuda()

        self.modulator = None
        if latent_dim is not None:
            self.modulator = Modulator(
                dim_in=latent_dim, dim_hidden=net.dim_hidden, num_layers=net.num_layers
            )
        mgrid = get_grid(self.output_shape)
        self.register_buffer("grid", mgrid)

    def forward(self, target=None, *, latent_idx=None, coords=None, output_shape=None):
        modulate = self.modulator is not None
        latent = self.latents[latent_idx] if modulate else None

        assert not (
            modulate ^ (latent is not None)
        ), "latent vector must be only supplied if `latent_dim` was passed in on instantiation"

        mods = self.modulator(latent) if modulate else None

        if coords is None:
            coords = self.grid.clone().detach().requires_grad_()

        out = self.net(coords, mods)

        if output_shape is None:
            output_shape = self.output_shape

        target_shape = output_shape + [self.output_channels]

        out = out.reshape(target_shape)

        if target is not None:
            # Changed from MSE loss as per the adobe paper
            return F.l1_loss(target.view(target_shape), out)

        return out

    def predict(self, latent):
        modulate = self.modulator is not None

        assert not (
            modulate ^ (latent is not None)
        ), "latent vector must be only supplied if `latent_dim` was passed in on instantiation"

        mods = self.modulator(latent) if modulate else None

        coords = self.grid.clone().detach().requires_grad_()

        out = self.net(coords, mods)

        output_shape = self.output_shape

        target_shape = output_shape + [self.output_channels]

        out = out.reshape(target_shape)

        return out
