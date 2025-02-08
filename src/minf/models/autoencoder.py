import torch
from torch import nn
import torch.nn.functional as F

from minf.models import SirenNet, Modulator, Encoder
from minf.models.utils import get_grid


class Autoencoder(nn.Module):
    def __init__(self, net, output_shape, latent_dim=64, in_channels=1):
        super().__init__()
        assert isinstance(net, SirenNet), "SirenWrapper must receive a Siren network"

        self.net = net
        self.encoder = Encoder(in_channels, latent_dim, 3)
        output_shape = list(output_shape)
        self.output_shape = output_shape[:-1]
        self.output_channels = output_shape[-1]

        self.modulator = None
        if latent_dim is not None:
            self.modulator = Modulator(
                dim_in=latent_dim, dim_hidden=net.dim_hidden, num_layers=net.num_layers
            )
        mgrid = get_grid(self.output_shape)
        self.register_buffer("grid", mgrid)

    def forward(self, target=None, *, coords=None, output_shape=None):
        modulate = self.modulator is not None
        latent = self.encoder(torch.movedim(target, -1, 1))

        assert not (
            modulate ^ (latent is not None)
        ), "latent vector must be only supplied if `latent_dim` was passed in on instantiation"

        mods = self.modulator(latent) if modulate else None

        if coords is None:
            coords = (
                self.grid.clone()
                .detach()
                .requires_grad_()
                .unsqueeze(0)
                .expand(latent.size(0), -1, -1)
            )

        out = self.net(coords, mods)

        if output_shape is None:
            output_shape = self.output_shape

        target_shape = [latent.size(0)] + output_shape + [self.output_channels]
        out = out.reshape(target_shape)

        if target is not None:
            return F.l1_loss(target.view(target_shape), out), out, latent

        return out

    def encode(self, target):
        return self.encoder(torch.movedim(target, -1, 1))

    def predict(self, latent):
        modulate = self.modulator is not None

        assert not (
            modulate ^ (latent is not None)
        ), "latent vector must be only supplied if `latent_dim` was passed in on instantiation"

        mods = self.modulator(latent) if modulate else None
        coords = (
            self.grid.clone()
            .detach()
            .requires_grad_()
            .unsqueeze(0)
            .expand(latent.size(0), -1, -1)
        )

        out = self.net(coords, mods)

        target_shape = [latent.size(0)] + self.output_shape + [self.output_channels]
        out = out.reshape(target_shape)

        return out
