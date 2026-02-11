import torch
import torch.nn as nn


class AntixKVanillaVAE(nn.Module):
    """
    Existing implementation adapted from:
    https://github.com/AntixK/PyTorch-VAE (models/vanilla_vae.py)
    """

    def __init__(self, image_channels: int = 1, latent_dim: int = 128, image_size: int = 128):
        super().__init__()
        if image_size % 32 != 0:
            raise ValueError("image_size must be divisible by 32 for AntixKVanillaVAE.")

        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.hidden_dims = [32, 64, 128, 256, 512]

        encoder_layers = []
        in_channels = image_channels
        for hidden_dim in self.hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)

        feat_spatial = image_size // (2 ** len(self.hidden_dims))
        feat_dim = self.hidden_dims[-1] * feat_spatial * feat_spatial
        self._feat_spatial = feat_spatial
        self._feat_dim = feat_dim

        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, feat_dim)

        decoder_layers = []
        rev_dims = list(reversed(self.hidden_dims))
        for i in range(len(rev_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        rev_dims[i],
                        rev_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(rev_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.decoder = nn.Sequential(*decoder_layers)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                rev_dims[-1],
                image_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = self.decoder_input(z)
        h = h.view(z.size(0), self.hidden_dims[-1], self._feat_spatial, self._feat_spatial)
        h = self.decoder(h)
        return self.final_layer(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
