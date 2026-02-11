from models.antixk_vae import AntixKVanillaVAE
from models.vae import ConvVAE


def build_vae(use_antixk_vae: bool, image_channels: int, latent_dim: int, image_size: int):
    if use_antixk_vae:
        return AntixKVanillaVAE(image_channels=image_channels, latent_dim=latent_dim, image_size=image_size)
    return ConvVAE(image_channels=image_channels, latent_dim=latent_dim, image_size=image_size)
