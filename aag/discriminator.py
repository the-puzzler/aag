"""PatchGAN discriminator and adversarial losses for the AE's refinement phase.

This is DC-AE's third training phase (arXiv 2410.10733 sec 3.2): after the
reconstruction-only run converges, "only tune the decoder's head layers while
freezing all the other layers" under a GAN loss, at low resolution. The paper
notes the reconstruction loss "is sufficient for learning to reconstruct the
content and semantics" while "the GAN loss mainly improves local details" --
which is precisely the failure we measured (high-frequency energy at 58.5% of
the original's, error concentrated in the busiest patches).

DC-AE defers its GAN hyperparameters to SD-VAE and its training pipeline is not
released, so the numbers here come from the SD-VAE source rather than from the
paper: taming-transformers' NLayerDiscriminator, and latent-diffusion's
LPIPSWithDiscriminator for the hinge loss and adaptive weighting.

Freezing the encoder is not just for GAN stability -- it keeps the latents
bit-identical, so a decoder refined here drops into the existing particle and
assignment pipeline without rebuilding anything.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    """PatchGAN, following taming-transformers' NLayerDiscriminator.

    n_layers is deliberately exposed because the standard n_layers=3 has a 70px
    receptive field -- larger than a whole 64x64 frame, which makes it an
    image-level critic rather than a patch one. n_layers=2 gives a 34px field,
    about half the frame, and keeps the "patch" in PatchGAN at this resolution.
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 2):
        super().__init__()
        kw, padw = 4, 1
        seq = [nn.Conv2d(in_channels, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 2, padw, bias=False),
                    nn.BatchNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 1, padw, bias=False),
                nn.BatchNorm2d(ndf * mult), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * mult, 1, kw, 1, padw)]
        self.main = nn.Sequential(*seq)
        self.apply(_init_weights)

    def forward(self, x):
        return self.main(x)


def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def hinge_d_loss(logits_real, logits_fake):
    """Discriminator hinge loss -- latent-diffusion's default disc_loss."""
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def g_loss_from(logits_fake):
    """Non-saturating generator term, as in LPIPSWithDiscriminator."""
    return -logits_fake.mean()


def adaptive_weight(rec_loss, g_loss, last_layer, max_weight: float = 1e4):
    """Balance the adversarial gradient against the reconstruction gradient.

        d_weight = ||d rec_loss / d last_layer|| / ||d g_loss / d last_layer||

    from latent-diffusion's calculate_adaptive_weight. Without it the right
    adversarial weight varies over training by orders of magnitude and has to be
    hand-tuned; with it the GAN term is held at a fixed fraction of whatever the
    reconstruction gradient currently is. Computed in fp32 -- under bf16
    autocast these norms are small enough for the ratio to be badly quantised.
    """
    rec_g = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0].float()
    gan_g = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0].float()
    w = rec_g.norm() / (gan_g.norm() + 1e-4)
    return w.clamp(0.0, max_weight).detach()


def decoder_head_parameters(ae, n_modules: int):
    """Freeze everything but the last n_modules of the decoder stack.

    DC-AE phase 3 tunes "the decoder's head layers" only. For the dcae decoder
    at 64x64 the stack is [UpBlock, UpBlock, UpBlock, Upsample, Conv2d(ch,3)],
    so n_modules=2 is the final upsample plus the output convolution.

    Returns the parameters left trainable, so the caller can hand exactly those
    to the optimiser rather than relying on requires_grad alone.
    """
    core = ae._orig_mod if hasattr(ae, "_orig_mod") else ae
    for p in core.parameters():
        p.requires_grad_(False)
    head = core.dec.net[-n_modules:]
    for p in head.parameters():
        p.requires_grad_(True)
    return [p for p in core.parameters() if p.requires_grad]


def last_conv_weight(ae):
    """The output convolution's weight -- the reference layer for adaptive_weight."""
    core = ae._orig_mod if hasattr(ae, "_orig_mod") else ae
    return core.dec.net[-1].weight
