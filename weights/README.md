# Released checkpoints

The exact weights behind the headline results. Loaders: see `experiments/*.sh`
for how each is constructed and used.

    celeba/ae.pt                  AE dim=64 (encoder feeds the assignment)
    celeba/generator_uncond.pt    FID 19.36  (2000 ep)
    celeba/generator_cond.pt      FID 20.83  (500 ep, 40 attributes)
    cifar10/ae.pt                 AE dim=64, 32x32
    cifar10/generator_cond.pt     class-conditional (500 ep)
    doom/frame_ae.pt              per-frame 2D AE dim=64 (world-model latents +
                                  the first-frame condition for video)
    doom/worldmodel_generator.pt  (z, 3 frames, action) -> next frame (500 ep)
    doom/video_ae.pt              3D video AE dim=64 (spatial grid latent)
    doom/video_generator.pt       first-frame-conditioned 16-frame clips,
                                  held-out FID 60.10 (ep 16)
