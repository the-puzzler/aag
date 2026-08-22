# Released checkpoints

The exact weights behind the headline results. Loaders: see `experiments/*.sh`
for how each is constructed and used.

    celeba/ae.pt                  AE dim=64 (encoder feeds the assignment)
    celeba/generator_uncond.pt    FID 19.36  (2000 ep)
    celeba/generator_cond.pt      FID 20.83  (500 ep, 40 attributes)
    cifar10/ae.pt                 AE dim=64, 32x32, 40 ep -- pairs with
                                  generator_cond.pt
    cifar10/generator_cond.pt     class-conditional (500 ep)
    cifar10/ae_uncond.pt          AE dim=64, 32x32, 150 ep (recon-FID 31.0 vs
                                  33.4 for ae.pt) -- pairs with
                                  generator_uncond.pt
    cifar10/generator_uncond.pt   FID 45.91 (40 ep, lpips_weight 32 rather than
                                  the usual 0.5, on a 64k-step unconditional
                                  assignment). Trained on the UNCONDITIONAL
                                  assignment: the earlier run used the
                                  class-conditional one by mistake and paid
                                  ~7 FID for scrambling the generator could not
                                  use. See experiments/cifar10_uncond.sh.
    doom/frame_ae.pt              per-frame 2D AE dim=64, trained on the 3x
                                  segment cache. Encodes the first-frame
                                  condition for video AND the 3-frame context
                                  for the world model -- both generators below
                                  depend on it, so replace all three together.
    doom/worldmodel_generator.pt  (z, 3 frames, action) -> next frame.
                                  Frame FID 60.18 against real held-out frames
                                  (fresh z, 6k samples), vs 84.0 for the previous
                                  version. Two assignment changes, in order of
                                  effect. First, most of the conditional budget
                                  goes to CONTEXT-only neighbourhoods rather than
                                  action-filtered ones, which took the frames
                                  factor of the independence ratio from 83.8 to
                                  ~1.1 -- z was never really leaking the action,
                                  it was leaking the visual context. Second, the
                                  action step samples the ACTION uniformly rather
                                  than sampling a particle uniformly, so rare
                                  actions get an equal share: Attack went from
                                  1.67 to 1.22 and the per-action mean from 1.17
                                  to 1.04.
                                  An unbalanced variant scores marginally better
                                  on FID (59.56) but was rejected on inspection of
                                  forward-motion rollouts -- FID measures frame
                                  quality, not whether the requested action is
                                  respected. See experiments/doom_worldmodel.sh.
    doom/video_ae.pt              3D video AE dim=64 (spatial grid latent),
                                  trained on the 3x cache; needed only to
                                  rebuild the assignment, not at generation time
    doom/video_generator.pt       first-frame-conditioned 16-frame clips,
                                  held-out FID 56.89. Trained on 550k particles
                                  (9 segments per episode instead of 3) with both
                                  AEs retrained: 10 ep at lr 2e-3, restart to 12
                                  at 5e-4, then to ep24 at 1e-3. FID plateaus at
                                  ~56.8 (56.75/57.99/56.80/56.88/56.89 over
                                  ep16-24); 56.89 is this checkpoint's own value,
                                  not the best of the run.
