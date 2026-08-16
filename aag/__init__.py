"""Persistent Global Gaussian Assignment — CIFAR experiment package.

Implements the recipe from `global_gaussian_assignment_report`:
  1. Train an ordinary reconstruction autoencoder, encode the full train set.
  2. Whiten the latent cloud, initialize one persistent particle per example.
  3. Build a persistent Gaussian assignment via:
        - greedy global 1D rank transport (direction search on a subset),
        - conditional offset-slab cleanup,
        - periodic radial chi_d calibration.
  4. Freeze the assignment, train a direct decoder D(z_i*) ~= h_i.
  5. Evaluate fresh-prior one-pass generation.
"""
