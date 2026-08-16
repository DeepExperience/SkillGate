# Runtime cache assets

This public directory contains data required by maintained runners, not launch
scripts:

- `pkg/`: TB2 uv cache, Harbor binaries, apt/pip inputs, and a pytest wheel;
- `images/`: the Claw sandbox Dockerfile cache;
- `cuda_fast_home/`: five environment symlinks into `/usr/local/cuda`.

Historical one-off waiters and remote launch helpers were removed with the
legacy compatibility tree. New scripts do not belong here.
