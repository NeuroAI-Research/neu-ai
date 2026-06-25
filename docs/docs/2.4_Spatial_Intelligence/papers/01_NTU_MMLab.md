# 1 [NTU MMLab](https://www.mmlab-ntu.com)

## 3D/4D Reconstruction

| Feature | 4RC (2026) | PhysX-Anything (2025) | STREAM3R (2025) |
| --- | --- | --- | --- |
| **Input** | Monocular video | Single image | Streaming images / Video frames |
| **Core Mechanism** | Spatiotemporal latent query encoding | Multi-round VLM dialog + Diffusion | Decoder-only Causal Attention + KVCache |
| **Geometry Format** | Point-maps & 3D displacement vectors | Highly compressed Voxel grids ($32^3$) $\to$ Fine meshes | Dual-coordinate Point-maps |
| **Temporal Handling** | Direct indexing of any target timestamp $\tau$ | N/A (Static asset generation) | Sequential incremental updates (On-the-fly) |
| **Downstream Focus** | Point tracking, dense motion modeling | Robotic manipulation / MuJoCo simulation | Online 3D Mapping, Monocular/Video depth |
