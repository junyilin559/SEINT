# README: Applying SEINT as a Regularization Term in Molecular Generation and Point Cloud Reconstruction

This project demonstrates the application of **SEINT** as a regularization term in two generative tasks: **point cloud reconstruction** and **molecular generation**. The modifications are based on the following open-source repositories:

- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) for point cloud reconstruction
- [E3 Diffusion for Molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules) for molecular generation

---

## 🔷 1. Point Cloud Reconstruction with SEINT  
**Base repository**: [Pang-Yatian/Point-MAE](https://github.com/Pang-Yatian/Point-MAE)

We integrate SEINT as a regularization term into the Point-MAE architecture for enhanced point cloud reconstruction performance. To test our implementation:

1. Clone the original repository:
   ```bash
   git clone https://github.com/Pang-Yatian/Point-MAE.git
   ```

2. Replace the following three directories in the original codebase with the modified versions from this repository:
   - `cfgs/`
   - `extensions/chamfer_dist/`
   - `models/`

3. The modifications incorporate SEINT regularization during training and evaluation.

4. Follow the original training and evaluation instructions in the Point-MAE repository.

---

## 🔷 2. Molecular Generation with SEINT  
**Base repository**: [ehoogeboom/e3_diffusion_for_molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules)

We also introduce SEINT into the E(3)-equivariant diffusion model for molecular generation. To run experiments:

1. Clone the original repository:
   ```bash
   git clone https://github.com/ehoogeboom/e3_diffusion_for_molecules.git
   ```

2. Replace the following file with the SEINT-modified version:
   - `en_diffusion/en_diffusion.py`

3. This updated file adds SEINT as an auxiliary regularization term during the diffusion process.

4. Use the original training and sampling scripts for evaluation.


