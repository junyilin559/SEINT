# An Efficient SE($p$)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation
This repository includes the implementation of our work **"An Efficient SE($p$)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation"**

## Introduction
Brief introduction to directories and files:
* `SEINT/`: Core code for our implementation of the SEINT/ISEINT algorithm package.
* Experiments for validating metric properties:
   * `SE(p) invariance/`: Code for implementing point cloud classification tasks.
   * `Cross Space tasks/`: Code for implementing cross-space tasks.
   * `High-dimensional data analysis/`: Code for testing with high-dimensional data.
   * `Metric consistency/`: Code for metric consistency detection.
* As a regularization term:
   * `E3-diffusion/`: Core code for its use as a regularization term in E(3)-equivariant diffusion models.
   * `Point-MAE/`: Core code for its use as a regularization term in Point-MAE.

## Requirements
* python >= 3.8
* numpy
* scipy
* matplotlib
* sklearn
* pytorch >= 2.4.1
* pandas


## Applying SEINT as a Regularization Term in Molecular Generation and Point Cloud Reconstruction

This project demonstrates the application of **SEINT** as a regularization term in two generative tasks: **point cloud reconstruction** and **molecular generation**. The modifications are based on the following open-source repositories:

- [E3 Diffusion for Molecules](https://github.com/ehoogeboom/e3_diffusion_for_molecules) for molecular generation
- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) for point cloud reconstruction

---

### 🔷 1. Molecular Generation with SEINT  
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
---

### 🔷 2. Point Cloud Reconstruction with SEINT  
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

3. The modifications incorporate SEINT regularization during pretraining.

4. Follow the original training and evaluation instructions in the Point-MAE repository.

## Main References
Hoogeboom, Emiel and Satorras, Victor Garcia and Vignac, Clement and Welling, Max "Equivariant diffusion for molecule generation in 3d." International conference on machine learning. PMLR, 2022. [\[Git\]](https://github.com/ehoogeboom/e3_diffusion_for_molecules)

Pang, Yatian and Wang, Wenxiao and Tay, Francis EH and Liu, Wei and Tian, Yonghong and Yuan, Li. "Masked autoencoders for point cloud self-supervised learning." European conference on computer vision. Cham: Springer Nature Switzerland, 2022. [\[Git\]](https://github.com/Pang-Yatian/Point-MAE)

Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T.H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, and Titouan Vayer. "POT Python Optimal Transport library." Journal of Machine Learning Research 22(78): 1-8, 2021. [\[Web\]](https://pythonot.github.io/)

Scetbon, Meyer, Gabriel Peyré, and Marco Cuturi. "Linear-time Gromov Wasserstein distances using low rank couplings and costs." International Conference on Machine Learning. PMLR, 2022. [\[Git\]](https://github.com/meyerscetbon/LinearGromov)

Vayer, Titouan and Flamary, Remi and Tavenard, Romain and Chapel, Laetitia and Courty, Nicolas "Sliced Gromov-Wasserstein." NeurIPS 2019-Thirty-third Conference on Neural Information Processing Systems. Vol. 32. 2019. [\[Git\]](https://github.com/tvayer/SGW)







