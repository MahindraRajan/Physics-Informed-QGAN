# Fast InversePhotonics: PINN + Quantum GAN Inverse Design Toolkit

A hybrid **physics-informed neural network (PINN) + quantum GAN (QGAN)** framework for **fast, scalable inverse design of photonic devices**.  
By embedding analytical resonance models (e.g., Fano-like spectral responses) into the loss function, the method ensures **physical fidelity**. Quantum-enhanced generative modeling enables efficient exploration of high-dimensional design spaces, achieving:

- **200× fewer training samples**
- **25× faster convergence**
- **~100× higher Q-factors** in generated device designs

This framework provides a scalable route for rapid prototyping of photonic components in **sensing, imaging, and optical communication**.

---

## Project Structure & Python File Descriptions

All Python scripts are located inside the [**`code/`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/tree/main/code) folder

- **`train_iwae_abs.py`**  
  Trains an **Importance-Weighted Autoencoder (IWAE)** on image datasets (64×64 RGB).  
  - Logs loss per epoch  
  - Saves pretrained model `pretrained_iwae.pth`  
  - Plots training curves (`losses-iwae-abs.png`)  
  - Configurable via `img_path`, `batch_size`, `epochs`, etc.

- **`train-test-qgan-pinn.py`**  
  Implements the **QGAN + PINN training pipeline**.  
  - Loads image/spectral datasets  
  - Builds quantum generator, discriminator, and IWAE decoder  
  - Applies physics-informed loss with resonance priors  
  - Outputs model checkpoints and generated designs  

- **`validation-qgan-pinn.py`**  
  Validates trained **QGAN + PINN models**.  
  - Performs spectral parameter optimization for target Q-factors  
  - Generates corresponding metasurface designs  
  - Evaluates spectral fidelity and saves results  

- **`train-test-gan-pinn.py`**  
  Baseline **classical GAN + PINN** framework.  
  - Convolutional generator and dual-head discriminator  
  - Trains with physics-based loss constraints  
  - Logs loss curves and generates output designs  

- **`validation-gan-pinn.py`**  
  Validates the **classical GAN + PINN** model.  
  - Optimizes design parameters  
  - Generates images and evaluates spectral targets  
  - Stores validation outputs  

- **`models.py`**  
  Contains model definitions:  
  - `IWAE`: encoder/decoder with importance-weighted training  
  - `BetaVAE`: baseline variational autoencoder  
  - `Discriminator`: with auxiliary regression head for physical parameters  
  - `QuantumGenerator`: variational quantum circuit (e.g., Efficient-SU2 ansatz) for quantum-assisted generation  

---

## Installation

### Requirements
- Python ≥ 3.9  
- [PyTorch](https://pytorch.org/)  
- [PennyLane](https://pennylane.ai/) (for quantum backends)  
- NumPy, SciPy, Matplotlib
