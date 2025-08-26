# Fast InversePhotonics: PINN + Quantum GAN Inverse Design Toolkit

A hybrid **physics-informed neural network (PINN) + quantum GAN (QGAN)** framework for **fast, scalable inverse design of photonic devices**.  
By embedding analytical resonance models (e.g., Fano-like spectral responses) into the loss function, the method ensures **physical fidelity**. Quantum-enhanced generative modeling enables efficient exploration of high-dimensional design spaces, achieving:

- **200× fewer training samples**
- **25× faster convergence**
- **~100× higher Q-factors** in generated device designs

This framework provides a scalable route for rapid prototyping of photonic components in **sensing, imaging, and optical communication**.

---

## Training Data

The absorption spectra used in this project are **taken from the dataset provided by the [UCLA Raman Lab – Multiclass Metasurface Inverse Design project](https://github.com/Raman-Lab-UCLA/Multiclass_Metasurface_InverseDesign/)**.  

- The dataset (`absorptionData_HybridGAN.csv`) contains absorption spectra for multiple structures over the wavelength range **4–12 μm**.  
- Each row corresponds to a unique **structure ID** (`Var1_1`) followed by spectral values (`Var1_2 … Var1_801`).  
- The spectra are converted into **frequency domain (THz)** for Fano lineshape fitting and physics-informed training.

```
C. Yeung, et al. Global Inverse Design across Multiple Photonic Structure Classes Using Generative Deep Learning. Advanced Optical Materials, 2021.
```

---

| Stage                                                | Key files & roles                                                                   | Notes                                                                                                                                                                                                                                                                                                                                                     |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Spectral pre‑processing**                       | `fitting.py`                                                                        | Reads raw absorption spectra (`absorptionData_HybridGAN.csv`), converts wavelengths to frequency, and fits a Fano lineshape to extract four physical parameters: amplitude $A_0$, resonance $\omega_0$, linewidth $\Gamma$, asymmetry $q$. Produces `fano_fit_results.csv`.                                                                               |
| **2. Latent‑space autoencoder**                      | `models.py` (`IWAE` class), `train_iwae_abs.py`                                     | The IWAE learns a compact latent representation of metasurface images (64×64 RGB). After training, `pretrained_iwae.pth` is saved for reuse.                                                                                                                                                                                                              |
| **3. Adversarial training with physics constraints** | Classical path: `train-test-gan-pinn.py`<br>Quantum path: `train-test-qgan-pinn.py` | Both scripts combine image generation with physics-informed loss. The discriminator has two heads: one for real/fake classification, one regressing the four Fano parameters. The loss penalizes departures from desired resonance/quality-factor targets. In the quantum variant, the generator is a PennyLane variational circuit (`QuantumGenerator`). |
| **4. Design validation / inverse design**            | `validation-gan-pinn.py`, `validation-qgan-pinn.py`                                 | Starting from random physics parameters, these scripts perform gradient-based search to reach specified targets, then decode the corresponding image.                                                                                                                                                                 |


## Project Structure & Python File Descriptions

All Python scripts are located inside the [**`code/`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/tree/main/code) folder

- [**`fitting.py`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/blob/main/code/fitting.py)
  Fits **Fano resonance parameters** to all absorption spectra in the dataset.  
  - Input: CSV (`absorptionData_HybridGAN.csv`) with structure IDs and spectra  
  - Converts wavelength (µm) to frequency (THz) using ω = c / λ  
  - Fits **Fano lineshape**:  
    $$A(\omega) = A_0 \cdot \frac{(q + \epsilon)^2}{1 + \epsilon^2}, \quad 
    \epsilon = \frac{2(\omega - \omega_0)}{\Gamma} $$  
  - Extracted parameters per structure:  
    - `A0` (amplitude)  
    - `q` (asymmetry factor)  
    - `w0_THz` (resonance frequency, THz)  
    - `Gamma_THz` (linewidth, THz)  
  - Outputs a new CSV (`fano_fit_results.csv`) with columns:  
    ```
    structure_name, A0, w0_THz, q, Gamma_THz
    ```

- [**`train_iwae_abs.py`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/blob/main/code/train_iwae_abs.py)  
  Trains an **Importance-Weighted Autoencoder (IWAE)** on image datasets (64×64 RGB).  
  - Logs loss per epoch  
  - Saves pretrained model `pretrained_iwae.pth`  
  - Plots training curves (`losses-iwae-abs.png`)  
  - Configurable via `img_path`, `batch_size`, `epochs`, etc.

- [**`train-test-qgan-pinn.py`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/blob/main/code/train-test-qgan-pinn.py)  
  Implements the **QGAN + PINN training pipeline**.  
  - Loads image/spectral datasets  
  - Builds quantum generator, discriminator, and IWAE decoder  
  - Applies physics-informed loss with resonance priors  
  - Outputs model checkpoints and generated designs  

- [**`validation-qgan-pinn.py`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/blob/main/code/validation-qgan-pinn.py)  
  Validates trained **QGAN + PINN models**.  
  - Performs spectral parameter optimization for target Q-factors  
  - Generates corresponding metasurface designs  
  - Evaluates spectral fidelity and saves results  

- [**`train-test-gan-pinn.py`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/blob/main/code/train-test-gan-pinn.py)  
  Baseline **classical GAN + PINN** framework.  
  - Convolutional generator and dual-head discriminator  
  - Trains with physics-based loss constraints  
  - Logs loss curves and generates output designs  

- [**`validation-gan-pinn.py`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/blob/main/code/validation-gan-pinn.py)  
  Validates the **classical GAN + PINN** model.  
  - Optimizes design parameters  
  - Generates images and evaluates spectral targets  
  - Stores validation outputs  

- [**`models.py`**](https://github.com/MahindraRajan/Physics-Informed-QGAN/blob/main/code/models.py)  
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
- NumPy, SciPy, Matplotlib, Pandas
