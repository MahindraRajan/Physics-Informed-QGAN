# FastInversePhotonics: PINN + Quantum GAN Inverse Design Toolkit

A hybrid **physics-informed neural network (PINN) + quantum GAN (QGAN)** framework enabling fast, data-efficient inverse design of dielectric metasurfaces with ultra-high Q resonances. Embedded analytical models (e.g., Fano-like spectral profiles) ensure physical fidelity, while quantum-enhanced generative modeling accelerates design exploration—achieving **200× fewer training samples**, **25× faster convergence**, and resulting in designs with **100× higher Q-factors**.

> A README serves as both documentation and the project’s first impression—keep it concise, informative, and inviting. :contentReference[oaicite:1]{index=1}

---

##  Project Structure & Python File Descriptions

│  
├── **`train_iwae_abs.py`**  
│   └ Trains an Importance-Weighted Autoencoder (IWAE) on your image dataset (64×64 RGB); logs loss per epoch, saves the pretrained model (`pretrained_iwae.pth`), and plots `losses-iwae-abs.png`. Adjustable via `img_path`, `batch_size`, `epochs`, etc.  

├── **`train-test-qgan-pinn.py`**  
│   └ Runs the full QGAN + PINN training pipeline: loads image data and spectral parameters, constructs a quantum generator, discriminator, and the IWAE decoder; applies a physics-informed loss incorporating the Fano resonance model; outputs model checkpoints and training visuals.  

├── **`validation-qgan-pinn.py`**  
│   └ Validates the trained QGAN + IWAE on design targets: optimizes spectral parameters to achieve desired Q-factors and resonance behavior, then uses the IWAE decoder to generate matching metasurface images and outputs results.  

├── **`train-test-gan-pinn.py`**  
│   └ Classical baseline using a standard GAN + physics constraints: builds and trains a convolutional generator and dual-head discriminator; logs losses and generates image outputs for comparison.  

├── **`validation-gan-pinn.py`**  
│   └ Validates the classical GAN + PINN model: performs parameter optimization, generates designs, evaluates spectral performance, and saves output images.  

└── **`models.py`**  
    └ Defines core models:
        - **IWAE**: encoder/decoder architecture with importance-weighted loss.
        - **BetaVAE**: reference variational autoencoder.
        - **Discriminator**: GAN discriminator with an auxiliary physics regression head.
        - **QuantumGenerator**: builds a parameterized variational circuit (e.g., PennyLane's Efficient-SU2) for quantum-assisted design generation.

---

##  Why This README Works

- **Clear Project Overview**: A brief, user-friendly introduction describing purpose, novelty, and high-level performance. :contentReference[oaicite:2]{index=2}  
- **Structured File Descriptions**: Users can easily grasp each script’s role and flow. Listing files under project sections fosters quick navigation. :contentReference[oaicite:3]{index=3}  
- **Readable Markdown**: Use of headings, bullet points, and inline code blocks improves clarity, while GitHub’s auto-generated table of contents boosts navigability. :contentReference[oaicite:4]{index=4}  
- **Engaging Tone**: A brief highlight quote and approachable language make the project inviting. :contentReference[oaicite:5]{index=5}

---

Let me know if you’d like to add badges (for CI, coverage), example usage code, installation steps, or links to contributing guidelines or license files.
::contentReference[oaicite:6]{index=6}
