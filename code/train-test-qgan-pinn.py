#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import random
import time
from models.models import IWAE, Discriminator, QuantumGenerator

# Get GPU Information (guarded)
print("CUDA is available: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("CUDA Device Count: {}".format(torch.cuda.device_count()))
    try:
        print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))
    except Exception as e:
        print("Could not get CUDA device name:", e)

# Physics-informed loss (Fano-like lineshape)
def physics_constraint_loss(pred_params, A_target=0.9, w0_target=50.0, Q_min=1e5,
                           lambda_A = 26, lambda_w0 = 14, lambda_Q = 26):
    """
    Physics-based loss for predicted parameters.
    Expects pred_params of shape (N, 4) in normalized form and rescales them
    to physical ranges (same scaling used in original code).
    pred_params columns: [A0 (0..1), w0_norm (0..1), Gamma_norm (0..1), q_norm (0..1)]
    Returns mean loss across batch.
    """

    # Rescale predicted params (same rescaling formula used in original code)
    A0 = pred_params[:, 0:1]  # [N,1], expected in [0,1]
    omega_0 = pred_params[:, 1:2] * 56.25 + 18.75   # maps [0,1] -> [18.75, 75]
    Gamma = pred_params[:, 2:3] * 6.387283913 + (-0.034279125)  # maps approx
    q = pred_params[:, 3:4] * 215.33279325 + (-41.84328885)

    # Use a standard Fano-like epsilon: epsilon = 2*(omega - omega0)/Gamma
    # Evaluate A at the target frequency w0_target (i.e., A(w_target) should be A_target)
    eps = 2.0 * (w0_target - omega_0) / (Gamma + 1e-8)
    A_pred = A0 * ((q + eps) ** 2 / (1.0 + eps ** 2))

    # Limit A to a maximum of 1.0 (physical)
    A_pred = torch.clamp(A_pred, max=1.0)

    # Quality factor penalty: Q = omega0 / Gamma
    Q_val = omega_0 / (torch.abs(Gamma) + 1e-8)
    Q_penalty = torch.relu(Q_min - Q_val) ** 2

    # Loss components
    loss_A = torch.clamp(A_target - A_pred, min=0.0) ** 2  # penalize if lower than target
    loss_w0 = ((omega_0 - w0_target) / 50.0) ** 2

    loss = (lambda_A * loss_A + lambda_w0 * loss_w0 + lambda_Q * Q_penalty).mean()
    return loss

# Load the pretrained QVAE and train the QGAN

def train_qgan(generator, discriminator, iwae, num_samples, excelDataTensor,
               dataloader, optimizerG, optimizerD, criterion, num_epochs,
               device, lambda_physics, label_dims, latent):
    generator.train()
    discriminator.train()
    iwae.eval()  # Pretrained IWAE in eval mode
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for batch_idx, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # Create smoothed real label per batch (one-sided smoothing)
            real_label_val = random.uniform(0.9, 1.0)
            # N = b_size * num_samples
            N = b_size * num_samples
            real_labels = torch.full((N,), real_label_val, device=device, dtype=torch.float)
            fake_labels = torch.full((N,), 0.0, device=device, dtype=torch.float)

            # Obtain latent_real from pretrained IWAE (ensure shape matches expectation)
            with torch.no_grad():
                recon_real, mu, logvar, latent_real = iwae(real_cpu, num_samples)
                # latent_real should be of shape [b_size * num_samples, latent]
                # If IWAE returns [num_samples, b_size, latent], we will reshape below.

            # Build label-conditioned inputs from excelDataTensor for this batch
            noise_label_list = []
            noise_joint_list = []

            # We will try to fetch b_size samples * num_samples from excelDataTensor
            base_idx = batch_idx * b_size
            for j in range(b_size):
                excelIndex = base_idx + j
                if excelIndex >= len(excelDataTensor):
                    # Out-of-range; break (or wrap if you prefer)
                    break
                gotdata = excelDataTensor[excelIndex]  # shape: (label_dims,)
                for _ in range(num_samples):
                    noise_label_list.append(gotdata)
                    rand_lat = torch.rand(latent, device=device)
                    joint = torch.cat((gotdata.to(device), rand_lat), dim=0)  # shape: (label_dims + latent,)
                    noise_joint_list.append(joint)

            if len(noise_joint_list) == 0:
                continue  # skip this batch if we couldn't construct conditioning

            noise = torch.stack(noise_joint_list, dim=0).to(device)   # shape: [N, nz]
            noise2 = torch.stack(noise_label_list, dim=0).to(device)  # shape: [N, label_dims]

            # Ensure latent_real shape matches expectation: try common reshapes
            try:
                if latent_real.dim() == 3 and latent_real.size(0) == num_samples:
                    # shape likely [num_samples, b_size, latent] -> reorder to [b_size * num_samples, latent]
                    latent_real = latent_real.permute(1, 0, 2).reshape(-1, latent)
                elif latent_real.dim() == 2 and latent_real.size(0) == b_size and num_samples == 1:
                    # already [b_size, latent]
                    pass
                elif latent_real.dim() == 2 and latent_real.size(0) == N:
                    pass
                else:
                    # fallback: reshape if possible, otherwise trim/pad
                    latent_real = latent_real.reshape(-1, latent)[:noise.size(0), :latent]
            except Exception:
                latent_real = latent_real.reshape(-1, latent)[:noise.size(0), :latent]

            latent_real = latent_real.to(device)

            # ---- Discriminator update ----
            discriminator.zero_grad()

            # Real examples (latent_real conditioned on actual labels)
            output_real, _ = discriminator(latent_real, noise2)
            # make shapes compatible (flatten logits to 1D and labels to 1D)
            output_real_1d = output_real.view(-1)
            real_labels_1d = real_labels.to(device=device, dtype=output_real_1d.dtype).view(-1)
            errD_real = criterion(output_real_1d, real_labels_1d)

            # Fake examples created by generator
            fake_latent = generator(noise)  # expected shape: [N, latent]
            output_fake, _ = discriminator(fake_latent.detach(), noise2)
            output_fake_1d = output_fake.view(-1)
            fake_labels_1d = fake_labels.to(device=device, dtype=output_fake_1d.dtype).view(-1)
            errD_fake = criterion(output_fake_1d, fake_labels_1d)

            errD_gan = errD_real + errD_fake

            # Physics penalty for discriminator (use some random conditioning for physics probe)
            physics_batch = max(2 * b_size, 1)
            noise3 = torch.rand(physics_batch, label_dims + latent, device=device)
            noise4 = torch.rand(physics_batch, label_dims).to(device)

            gen_phys_real = generator(noise3)  # latent-like tensors
            _, pred_phys = discriminator(gen_phys_real.detach(), noise4)
            physics_loss_D = physics_constraint_loss(pred_phys)

            errD = errD_gan + lambda_physics * physics_loss_D
            errD.backward()
            optimizerD.step()

            # ---- Generator update ----
            generator.zero_grad()
            output_for_G, _ = discriminator(fake_latent, noise2)
            output_for_G_1d = output_for_G.view(-1)
            real_labels_for_G = real_labels.to(device=device, dtype=output_for_G_1d.dtype).view(-1)
            errG_gan = criterion(output_for_G_1d, real_labels_for_G)

            # Physics penalty for generator (use fresh physics samples)
            gen_phys_fake = generator(noise3)  # shape [physics_batch, latent]
            _, pred_phys_fake = discriminator(gen_phys_fake, noise4)
            physics_loss_G = physics_constraint_loss(pred_phys_fake)

            errG = errG_gan + lambda_physics * physics_loss_G
            errG.backward()
            optimizerG.step()

            # Append losses for monitoring (global lists expected)
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}][Batch {batch_idx}/{num_batches}] "
                  f"D(GAN): {errD.item():.4f} | G(GAN): {errG.item():.4f}")

    print("QGAN Training Completed with Pretrained IWAE.")

# Testing function to check the generator's output using testTensor
def test_generator(generator, iwae, testTensor, device, img_list):
    generator.eval()
    iwae.eval()

    with torch.no_grad():
        fake_latent = generator(testTensor.to(device))
        # Use the iwae.decoder to generate images from latent
        fake_images = iwae.decoder(fake_latent)
        fake = fake_images.detach().cpu()

    img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))
    return img_list

def Excel_Tensor(spectra_path):
    # Location of excel data
    excelData = pd.read_csv(spectra_path, header=0, index_col=0)
    excelDataSpectra = excelData.iloc[:, :4]  # first 4 columns used as conditioning labels
    excelDataTensor = torch.tensor(excelDataSpectra.values).type(torch.FloatTensor)
    return excelData, excelDataSpectra, excelDataTensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# Main script
if __name__ == '__main__':
    # Parameters
    n_generators = 1
    n_qubits = 9
    q_depth = 2
    nc = 3
    image_size = 64
    batch_size = 16
    num_samples = 5
    num_epochs = 500
    workers = 1 #Number of workers for dataloader (on Windows set to 0)
    lrG = 1e-5  # generator LR
    lrD = 1e-4  # discriminator LR
    lambda_physics = 1e-12
    label_dims = 4
    img_list = []
    G_losses = []
    D_losses = []
    latent = 5
    nz = label_dims + latent
    beta = 2.0
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    beta2 = 0.999

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    spectra_path = 'C:/.../fano_fit_results.csv'  # CSV file with physics parameters ""
    img_path = 'C:/.../Images/'   # Root directory for images
    save_dir = 'C:/.../PINN_GAN_SAVE/'   # Save models after training
    pretrained_iwae_path = 'C:/.../pretrained_iwae.pth'      # Pretrained IWAE weights (optional but recommended)

    excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)

    dataset = dset.ImageFolder(root=img_path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor()
                               ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    num_batches = len(dataloader)
    print("No. of batches per epoch:", num_batches)

    # Plot some training images (optional; guard for empty dataloader)
    try:
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
    except StopIteration:
        print("Dataloader is empty. Check img_path and dataset.")

    # Initialize Quantum Generator and Discriminator
    generator = QuantumGenerator(n_qubits, q_depth, n_generators).to(device)
    discriminator = Discriminator(n_qubits, label_dims).to(device)

    # Apply the weights_init function to randomly initialize all weights
    discriminator.apply(weights_init)

    # Load the pretrained QVAE from a specific file (safe map_location)
    iwae = IWAE(n_qubits=n_qubits, nc=nc, beta=beta, num_samples=num_samples).to(device)
    try:
        iwae.load_state_dict(torch.load(pretrained_iwae_path, map_location=device))
        print("Loaded pretrained_iwae.pth")
    except Exception as e:
        print("Could not load pretrained_iwae.pth:", e)
        # Optionally raise here if IWAE is required:
        # raise

    # Define optimizers
    optimizerG = optim.Adam(generator.parameters(), lr=lrG, betas=(beta1, beta2))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lrD, betas=(beta1, beta2))
    criterion = nn.BCELoss(reduction='mean')  # or BCEWithLogitsLoss depending on discriminator output

    start_time = time.time()
    local_time = time.ctime(start_time)
    print('Start Time = %s' % local_time)

    # Train the QGAN
    train_qgan(generator, discriminator, iwae, num_samples, excelDataTensor,
               dataloader, optimizerG, optimizerD, criterion, num_epochs,
               device, lambda_physics, label_dims, latent)

    local_time = time.ctime(time.time())
    print('End Time = %s' % local_time)
    run_time = (time.time() - start_time) / 3600
    print('Total Time Lapsed = %s Hours' % run_time)

    # Save the final models
    torch.save(generator.state_dict(), 'final_generator_qgan_iwae.pth')
    torch.save(discriminator.state_dict(), 'final_discriminator_qgan_iwae.pth')

    # Test the generator using testTensor
    try:
        generator.load_state_dict(torch.load('final_generator_qgan_iwae.pth', map_location=device))
    except Exception as e:
        print("Could not load final generator state dict:", e)

    fixed_noise = torch.rand(64, nz, device=device)

    # Call the test function
    img_list = test_generator(generator, iwae, fixed_noise, device, img_list)

    # Plot and save G and D Training Losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses-qgan-iwae.png')
    plt.show()

    # Plot the real images and the fake images
    try:
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
        plt.savefig('fake-and-real-iwae.png')
        plt.show()
    except Exception as e:
        print("Could not plot images:", e)
