#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import pandas as pd
import random
import time
from PIL import Image

# Assuming your custom models are in a file named models.py
from models import IWAE, Discriminator, QuantumGenerator

# Get GPU Information
print("CUDA is available: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("CUDA Device Count: {}".format(torch.cuda.device_count()))
    try:
        print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))
    except Exception as e:
        print("Could not get CUDA device name:", e)

# ==========================================
# ### FIX 1: Custom Dataset to prevent misalignment ###
# ==========================================
class MetasurfaceDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, max_samples=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # ### NEW: Allows you to artificially shrink dataset to 64, 500, or 3000 
        # samples for fair benchmarking (Reviewer 1 & 2)
        if max_samples is not None:
            self.data_frame = self.data_frame.iloc[:max_samples]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 1. Get name and strip accidental spaces
        base_name = str(self.data_frame.iloc[idx, 0]).strip()
        
        # 2. Add extension if missing
        if not base_name.endswith(('.png', '.jpg', '.jpeg')):
            base_name += '.png'
            
        # 3. Join path
        img_name = os.path.join(self.img_dir, base_name)
        
        # 4. SAFETY CHECK: Tell us exactly what is missing and stop cleanly
        if not os.path.exists(img_name):
            raise FileNotFoundError(f"\nCRITICAL ERROR: Python cannot find this image:\n{img_name}\n"
                                    f"Please check if it is spelled correctly in the CSV and actually exists in the folder.")
        
        image = Image.open(img_name).convert('RGB')
        labels = self.data_frame.iloc[idx, 1:5].values.astype('float32')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels)

# ==========================================
# ### FIX 2: Dynamic Physics Loss Function ###
# ==========================================
def physics_constraint_loss(pred_params, condition_labels, 
                            lambda_A=26, lambda_w0=14, lambda_Q=26):
    """
    Penalizes the generator if its physics outputs don't match the specific 
    conditions it was asked to generate.
    """
    # 1. Un-normalize PREDICTED params
    A0_pred = pred_params[:, 0:1]  
    omega_0_pred = pred_params[:, 1:2] * 56.25 + 18.75
    Gamma_pred = pred_params[:, 2:3] * 6.387283913 + (-0.034279125)
    q_pred = pred_params[:, 3:4] * 215.33279325 + (-41.84328885)

    # 2. Un-normalize TARGET condition labels
    A_target = condition_labels[:, 0:1]
    w0_target = condition_labels[:, 1:2] * 56.25 + 18.75
    Gamma_target = condition_labels[:, 2:3] * 6.387283913 + (-0.034279125)
    
    # Dynamically calculate the target Q-factor
    Q_min_target = w0_target / (torch.abs(Gamma_target) + 1e-8)

    # 3. Calculate Fano Absorption of the PREDICTION at the TARGET frequency
    eps = 2.0 * (w0_target - omega_0_pred) / (Gamma_pred + 1e-8)
    A_pred_at_target = A0_pred * ((q_pred + eps) ** 2 / (1.0 + eps ** 2))
    A_pred_at_target = torch.clamp(A_pred_at_target, max=1.0)

    # 4. Calculate predicted Q-factor
    Q_val = omega_0_pred / (torch.abs(Gamma_pred) + 1e-8)
    
    # 5. Component Losses
    loss_A = torch.clamp(A_target - A_pred_at_target, min=0.0) ** 2 
    loss_w0 = ((omega_0_pred - w0_target) / 50.0) ** 2
    Q_penalty = torch.relu(Q_min_target - Q_val) ** 2

    # Total weighted loss
    loss = (lambda_A * loss_A + lambda_w0 * loss_w0 + lambda_Q * Q_penalty).mean()
    return loss

# ==========================================
# ### FIX 3: Training Loop Logic ###
# ==========================================
def train_qgan(generator, discriminator, iwae, num_samples, dataloader, 
               optimizerG, optimizerD, criterion, num_epochs, device, 
               lambda_physics, label_dims, latent):
    generator.train()
    discriminator.train()
    iwae.eval() 
    mse_loss = nn.MSELoss() # Used for discriminator physics training
    
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for batch_idx, (real_cpu, real_conditions) in enumerate(dataloader):
            real_cpu = real_cpu.to(device)
            real_conditions = real_conditions.to(device)
            b_size = real_cpu.size(0)

            # Smooth real labels (e.g., 0.95 instead of 1.0)
            real_label_val = random.uniform(0.9, 1.0)
            N = b_size * num_samples
            real_labels = torch.full((N,), real_label_val, device=device, dtype=torch.float)
            fake_labels = torch.full((N,), 0.0, device=device, dtype=torch.float)

            # Obtain latent_real from pretrained IWAE
            with torch.no_grad():
                recon_real, mu, logvar, latent_real = iwae(real_cpu, num_samples)
                
                # Reshape to [N, latent] if necessary
                if latent_real.dim() == 3:
                    latent_real = latent_real.permute(1, 0, 2).reshape(-1, latent)

            # Construct conditional inputs for the Generator
            # Repeat the real conditions 'num_samples' times to match latent_real size
            noise_labels = real_conditions.repeat_interleave(num_samples, dim=0)
            
            # Random latent noise
            random_latents = torch.rand(N, latent, device=device)
            
            # Concatenate labels and random noise
            noise_joint = torch.cat((noise_labels, random_latents), dim=1)

            # ---- 1. Train Discriminator ----
            discriminator.zero_grad()

            # Real Pass
            output_real, pred_phys_real = discriminator(latent_real, noise_labels)
            errD_real = criterion(output_real.view(-1), real_labels)

            # Physics Loss (Discriminator): Learn to predict real Fano params accurately
            physics_loss_D = mse_loss(pred_phys_real, noise_labels)

            # Fake Pass
            fake_latent = generator(noise_joint) 
            output_fake, _ = discriminator(fake_latent.detach(), noise_labels)
            errD_fake = criterion(output_fake.view(-1), fake_labels)

            errD = errD_real + errD_fake + (lambda_physics * physics_loss_D)
            errD.backward()
            optimizerD.step()

            # ---- 2. Train Generator ----
            generator.zero_grad()
            
            output_for_G, pred_phys_fake = discriminator(fake_latent, noise_labels)
            errG_gan = criterion(output_for_G.view(-1), real_labels)

            # Physics Loss (Generator): Force generated physics to match target conditions
            physics_loss_G = physics_constraint_loss(pred_phys_fake, noise_labels)

            errG = errG_gan + (lambda_physics * physics_loss_G)
            errG.backward()
            optimizerG.step()

            # Tracking
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if batch_idx % 50 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}][Batch {batch_idx}/{len(dataloader)}] "
                      f"D Loss: {errD.item():.4f} | G Loss: {errG.item():.4f} "
                      f"| G Phys: {physics_loss_G.item():.4f}")

    print("QGAN Training Completed.")

# Testing function
def test_generator(generator, iwae, testTensor, device, img_list):
    generator.eval()
    iwae.eval()
    with torch.no_grad():
        fake_latent = generator(testTensor.to(device))
        fake_images = iwae.decoder(fake_latent)
        fake = fake_images.detach().cpu()
    img_list.append(vutils.make_grid(fake, nrow=8, padding=2, normalize=True))
    return img_list

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == '__main__':
    # Hyperparameters
    n_generators = 1
    n_qubits = 9
    q_depth = 2
    nc = 3
    image_size = 64
    batch_size = 16
    num_samples = 5
    num_epochs = 500
    workers = 1 
    lrG = 1e-5
    lrD = 1e-4
    lambda_physics = 1e-3  # You may need to tune this to balance GAN and Phys loss
    label_dims = 4
    latent = 5
    nz = label_dims + latent
    beta1 = 0.5
    beta2 = 0.999
    
    img_list = []
    G_losses = []
    D_losses = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Paths
    spectra_path = 'C:/.../fano_fit_results.csv'  
    img_path = 'C:/.../Images/'   
    pretrained_iwae_path = 'C:/.../pretrained_iwae.pth'      

    # Dataset & Dataloader
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # NOTE: To do a fair benchmark (Reviewer 2), pass max_samples=64 or 3000 here
    dataset = MetasurfaceDataset(csv_file=spectra_path, img_dir=img_path, transform=transform, max_samples=None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    print("Total training samples:", len(dataset))
    print("No. of batches per epoch:", len(dataloader))

    # Initialize Models
    generator = QuantumGenerator(n_qubits, q_depth, n_generators).to(device)
    discriminator = Discriminator(n_qubits, label_dims).to(device)
    discriminator.apply(weights_init)

    iwae = IWAE(n_qubits=n_qubits, nc=nc, beta=2.0, num_samples=num_samples).to(device)
    try:
        iwae.load_state_dict(torch.load(pretrained_iwae_path, map_location=device))
        print("Loaded pretrained IWAE successfully.")
    except Exception as e:
        print("Could not load pretrained IWAE:", e)

    # Optimizers
    optimizerG = optim.Adam(generator.parameters(), lr=lrG, betas=(beta1, beta2))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lrD, betas=(beta1, beta2))
    criterion = nn.BCELoss(reduction='mean') 

    # TRAIN
    start_time = time.time()
    print('Start Time = %s' % time.ctime(start_time))

    train_qgan(generator, discriminator, iwae, num_samples, dataloader, 
               optimizerG, optimizerD, criterion, num_epochs, device, 
               lambda_physics, label_dims, latent)

    print('Total Time Lapsed = %.2f Hours' % ((time.time() - start_time) / 3600))

    # Save Models
    torch.save(generator.state_dict(), 'final_generator_qgan_iwae.pth')
    torch.save(discriminator.state_dict(), 'final_discriminator_qgan_iwae.pth')

    # ==========================================
    # ### FIX 4: True Conditional Testing ###
    # ==========================================
    # We test the model by asking it to generate a SPECIFIC High-Q target.
    # Example normalized targets: A0=0.95, w0=0.5 (approx 46 THz), Gamma=0.01 (Very narrow), q=0.5
    target_fano_params = torch.tensor([[0.95, 0.5, 0.01, 0.5]], device=device)
    
    # Create 64 variations of this target
    target_fano_batch = target_fano_params.repeat(64, 1)
    
    # Combine with random latent noise to explore different geometries for the same target
    random_latent_noise = torch.rand(64, latent, device=device)
    custom_fixed_noise = torch.cat((target_fano_batch, random_latent_noise), dim=1)

    img_list = test_generator(generator, iwae, custom_fixed_noise, device, img_list)

    # Plot Losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses-qgan-iwae.png')
    plt.show()

    # Plot Fake Images
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated High-Q Metasurfaces (Conditional Target)")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig('conditional_fake_images.png')
    plt.show()
