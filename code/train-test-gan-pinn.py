#!/usr/bin/env python3
from __future__ import print_function
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import numpy as np
import random

# -----------------------------
# 0) USER CONFIGURATION
# -----------------------------
spectra_path = 'C:/.../fano_fit_results.csv'  
img_path = 'C:/.../Images/'   
save_dir = 'C:/.../PINN_GAN_SAVE/'   

# -----------------------------
# 0b) Basic checks and device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA is available: {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    try:
        print("CUDA Device Count: {}".format(torch.cuda.device_count()))
        print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))
    except Exception as e:
        print("Could not query CUDA device name:", e)

# -----------------------------
# 1) FIXED HYPERPARAMS
# -----------------------------
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

workers = 1
image_size  = 64       
nc          = 3        
latent_dim  = 16       
physics_dim = 4        
nz = physics_dim + latent_dim
batch_size  = 16
num_epochs  = 500
lr          = 1e-5
beta1       = 0.5
ngpu = torch.cuda.device_count()

# -----------------------------
# 2) DATASET & DATALOADER
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset = dset.ImageFolder(root=img_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=True)
num_batches = len(dataloader)
print("Number of batches per epoch:", num_batches)

# -----------------------------
# 3) LOAD PHYSICS PARAMETERS CSV
# -----------------------------
def Excel_Tensor(spectra_path):
    excelData = pd.read_csv(spectra_path, header=0, index_col=0)
    excelDataSpectra = excelData.iloc[:, :physics_dim]
    excelDataTensor = torch.tensor(excelDataSpectra.values, dtype=torch.float)
    return excelData, excelDataSpectra, excelDataTensor

excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)

def build_physics_for_dataset(dataset, excelData):
    csv_index = list(map(str, excelData.index.astype(str)))
    csv_index_set = set(csv_index)
    physics_list = []
    fallback = False
    for (path, cls) in dataset.imgs:
        base = os.path.basename(path)
        name_noext = os.path.splitext(base)[0]
        chosen_idx = None
        if name_noext in csv_index_set:
            chosen_idx = csv_index.index(name_noext)
        elif base in csv_index_set:
            chosen_idx = csv_index.index(base)
        else:
            fallback = True
            break
        physics_list.append(torch.tensor(excelData.iloc[chosen_idx, :physics_dim].values, dtype=torch.float))
    if fallback:
        print("Falling back to sequential mapping (requires images and CSV rows aligned).")
        physics_list = []
        n = len(dataset.imgs)
        for i in range(n):
            if i < len(excelDataTensor):
                physics_list.append(excelDataTensor[i])
            else:
                physics_list.append(torch.zeros(physics_dim, dtype=torch.float))
    return torch.stack(physics_list)

physics_for_dataset = build_physics_for_dataset(dataset, excelData)

# -----------------------------
# 4) GENERATOR
# -----------------------------
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = physics_dim + latent_dim
        self.fc = nn.Linear(self.input_dim, 512 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False)

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(-1, 512, 4, 4) 
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))  
        return x

# -----------------------------
# 5) DISCRIMINATOR
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.l1 = nn.Linear(physics_dim, image_size*image_size*nc, bias=False)
        self.conv1 = nn.Conv2d(2*nc, 64, 4, 2, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.bn4   = nn.BatchNorm2d(512)
        self.flatten = nn.Flatten()
        self.fc      = nn.Linear(512 * 4 * 4, 256)
        self.gan_output = nn.Linear(256, 1)
        self.physics_output = nn.Linear(256, physics_dim)

    def forward(self, img, label):
        x1 = img
        x2 = self.l1(label) 
        x2 = x2.view(-1, nc, image_size, image_size)
        x = torch.cat((x1, x2), 1) 
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc(x), 0.2)
        gan_logits = self.gan_output(x)      
        physics_pred = self.physics_output(x) 
        return gan_logits, physics_pred

def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif 'Linear' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# -----------------------------
# 7) LOSS FUNCTIONS & OPTIMIZERS
# -----------------------------
criterion_gan = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss() # ### FIX 2a: Needed for Discriminator training

# ### FIX 1: Dynamic Physics Loss that checks the PREDICTION against the TARGET CONDITIONS
def physics_constraint_loss(pred_params, condition_labels, lambda_A=26, lambda_w0=14, lambda_Q=26):
    # 1. Un-normalize PREDICTED params
    A0_pred = pred_params[:, 0:1]  
    omega_0_pred = pred_params[:, 1:2] * 56.25 + 18.75 
    Gamma_pred = pred_params[:, 2:3] * 6.387283913 - 0.034279125
    q_pred = pred_params[:, 3:4] * 215.33279325 - 41.84328885

    # 2. Un-normalize TARGET condition labels (What we asked the generator to make)
    A_target = condition_labels[:, 0:1]
    w0_target = condition_labels[:, 1:2] * 56.25 + 18.75
    Gamma_target = condition_labels[:, 2:3] * 6.387283913 - 0.034279125
    
    Q_min_target = w0_target / (torch.abs(Gamma_target) + 1e-6)

    # 3. Fano Math Evaluated at the Target Frequency
    delta = 2.0 * (w0_target - omega_0_pred) / (Gamma_pred + 1e-6)
    A = A0_pred * ((q_pred + delta)**2 / (1.0 + delta**2))
    A = torch.clamp(A, max=1.0)

    Q_val = omega_0_pred / (torch.abs(Gamma_pred) + 1e-6)
    Q_penalty = torch.relu(Q_min_target - Q_val) ** 2

    # 4. Component Losses
    loss_A = torch.clamp(A_target - A, min=0.0)**2
    loss_w0 = ((omega_0_pred - w0_target) / 50.0)**2

    loss = (lambda_A * loss_A) + (lambda_w0 * loss_w0) + (lambda_Q * Q_penalty)
    return loss.mean()

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

def set_requires_grad(net, requires_grad=False):
    for p in net.parameters():
        p.requires_grad = requires_grad

os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# 8) TRAINING LOOP
# -----------------------------
img_list = []
G_losses = []
D_losses = []
iters = 0
lambda_physics = 1e-3 # Note: You may need to tune this to balance GAN and Phys loss

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        b_size = real_imgs.size(0)
        start_idx = i * batch_size
        end_idx = start_idx + b_size
        
        # Target conditions for this batch
        physics_batch = physics_for_dataset[start_idx:end_idx].to(device) 
        real_imgs = real_imgs.to(device)

        label_real = torch.empty(b_size, 1, device=device).uniform_(0.9, 1.0)
        label_fake = torch.empty(b_size, 1, device=device).uniform_(0.0, 0.1)

        # ---------------------
        # Train Discriminator
        # ---------------------
        set_requires_grad(netD, True)
        netD.zero_grad()
        
        # Real images pass
        gan_out_real_logits, pred_phys_D_real_raw = netD(real_imgs, physics_batch)
        loss_real = criterion_gan(gan_out_real_logits, label_real)

        # ### FIX 2b: Train Discriminator to accurately predict real physics using MSE
        # The discriminator must learn the correct physics from REAL data, not the Fano equation.
        pred_phys_D_real = torch.sigmoid(pred_phys_D_real_raw)
        loss_physics_D = mse_loss(pred_phys_D_real, physics_batch)

        # Fake images pass
        latent_noise = torch.rand(b_size, latent_dim, device=device)
        gen_input = torch.cat((physics_batch, latent_noise), dim=1)
        fake_imgs = netG(gen_input)

        gan_out_fake_logits, _ = netD(fake_imgs.detach(), physics_batch)
        loss_fake = criterion_gan(gan_out_fake_logits, label_fake)

        loss_D_gan = loss_real + loss_fake
        loss_D = loss_D_gan + (lambda_physics * loss_physics_D)
        loss_D.backward()
        optimizerD.step()

        # ---------------------
        # Train Generator
        # ---------------------
        set_requires_grad(netD, False) 
        netG.zero_grad()

        latent_noise2 = torch.rand(b_size, latent_dim, device=device)
        gen_input2 = torch.cat((physics_batch, latent_noise2), dim=1)
        fake_imgs_for_G = netG(gen_input2)
        
        gan_out_fake_logits_forG, pred_phys_G_raw = netD(fake_imgs_for_G, physics_batch)
        loss_G_gan = criterion_gan(gan_out_fake_logits_forG, label_real)

        # ### FIX 3: Pass the specific targets (physics_batch) to the physics loss
        pred_phys_G = torch.sigmoid(pred_phys_G_raw)
        loss_physics_G = physics_constraint_loss(pred_phys_G, physics_batch)

        loss_G = loss_G_gan + (lambda_physics * loss_physics_G)
        loss_G.backward()
        optimizerG.step()

        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}][Batch {i}/{num_batches}] "
                  f"D(GAN): {loss_D.item():.4f} | G(GAN): {loss_G.item():.4f}")

        iters += 1

# -----------------------------
# 9) SAVE MODELS
# -----------------------------
torch.save(netG.state_dict(), os.path.join(save_dir, 'netG.pth'))
torch.save(netD.state_dict(), os.path.join(save_dir, 'netD.pth'))
print("Training Complete!")

# -----------------------------
# 10) CONDITIONAL TESTING BLOCK
# -----------------------------
# ### FIX 4: Prove the generator can hit a specific target.
# Set a custom target (e.g., A0=0.95, w0=0.5, Gamma=0.01, q=0.5)
target_fano_params = torch.tensor([[0.95, 0.5, 0.01, 0.5]], device=device)
target_fano_batch = target_fano_params.repeat(64, 1)

# Generate 64 random latent variations for this exact same target
random_latent_noise = torch.rand(64, latent_dim, device=device)
custom_fixed_noise = torch.cat((target_fano_batch, random_latent_noise), dim=1)

with torch.no_grad():
    viz_fake_conditional = netG(custom_fixed_noise).detach().cpu()

# Plot Losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator Loss")
plt.plot(D_losses,label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot Generated Conditional Images
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Metasurfaces (Conditional Target: High-Q)")
plt.imshow(np.transpose(vutils.make_grid(viz_fake_conditional, padding=2, normalize=True), (1,2,0)))
plt.show()
