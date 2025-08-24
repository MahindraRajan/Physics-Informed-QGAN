#!/usr/bin/env python3
from __future__ import print_function
import os
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
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
# Update these paths before running
spectra_path = 'C:/.../absorptionData_HybridGAN.csv'  # CSV file with physics parameters (index should map to images)
img_path = 'C:/.../Images'   # Root directory for images used by ImageFolder
save_dir = 'C:/.../PINN_GAN_SAVE/'   # Save models after training

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
#Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#Number of workers for dataloader (on Windows set to 0)
workers = 1
image_size  = 64       # Image dimensions: 64x64
nc          = 3        # Number of channels (RGB)
latent_dim  = 16       # Dimension of the noise part of the input
physics_dim = 4        # Use 4 values from the CSV (physics parameters)
nz = physics_dim + latent_dim
batch_size  = 16
num_epochs  = 5000
lr          = 1e-5
beta1       = 0.5
ngpu = torch.cuda.device_count()

# -----------------------------
# 2) DATASET & DATALOADER
# -----------------------------
# Using ImageFolder to load images (dataset.imgs holds filenames)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset = dset.ImageFolder(root=img_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
num_batches = len(dataloader)
print("Number of batches per epoch:", num_batches)

# -----------------------------
# 3) LOAD PHYSICS PARAMETERS CSV
# -----------------------------
def Excel_Tensor(spectra_path):
    # Load the CSV file (with header and using the first column as index)
    excelData = pd.read_csv(spectra_path, header=0, index_col=0)
    # Extract the first 4 columns as the physics parameters (of the absorption spectrum)
    excelDataSpectra = excelData.iloc[:, :physics_dim]
    # Convert to a tensor of shape [num_samples, physics_dim]
    excelDataTensor = torch.tensor(excelDataSpectra.values, dtype=torch.float)
    return excelData, excelDataSpectra, excelDataTensor

excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)
print("CSV rows:", len(excelDataTensor))

# Build a physics tensor aligned with dataset order.
# We try to match by filename (without extension) to CSV index; if matching fails we fall back to sequential mapping.
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
        # As fallback, map rows sequentially in dataset order.
        print("Could not match filenames to CSV index exactly. Falling back to sequential mapping (requires images and CSV rows aligned).")
        physics_list = []
        n = len(dataset.imgs)
        for i in range(n):
            if i < len(excelDataTensor):
                physics_list.append(excelDataTensor[i])
            else:
                # pad with zeros if CSV shorter than images
                physics_list.append(torch.zeros(physics_dim, dtype=torch.float))
    return torch.stack(physics_list)

physics_for_dataset = build_physics_for_dataset(dataset, excelData)
print("Physics-for-dataset shape:", physics_for_dataset.shape)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# -----------------------------
# 4) GENERATOR
# -----------------------------
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = physics_dim + latent_dim

        # Fully connected layer to project the input into a 4x4 feature map with 512 channels
        self.fc = nn.Linear(self.input_dim, 512 * 4 * 4)

        # Upsampling with ConvTranspose2d layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn1     = nn.BatchNorm2d(256)

        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)
        self.bn2     = nn.BatchNorm2d(128)

        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)
        self.bn3     = nn.BatchNorm2d(64)

        self.deconv4 = nn.ConvTranspose2d(64, nc, 4, 2, 1, bias=False)

    def forward(self, noise):
        x = self.fc(noise)
        x = x.view(-1, 512, 4, 4)  # Reshape to [batch_size, 512, 4, 4]
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))  # Output image in [-1, 1]
        return x

# -----------------------------
# 5) DISCRIMINATOR
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.l1 = nn.Linear(physics_dim, image_size*image_size*nc, bias=False)

        # Shared convolutional feature extractor
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

        # GAN head: produce raw logits (use BCEWithLogitsLoss)
        self.gan_output = nn.Linear(256, 1)

        # Physics head for predicting the 4-dimensional physics parameters (raw)
        self.physics_output = nn.Linear(256, physics_dim)

    def forward(self, img, label):
        # label: [batch, physics_dim]
        x1 = img
        x2 = self.l1(label)  # maps physics vector to an image-shaped tensor
        x2 = x2.view(-1, nc, image_size, image_size)
        x = torch.cat((x1, x2), 1)  # concat on channel dim -> 2*nc channels
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc(x), 0.2)
        gan_logits = self.gan_output(x)      # raw logits
        physics_pred = self.physics_output(x) # raw physics predictions (not squashed)
        return gan_logits, physics_pred

# -----------------------------
# 6) Initialize networks
# -----------------------------
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

if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

netG.apply(weights_init)
netD.apply(weights_init)

print(netG)
print(netD)

# -----------------------------
# 7) LOSS FUNCTIONS & OPTIMIZERS
# -----------------------------
criterion_gan = nn.BCEWithLogitsLoss()  # more stable than Sigmoid + BCELoss

# Physics-informed loss (unchanged mathematical form but expects inputs in [0,1])
def physics_constraint_loss(pred_params, A_target=0.9, w0_target=50.0, Q_min=1e5,
                            lambda_A=26, lambda_w0=14, lambda_Q=26):
    """
    Physics-based loss evaluated at ω = ω₀.
    Assumes pred_params are normalized in [0,1] and need rescaling back to THz.
    A0 is the first parameter (0–1), followed by w0, Gamma, q.
    """
    # Ensure shape [batch,4]
    A0      = pred_params[:, 0:1]           # Target A(w0)
    omega_0 = pred_params[:, 1:2] * 56.25 + 18.75   # [18.75, 75] THz
    Gamma   = pred_params[:, 2:3] * 6.387283913 + (-0.034279125)  # [~–0.03, 6.35]
    q       = pred_params[:, 3:4] * 215.33279325 + (-41.84328885)  # ~[-41.84, 173.49]

    delta = torch.zeros_like(omega_0)
    A = A0 * ((q + delta)**2 / (1 + delta**2))

    # Limit A to a maximum of 1.0
    A = torch.min(A, torch.ones_like(A))

    Q_val = omega_0 / (Gamma + 1e-6)
    Q_penalty = torch.relu(Q_min - Q_val) ** 2

    loss_A = torch.clamp(A_target - A, min=0.0)**2
    loss_w0 = ((omega_0 - w0_target) / 50)**2
    loss_1 = lambda_A * loss_A + lambda_w0 * loss_w0
    loss_2 = lambda_Q * Q_penalty
    loss = loss_1 + loss_2
    return loss.mean()

lambda_physics = 2.0            # Weight for the physics loss term

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

# Create fixed noise for visualizing progression (use first physics rows if available)
vis_batch = min(64, len(physics_for_dataset))
fixed_phys = physics_for_dataset[:vis_batch]
fixed_latent = torch.rand(vis_batch, latent_dim)
fixed_noise = torch.cat((fixed_phys, fixed_latent), dim=1).to(device)

# Utility
def set_requires_grad(net, requires_grad=False):
    for p in net.parameters():
        p.requires_grad = requires_grad

# Make sure save dir exists
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# 8) TRAINING LOOP
# -----------------------------
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i, (real_imgs, _) in enumerate(dataloader):
        b_size = real_imgs.size(0)
        start_idx = i * batch_size
        end_idx = start_idx + b_size
        # Get the corresponding physics conditioning for this batch
        physics_batch = physics_for_dataset[start_idx:end_idx].to(device)  # shape [b_size, physics_dim]

        real_imgs = real_imgs.to(device)

        # Labels
        label_real = torch.empty(b_size, 1, device=device).uniform_(0.9, 1.0)
        label_fake = torch.empty(b_size, 1, device=device).uniform_(0.0, 0.1)

        # ---------------------
        # Train Discriminator
        # ---------------------
        set_requires_grad(netD, True)
        netD.zero_grad()
        # Real images
        gan_out_real_logits, _ = netD(real_imgs, physics_batch)
        loss_real = criterion_gan(gan_out_real_logits, label_real)

        # Fake images produced conditioned on the same physics_batch
        latent_noise = torch.rand(b_size, latent_dim, device=device)
        gen_input = torch.cat((physics_batch, latent_noise), dim=1)
        fake_imgs = netG(gen_input)

        gan_out_fake_logits, _ = netD(fake_imgs.detach(), physics_batch)
        loss_fake = criterion_gan(gan_out_fake_logits, label_fake)

        # Physics-informed loss on discriminator predictions for generated data
        # Use a random physics vector for the D's physics-loss evaluation (or could reuse physics_batch)
        random_phys = torch.rand(b_size, physics_dim, device=device)
        gen_for_phys = netG(torch.cat((random_phys, torch.rand(b_size, latent_dim, device=device)), dim=1))
        _, pred_phys_D_raw = netD(gen_for_phys.detach(), random_phys)
        pred_phys_D = torch.sigmoid(pred_phys_D_raw)  # squash to [0,1]
        loss_physics_D = physics_constraint_loss(pred_phys_D)

        loss_D_gan = loss_real + loss_fake
        loss_D = loss_D_gan + lambda_physics * loss_physics_D
        loss_D.backward()
        optimizerD.step()

        # ---------------------
        # Train Generator
        # ---------------------
        set_requires_grad(netD, False)  # freeze D params when updating G
        netG.zero_grad()

        # GAN loss for generator: want D(fake) -> real
        latent_noise2 = torch.rand(b_size, latent_dim, device=device)
        gen_input2 = torch.cat((physics_batch, latent_noise2), dim=1)
        fake_imgs_for_G = netG(gen_input2)
        gan_out_fake_logits_forG, pred_phys_G_raw = netD(fake_imgs_for_G, physics_batch)
        loss_G_gan = criterion_gan(gan_out_fake_logits_forG, label_real)

        # Physics loss for generator: we want predicted physics (by D) to be physically meaningful.
        pred_phys_G = torch.sigmoid(pred_phys_G_raw)
        loss_physics_G = physics_constraint_loss(pred_phys_G)

        loss_G = loss_G_gan + lambda_physics * loss_physics_G
        loss_G.backward()
        optimizerG.step()

        # Save losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == num_batches-1)):
            with torch.no_grad():
                viz_fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(viz_fake, padding=2, normalize=True))

        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}][Batch {i}/{num_batches}] "
                  f"D(GAN): {loss_D.item():.4f} | G(GAN): {loss_G.item():.4f}")

        iters += 1

    # end of dataloader loop for epoch

# -----------------------------
# 9) SAVE MODELS
# -----------------------------
torch.save(netG.state_dict(), os.path.join(save_dir, 'netG.pth'))
torch.save(netD.state_dict(), os.path.join(save_dir, 'netD.pth'))
print(f"Models saved to: {save_dir}")
print("Training Complete!")

#Plot and save G and D Training Losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator Loss")
plt.plot(D_losses,label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot real and fake images (if we have stored any)
real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=5, normalize=True).cpu(),(1,2,0)))

plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
if len(img_list) > 0:
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
else:
    plt.text(0.5, 0.5, "No generated images saved", horizontalalignment='center')
plt.show()
