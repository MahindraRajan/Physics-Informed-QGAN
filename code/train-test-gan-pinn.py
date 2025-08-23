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

#Get GPU Information
print("CUDA is available: {}".format(torch.cuda.is_available()))
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))

# -----------------------------
# 1) CONFIGURATION & PATHS
# -----------------------------
spectra_path = 'C:/.../absorptionData_HybridGAN.csv'  # CSV file with physics parameters ""
img_path = 'C:/.../Images'   # Root directory for images
save_dir = 'C:/.../PINN_GAN_SAVE/'   # Save models after training

#Does not truncate tensor contents (Can set "Default")
torch.set_printoptions(profile="full")

#Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#Number of workers for dataloader (for Windows workers must = 0, for reference: https://github.com/pytorch/pytorch/issues/2341)
workers = 2 
image_size  = 64       # Image dimensions: 64x64
nc          = 3        # Number of channels (RGB)
latent_dim  = 16       # Dimension of the noise vector
physics_dim = 4        # Use 4 values from the CSV (physics parameters)
nz = physics_dim + latent_dim
batch_size  = 16
num_epochs  = 500
lr          = 1e-5
beta1       = 0.5
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()

# -----------------------------
# 2) CREATE THE DATASET & DATALOADER
# -----------------------------
# Using ImageFolder to load images (dataset.imgs holds filenames)
dataset = dset.ImageFolder(root=img_path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

                           ]))
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=workers)

num_batches = len(dataloader)
print("No. of epochs:", num_batches)

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

# -----------------------------
# 3) FUNCTION TO LOAD PHYSICS PARAMETERS
# -----------------------------
def Excel_Tensor(spectra_path):
    # Load the CSV file (with header and using the first column as index)
    excelData = pd.read_csv(spectra_path, header=0, index_col=0)
    # Extract the first 4 columns as the physics parameters (of the absorption spectrum)
    excelDataSpectra = excelData.iloc[:,:4]
    # Convert to a tensor of shape [num_samples, 4]
    excelDataTensor = torch.tensor(excelDataSpectra.values, dtype=torch.float)
    return excelData, excelDataSpectra, excelDataTensor

excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)

# -----------------------------
# 4) GENERATOR
# -----------------------------
# The Generator takes a concatenated input of physics parameters (4) + noise (latent_dim)
# and outputs a 64x64 RGB image.
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
        # Concatenate the physics parameters and noise vector
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
# The Discriminator has two heads:
#  - One for classifying images as real or fake (GAN loss)
#  - One for predicting the physics parameters (should match the input physics parameters)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.l1 = nn.Linear(4, image_size*image_size*nc, bias=False)  
        
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
        
        # GAN head for real/fake classification
        self.gan_output = nn.Linear(256, 1)
        
        # Physics head for predicting the 4-dimensional physics parameters
        self.physics_output = nn.Linear(256, physics_dim)

    def forward(self, img, label):
        x1 = img
        x2 = self.l1(label)
        x2 = x2.view(-1, nc, image_size, image_size)
        x = torch.cat((x1,x2),1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc(x), 0.2)
        gan_pred = torch.sigmoid(self.gan_output(x))
        physics_pred = self.physics_output(x)
        return gan_pred, physics_pred

# -----------------------------
# 6) NETWORK INITIALIZATION
# -----------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Create the generator and discriminator
netG = Generator(ngpu).to(device)
netD = Discriminator(ngpu).to(device)

#Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)
netD.apply(weights_init)

#Print the model
print(netG)
print(netD)

# -----------------------------
# 7) LOSS FUNCTIONS & OPTIMIZERS
# -----------------------------
criterion_gan = nn.BCELoss()      # For real/fake classification

# Physics-informed loss
def physics_constraint_loss(pred_params, A_target=0.9, w0_target=50.0, Q_min=1e5, lambda_A=26, lambda_w0=14, lambda_Q=26): # lambda's are the physics hyperparameters
    
    """
    Physics-based loss evaluated at ω = ω₀.
    Assumes pred_params are normalized and need rescaling back to THz.
    A0 is the first parameter (0–1), followed by w0, Gamma, q.
    """
    
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
#Establish convention for real and fake labels during training
real_label = random.uniform(0.9,1.0)
fake_label = 0
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.rand(64, nz, device=device)

# -----------------------------
# 8) TRAINING LOOP
# -----------------------------
# Since shuffle=False in the DataLoader, we assume the order of images matches the order in excelDataTensor.
img_list = []
G_losses = []
D_losses = []
iters = 0
noise = torch.Tensor()
noise2 = torch.Tensor()
print("Starting Training Loop...")
#For each epoch
x=0
for epoch in range(num_epochs):
    x = 0
    for i, (real_imgs, _) in enumerate(dataloader):
        netD.zero_grad()
        # Format batch
        real_imgs = real_imgs.to(device)
        b_size = real_imgs.size(0)
        label_real = torch.empty(b_size, 1, device=device, dtype=torch.float).uniform_(0.9, 1.0)
        label_fake = torch.empty(b_size, 1, device=device, dtype=torch.float).uniform_(0.0, 0.1)

        # Generate batch of Spectra,  latent vectors, and Properties     
        for j in range(batch_size):
            excelIndex = x*batch_size+j
            try:
                gotdata = excelDataTensor[excelIndex]
            except IndexError:
                break
            tensorA = excelDataTensor[excelIndex].view(1,4)
            noise2 = torch.cat((noise2,tensorA),0)      
            
            tensor1 = torch.cat((excelDataTensor[excelIndex],torch.rand(latent_dim)))
            tensor2 = tensor1.unsqueeze(1)       
            tensor3 = tensor2.permute(1,0)
            noise = torch.cat((noise,tensor3),0) 
                              
        noise = noise.to(device)            
        noise2 = noise2.to(device)
        noise3 = torch.rand(2*b_size, physics_dim + latent_dim).to(device)
        noise4 = torch.rand(2*b_size, physics_dim).to(device)

        # D(x)
        gan_out_real, _ = netD(real_imgs, noise2)
        loss_real = criterion_gan(gan_out_real, label_real)

        # D(G(z))
        fake_imgs = netG(noise)
        gan_out_fake, _ = netD(fake_imgs.detach(), noise2)
        loss_fake = criterion_gan(gan_out_fake, label_fake)

        # Physics loss from predicted parameters
        gen_real = netG(noise3)
        _, pred_phys = netD(gen_real.detach(), noise4)
        loss_physics_D = physics_constraint_loss(pred_phys)

        loss_D_gan = loss_real + loss_fake
        loss_D = loss_D_gan + lambda_physics * loss_physics_D
        loss_D.backward()
        optimizerD.step()

        # --- Train Generator ---
        netG.zero_grad()
        fake_imgs = netG(noise)
        gan_out_fake, _ = netD(fake_imgs, noise2)
        loss_G_gan = criterion_gan(gan_out_fake, label_real)

        gen_fake = netG(noise3)
        _, pred_phys = netD(gen_fake.detach(), noise4)
        loss_physics_G = physics_constraint_loss(pred_phys)
        loss_G = loss_G_gan + lambda_physics * loss_physics_G
        loss_G.backward()
        optimizerG.step()

        if i % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}][Batch {i}/{len(dataloader)}] "
                  f"Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f} | "
                  f"D(GAN): {loss_D_gan.item():.4f} | G(GAN): {loss_G_gan.item():.4f} | "
                  f"Phys_D: {loss_physics_D.item():.4f} | Phys_G: {loss_physics_G.item():.4f}")

        # Save Losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
        noise = torch.Tensor()
        noise2 = torch.Tensor()
        x += 1


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

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
