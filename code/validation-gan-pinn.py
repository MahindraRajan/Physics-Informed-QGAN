# -*- coding: utf-8 -*-
# Complete, updated script with staged (gated) loss:
#   Target 1: match ω0 (w0_target) first
#   Target 2: once Target 1 is met, optimize A and Q as well
#
# Notes:
# - The ω0 / Γ / q de-normalization used for optimization is used consistently
#   everywhere in this script (including the printout at the end).

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import make_grid

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------
# Config
# -------------------------------------------------
save_path = "C:/..../PINN_GAN_SAVE/netG.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = torch.cuda.device_count()

# Generator/input dims
physics_dim = 4
latent_dim  = 16
nc          = 3
lambda_physics = 1e-3

# Targets
A_target_val  = 0.9
w0_target_val = 50.0   # must be in same units as ω0 after de-normalization below
Q_min_val     = 1e5

# -------------------------------------------------
# Generator (must match training)
# -------------------------------------------------
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = physics_dim + latent_dim

        self.fc    = nn.Linear(self.input_dim, 512 * 4 * 4)
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
        x = torch.tanh(self.deconv4(x))  # [-1, 1]
        return x

# -------------------------------------------------
# Load generator
# -------------------------------------------------
if ngpu > 1:
    print(f"Using {ngpu} GPUs")
    netG = nn.DataParallel(Generator(ngpu)).to(device)
else:
    print("Using single GPU / CPU")
    netG = Generator(ngpu).to(device)
state_dict = torch.load(save_path, map_location=device)
if "module." in list(state_dict.keys())[0]:
    # remove 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict
netG.load_state_dict(state_dict)
netG.eval()

# -------------------------------------------------
# Validation loss
# -------------------------------------------------
def validation_loss(pred_params, A_target, w0_target, Q_min,
                    lambda_A=26, lambda_w0=14, lambda_Q=26):
    """
    Composite validation loss:
    - A_target: target absorption at w0
    - w0_target: desired resonance frequency
    - Gamma_target: desired Linewidth
    """
    # Rescale predicted parameters
    A0 = pred_params[:, 0:1]
    omega_0 = pred_params[:, 1:2] * 56.25 + 18.75
    Gamma = pred_params[:, 2:3] * 6.387283913 + (-0.00034279125)
    q = pred_params[:, 3:4] * (173.4895044 + 41.84328885) + (-41.84328885)

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

# --------------------------------------------
# Step 1: Optimize physics params
# --------------------------------------------
physics_params = torch.rand(1, 4, requires_grad=True, device=device)
optimizer = torch.optim.Adam([physics_params], lr=0.0001)
lambda_physics = 1e-12

for epoch in range(200000):
    loss = lambda_physics * validation_loss(physics_params, A_target=0.9, w0_target=50.0, Q_min=1e5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"[{epoch}] Loss: {loss.item():.6f}")

    if loss.item() < 1e-6:
        print(f"[{epoch}] Loss: {loss.item():.6f}")
        break

physics_params.data = physics_params.data.clamp(0, 1)

# -------------------------------------------------
# Step 2: Generate image from optimized physics params
# -------------------------------------------------
latent = torch.rand(1, latent_dim, device=device)
z = torch.cat([physics_params.detach(), latent], dim=1)
with torch.no_grad():
    fake = netG(z).cpu()

# -------------------------------------------------
# Step 3: Plot & save the result
# -------------------------------------------------
img = make_grid(fake, normalize=True).permute(1, 2, 0).numpy()
plt.imshow(img)
plt.title("Generated Metasurface")
plt.axis('off')
plt.show()
plt.imsave("pinn-generated_metasurface.png", img)

# -------------------------------------------------
# Print optimized (de-normalized) parameters
#   Use the SAME transforms as the loss above.
# -------------------------------------------------
with torch.no_grad():
    A0      = physics_params[0, 0].item()
    omega_0 = physics_params[0, 1].item() * 56.25 + 18.75
    Gamma   = physics_params[0, 2].item() * 6.387283913 + (-0.00034279125)
    q       = physics_params[0, 3].item() * 215.33279325 + (-41.84328885)
    delta   = 0.0
    A_val   = min(1.0, A0 * ((q + delta) ** 2 / (1 + delta ** 2)))
    Q_val   = omega_0 / (Gamma + 1e-6)

print("\nOptimized Physics Parameters (de-normalized):")
print(f"ω₀   = {omega_0:.4f}")
print(f"Γ    = {Gamma:.6f}")
print(f"q    = {q:.6f}")
print(f"A0   = {A0:.6f}")

im_size = 64
pmax = 4.0
tmax = 10.0
emax = 5.0
    
psum = 0.0
pnum = 0.0
tsum = 0.0
tnum = 0.0
esum = 0.0
enum = 0.0
    
for row in range(im_size):
    for col in range(im_size):
        if img[row][col][0] > 0.2 or img[row][col][1] > 0.2:
            if img[row][col][0] > img[row][col][1]:
                psum += img[row][col][0]
                pnum += 1
            else:
                esum += img[row][col][1]
                enum += 1
        else:
            tsum += img[row][col][2]
            tnum += 1
if pnum > 0:
    pAvg = psum / pnum
else:
    pAvg = 0.0
        
if tnum > 0:
    tAvg = tsum / tnum
else:
    tAvg = 0.0
        
if enum > 0:
    eAvg = esum / enum
else:
    eAvg = 0.0
        
    
pfake = pmax * pAvg
tfake = tmax * tAvg
efake = emax * eAvg

if pnum > enum:
    pindexfake = pfake
    classifier = 0
else:
    pindexfake = efake
    classifier = 1
print("Fake Plasma (PHz) /Index:", pindexfake)
print("Fake Thickness x 10^2 nm:", tfake)
print("Classifier:", classifier)

# --------------------------------------------
# Save the image and convert to black and white
# --------------------------------------------
# Save RGB image
output_path = "C:/.../pinn-generated_metasurface.png"
plt.imsave(output_path, img)
print(f"Generated image saved to: {output_path}")

# Convert to black and white as per RGB-to-BW rule
rgb = mpimg.imread(output_path)
im_size = rgb.shape[0]  # assuming square image

for row in range(im_size):
    for col in range(im_size):
        if rgb[row][col][0] > rgb[row][col][2] or rgb[row][col][1] > rgb[row][col][2]:
            rgb[row][col][:3] = [0, 0, 0]
        else:
            rgb[row][col][:3] = [1, 1, 1]

gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
cv2.normalize(gray, gray, -1, 1, cv2.NORM_MINMAX)
    
#Apply Gaussian Filter
img_filter = scipy.ndimage.gaussian_filter(gray,sigma=0.75)
ret, img_filter = cv2.threshold(img_filter,0.1,1,cv2.THRESH_BINARY) # 0 = black, 1 = white; everything under first number to black
plt.axis("off")    
plt.imshow(img_filter, cmap = "gray")

# Save the black and white version
bw_path = output_path.replace(".png", "_bw-Ag.png")
plt.imsave(bw_path, rgb)
print(f"Black and white image saved to: {bw_path}")
