# -*- coding: utf-8 -*-
from __future__ import print_function
from Utilities.ConvertImageToBinary import Binary
from pathlib import Path
import scipy
from scipy import ndimage
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torchvision.utils import make_grid
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import os
import pennylane as qml
from models.models import IWAE, QuantumGenerator

#Location of Saved Generator and IWAE
genDir = "C:/.../PINN_QGAN_SAVE/final_generator_qgan_iwae.pth"
vaeDir = "C:/.../PINN_QGAN_SAVE/pretrained_iwae.pth"

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load state_dict and move to eval mode
n_qubits, q_depth, n_generator = 9, 2, 1
nc, beta, num_samples = 3, 2.0, 5
generator = QuantumGenerator(n_qubits, q_depth, n_generator).to(device)
generator.load_state_dict(torch.load(genDir))
generator.eval()
iwae = IWAE(n_qubits=n_qubits, nc=nc, beta=beta, num_samples=num_samples).to(device)
iwae.load_state_dict(torch.load(vaeDir))
iwae.eval()

# Physics-informed loss with Q > 1e4 constraint
def validation_loss(pred_params, A_target, w0_target, Q_min, lambda_A=26, lambda_w0=14, lambda_Q=26):
    A0 = pred_params[:, 0:1]
    omega_0 = pred_params[:, 1:2] * 56.25 + 18.75
    Gamma = pred_params[:, 2:3] * 6.387283913 + (-0.00034279125)
    q = pred_params[:, 3:4] * 215.33279325 + (-41.84328885)

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

# Optimize for high-Q (>1e4)
physics_params = torch.rand(1, 4, requires_grad=True, device=device)
optimizer = torch.optim.Adam([physics_params], lr=0.0001)

for epoch in range(200000):
    loss = validation_loss(physics_params, A_target=0.9, w0_target=50.0, Q_min=1e5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"[{epoch}] Loss: {loss.item()*1e-9:.6f}")
    if loss.item() < 1e-6:
        print(f"[{epoch}] Loss: {loss.item():.6f}")
        break

physics_params.data = physics_params.data.clamp(0, 1)

# Generate image
latent = torch.rand(1, 5, device=device)
z = torch.cat([physics_params.detach(), latent], dim=1)
fake_latent = generator(z)
fake_images = iwae.decoder(fake_latent)
img = fake_images.detach().cpu().permute(3, 2, 1, 0).squeeze()

# Plot result
img = make_grid(img, normalize=True).permute(1, 0, 2).numpy()
plt.imshow(img)
plt.title("Generated Metasurface (High-Q)")
plt.axis('off')
plt.show()
plt.imsave("qpinn-generated_metasurface-highQF.png", img)

A0 = physics_params[0, 0].item()
omega_0 = physics_params[0, 1].item() * 56.25 + 18.75
Gamma = physics_params[0, 2].item() * 6.387283913 + (-0.00034279125)
q = physics_params[0, 3].item() * (173.4895044 + 41.84328885) + (-41.84328885)
Q_factor = omega_0 / (Gamma + 1e-6)

print(f"\nOptimized Physics Parameters:")
print(f"ω₀   = {omega_0:.4f} THz")
print(f"Γ    = {abs(Gamma):.4f} THz")
print(f"A0    = {A0:.4f}")
print(f"q    = {q:.4f}")
print(f"Q     = {abs(Q_factor):.2f}")

im_size = 64
pmax = 0.0
tmax = 1.0
emax = 4.0

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
        if img[row][col][2] > 0.2:
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
print("Fake Index:", pindexfake)
print("Fake Thickness:", tfake, "um")
print("Classifier:", classifier)

# --------------------------------------------
# Save the image and convert to black and white
# --------------------------------------------
# Save RGB image
output_path = "C:/.../qpinn-generated_metasurface-highQF.png"
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
bw_path = output_path.replace(".png", "_bw.png")
plt.imsave(bw_path, rgb)
print(f"Black and white image saved to: {bw_path}")
