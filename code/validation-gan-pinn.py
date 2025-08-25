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
save_path = "PINN_GAN_SAVE/netG.pth"
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

# Gating / training
w0_tol = 2.0                  # absolute tolerance to consider Target 1 "achieved"
keep_w0_weight_after = 0.1    # keep a small ω0 weight after gating to avoid drift
max_epochs = 200000
lr = 1e-4
print_every = 2000

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
# Staged validation loss (ω0 first, then A & Q)
# -------------------------------------------------
def staged_validation_loss(
    pred_params,
    A_target,
    w0_target,
    Q_min,
    lambda_A=26, lambda_w0=14, lambda_Q=26,
    w0_tol=2.0,
    keep_w0_weight_after=0.0
):
    """
    pred_params in [0,1] for each column; de-normalized as below
    Returns: (loss_mean, info_dict)
    """
    # De-normalize to physical space (consistent everywhere)
    A0      = pred_params[:, 0:1]
    omega_0 = pred_params[:, 1:2] * 56.25 + 18.75              # ω0 in target units
    Gamma   = pred_params[:, 2:3] * 6.387283913 + (-0.00034279125)
    q       = pred_params[:, 3:4] * 215.33279325 + (-41.84328885)

    # Physics for A and Q
    delta = torch.zeros_like(omega_0)
    A = A0 * ((q + delta) ** 2 / (1 + delta ** 2))
    A = torch.min(A, torch.ones_like(A))  # cap A at 1.0

    Q_val     = omega_0 / (Gamma + 1e-6)
    Q_penalty = torch.relu(Q_min - Q_val) ** 2

    # Per-term losses
    loss_w0 = ((omega_0 - w0_target) / 50.0) ** 2
    loss_A  = torch.clamp(A_target - A, min=0.0) ** 2

    # Gate: Target 1 is "achieved" if |ω0 - w0_target| <= w0_tol
    t1_mask = (torch.abs(omega_0 - w0_target) <= w0_tol).float()  # [B,1]

    # Compose staged loss
    loss_before = lambda_w0 * loss_w0
    loss_after  = (keep_w0_weight_after * loss_w0
                   + lambda_A * loss_A
                   + lambda_Q * Q_penalty)

    loss = (1.0 - t1_mask) * loss_before + t1_mask * loss_after
    loss_mean = loss.mean()

    info = {
        "loss_w0": loss_w0.mean().detach(),
        "loss_A": loss_A.mean().detach(),
        "Q_penalty": Q_penalty.mean().detach(),
        "t1_reached_frac": t1_mask.mean().detach(),
        "omega_0": omega_0.detach(),
        "A": A.detach(),
        "Q_val": Q_val.detach(),
    }
    return loss_mean, info

# -------------------------------------------------
# Step 1: Optimize physics params (flowchart style)
# -------------------------------------------------
physics_params = torch.rand(1, physics_dim, requires_grad=True, device=device)
optimizer = torch.optim.Adam([physics_params], lr=lr)

phase = 1  # 1: ω0 only; 2: A & Q (keep small ω0 weight to avoid drift)

for epoch in range(max_epochs):
    optimizer.zero_grad()

    if phase == 1:
        loss, info = staged_validation_loss(
            physics_params, A_target=A_target_val, w0_target=w0_target_val, Q_min=Q_min_val,
            lambda_A=26, lambda_w0=14, lambda_Q=26,
            w0_tol=w0_tol, keep_w0_weight_after=0.0
        )
        # Switch to Phase 2 after Target 1 achieved for the batch
        if info["t1_reached_frac"].item() >= 1.0:
            phase = 2
    else:
        loss, info = staged_validation_loss(
            physics_params, A_target=A_target_val, w0_target=w0_target_val, Q_min=Q_min_val,
            lambda_A=26, lambda_w0=14, lambda_Q=26,
            w0_tol=w0_tol, keep_w0_weight_after=keep_w0_weight_after
        )

    (lambda_physics * loss).backward()
    optimizer.step()

    if epoch % print_every == 0:
        print(f"[Epoch = {epoch}], loss={loss.item():.4e} ")

    # Optional global stop when both targets are satisfied in Phase 2
    if phase == 2:
        a_ok = (info["A"] >= (A_target_val - 1e-3)).all().item()
        q_ok = (info["Q_val"] >= (Q_min_val - 1e-3)).all().item()
        if a_ok and q_ok:
            print(f"[{epoch}] Both targets satisfied; stopping.")
            break

with torch.no_grad():
    physics_params.clamp_(0, 1)

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
print(f"A(δ=0) capped = {A_val:.6f}")
print(f"Q    = {Q_val:.2f}")


print(f"\nOptimized Physics Parameters:")
print(f"ω₀   = {omega_0:.2f} THz")
print(f"Γ    = {Gamma:.2f} THz")
print(f"κ    = {kappa:.4f}")
print(f"q    = {q:.4f}")

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
