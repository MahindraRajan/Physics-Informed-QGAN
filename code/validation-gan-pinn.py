import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import matplotlib.image as mpimg
import numpy as np
import scipy
from scipy import ndimage
import cv2

# --------------------------------------------
# Define Generator class (must match training)
# --------------------------------------------
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

# --------------------------------------------
# Load state_dict and move to eval mode
# --------------------------------------------
save_path = "C:/.../PINN_GAN_SAVE/netG.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = nn.DataParallel(Generator(ngpu=2)).to(device)
netG.load_state_dict(torch.load(save_path, map_location=device))
netG.eval()

# --------------------------------------------
# Physics-informed loss for A(w₀)
# --------------------------------------------
def validation_loss(pred_params, A_target, w0_target, Q_min, lambda_A=26, lambda_w0=14, lambda_Q=26):
    """
    Composite validation loss:
    - A_target: target absorption at w0
    - w0_target: desired resonance frequency
    - Gamma_target: desired Linewidth
    """
    # Rescale predicted parameters
    omega_0 = pred_params[:, 0:1] * 50 + 25  # [25, 75]
    Gamma   = pred_params[:, 1:2] * 10 + 1   # [1, 11]
    kappa   = pred_params[:, 2:3]
    q       = pred_params[:, 3:4] * 215.33279325 + (-41.84328885)  # ~[-41.84, 173.49]

    delta = torch.zeros_like(omega_0)
    A = ((4 * kappa) / (1 + kappa)**2) * ((q**2) / (1 + delta**2))

    # Loss terms
    loss_A  = (A - A_target)**2
    loss_w0 = ((omega_0 - w0_target) / 50)**2
    loss_q  = ((q - q_target)/10)**2

    total_loss = lambda_A * loss_A + lambda_w0 * loss_w0 + lambda_q * loss_q
    return total_loss.mean()

# --------------------------------------------
# Step 1: Optimize physics params
# --------------------------------------------
physics_params = torch.rand(1, 4, requires_grad=True, device=device)
optimizer = torch.optim.Adam([physics_params], lr=0.0001)

for epoch in range(200000):
    loss = validation_loss(physics_params, A_target=0.9, w0_target=50.0, q_target=20.87)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"[{epoch}] Loss: {loss.item():.6f}")

    if loss.item() < 1e-6:
        print(f"[{epoch}] Loss: {loss.item():.6f}")
        break

physics_params.data = physics_params.data.clamp(0, 1)

# --------------------------------------------
# Step 2: Generate image
# --------------------------------------------
latent = torch.rand(1, 16, device=device)
z = torch.cat([physics_params.detach(), latent], dim=1)
fake = netG(z).detach().cpu()

# --------------------------------------------
# Step 3: Plot result
# --------------------------------------------
img = make_grid(fake, normalize=True).permute(1, 2, 0).numpy()
plt.imshow(img)
plt.title("Generated Metasurface")
plt.axis('off')
plt.show()
plt.imsave("pinn-generated_metasurface.png", img)

# --------------------------------------------
# Print optimized parameters
# --------------------------------------------
omega_0 = physics_params[0, 0].item() * 50 + 25
Gamma   = physics_params[0, 1].item() * 10 + 1
kappa   = physics_params[0, 2].item()
q       = physics_params[0, 3].item() * (173.4895044 + 41.84328885) + (-41.84328885)

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
