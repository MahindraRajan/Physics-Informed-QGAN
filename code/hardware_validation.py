import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import scipy.ndimage
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService
from models import IWAE  # Ensure correct import path

# === User-configurable parameters ===
IBM_TOKEN = "YOUR API TOKEN HERE"
IBM_INSTANCE = "INSTANCE NAME / CRN NUMBER"
n_qubits = 9
q_depth = 2
n_generators = 1
latent_dim = 5
shots = 4096
A_target = 0.9
w0_target = 50.0
Q_min = 2e5
gen_path = ".../final_generator_qgan_iwae.pth"
vae_path = ".../pretrained_iwae.pth"
output_image_path = "qpinn_generated_metasurface_highQ.png"
bw_image_path = output_image_path.replace(".png", "_bw.png")

# === Device setup for classical computations ===
device_torch = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Torch device:", device_torch)

# === Setup IBM Q service and choose backend ===
QiskitRuntimeService.save_account(token=IBM_TOKEN, instance=IBM_INSTANCE, set_as_default=True, overwrite=True)
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print("Selected backend:", backend)

# === Configure PennyLane hardware device with mitigation options ===
dev_hw = qml.device(
    "qiskit.remote",
    wires=n_qubits,
    backend=backend,
    shots=shots,
    optimization_level=3,
    resilience_level=2,      # Enables ZNE + (if supported) readout
    seed_transpiler=42
)

# === Modified quantum generator class with fixed device handling ===
class QuantumGeneratorHW(nn.Module):
    def __init__(self, n_qubits, q_depth, n_generators, device_hw, torch_device=torch.device("cpu")):
        super(QuantumGeneratorHW, self).__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators
        self.dev = device_hw
        self.torch_device = torch_device
        
        self.q_params = nn.ParameterList([
            nn.Parameter(
                nn.init.orthogonal_(torch.rand(n_qubits, n_qubits, device=self.torch_device) * 0.02 - 0.01),
                requires_grad=True
            )
            for _ in range(n_generators)
        ])
        
        self.bias = nn.ParameterList([
            nn.Parameter(
                torch.empty(n_qubits, device=self.torch_device).uniform_(-0.01, 0.01),
                requires_grad=True
            )
            for _ in range(n_generators)
        ])
    
    def efficient_su2_entanglement(self, params):
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
    
    def _single_branch_circuit(self, params):
        @qml.qnode(self.dev, interface='torch')
        def circuit(theta):
            for _ in range(self.q_depth):
                self.efficient_su2_entanglement(theta)
            return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])
        # Pass a torch tensor directly
        return circuit(params.cpu().detach().numpy())

    def forward(self, x):
        batch_size = x.size(0)
        results = []
        x = x.to(self.torch_device)
        for i in range(batch_size):
            xi = x[i]
            branch_results = []
            for W, b in zip(self.q_params, self.bias):
                theta = xi.matmul(W) + b
                theta = theta.to(self.torch_device)
                out = self._single_branch_circuit(theta)
                # Ensure out is a torch tensor
                if not isinstance(out, torch.Tensor):
                    out = torch.tensor(out, dtype=torch.float32, device=self.torch_device)
                branch_results.append(out)
            concatenated = torch.cat(branch_results, dim=0)
            results.append(concatenated)
        return torch.stack(results, dim=0)

# === Load models ===
generator_hw = QuantumGeneratorHW(n_qubits, q_depth, n_generators, dev_hw, torch_device=device_torch).to(device_torch)
generator_hw.load_state_dict(torch.load(gen_path, map_location=device_torch))
generator_hw.eval()

iwae = IWAE(n_qubits=n_qubits, nc=3, beta=2.0, num_samples=5).to(device_torch)
iwae.load_state_dict(torch.load(vae_path, map_location=device_torch))
iwae.eval()

# === Validation loss function (unchanged) ===
def validation_loss(pred_params, A_target, w0_target, Q_min, lambda_A=26, lambda_w0=14, lambda_Q=26):
    A0 = pred_params[:, 0:1]
    omega_0 = pred_params[:, 1:2] * 56.25 + 18.75
    Gamma = pred_params[:, 2:3] * 6.387283913 + (-0.00034279125)
    q_val = pred_params[:, 3:4] * 215.33279325 + (-41.84328885)
    delta = torch.zeros_like(omega_0)
    A = A0 * ((q_val + delta)**2 / (1 + delta**2))
    A = torch.min(A, torch.ones_like(A))
    Q_val = omega_0 / (Gamma + 1e-6)
    Q_penalty = torch.relu(Q_min - Q_val)**2
    loss_A = torch.clamp(A_target - A, min=0.0)**2
    loss_w0 = ((omega_0 - w0_target) / 50)**2
    loss1 = lambda_A * loss_A + lambda_w0 * loss_w0
    loss2 = lambda_Q * Q_penalty
    loss  = loss1 + loss2
    return loss.mean()

# === Optimize physics parameters ===
physics_params = torch.rand(1, 4, requires_grad=True, device=device_torch)
optimizer = torch.optim.Adam([physics_params], lr=1e-4)

for epoch in range(200000):
    loss = validation_loss(physics_params, A_target, w0_target, Q_min)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"[{epoch}] Loss: {loss.item():.6f}")
    if loss.item() < 1e-6:
        print(f"Converged at epoch {epoch}, loss={loss.item():.6f}")
        break
physics_params.data = physics_params.data.clamp(0.0, 1.0)

# === Prepare latent input and run circuit ===
latent = torch.rand(1, latent_dim, device=device_torch)
# Convert to CPU for QNode if needed
theta_cpu = physics_params.detach().cpu().float()
latent_cpu = latent.detach().cpu().float()

# Execute the quantum generator circuit
fake_latent = generator_hw.forward(torch.cat([theta_cpu, latent_cpu], dim=1))
fake_latent = fake_latent.to(device_torch)

# === Decode to image via VAE decoder ===
with torch.no_grad():
    fake_images = iwae.decoder(fake_latent)
fake_img = fake_images.cpu().squeeze().permute(1,2,0).numpy()

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
        if fake_img[row][col][0] > 0.2 or fake_img[row][col][1] > 0.2:
            if fake_img[row][col][0] > fake_img[row][col][1]:
                psum += fake_img[row][col][0]
                pnum += 1
            else:
                esum += fake_img[row][col][1]
                enum += 1
        if fake_img[row][col][2] > 0.2:
            tsum += fake_img[row][col][2]
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

print("Fake Plasma/Index:", efake)
print("Fake Thickness:", tfake)

# === Save RGB image ===
plt.imshow(fake_img, interpolation='none')
plt.axis('off')
plt.title("Generated Metasurface (High‐Q) via Hardware")
plt.show()
plt.imsave(output_image_path, fake_img)

# === Convert to black & white and save ===
rgb = mpimg.imread(output_image_path)
h,w,_ = rgb.shape
for r in range(h):
    for c in range(w):
        if rgb[r,c,0] > rgb[r,c,2] or rgb[r,c,1] > rgb[r,c,2]:
            rgb[r,c,:3] = [0,0,0]
        else:
            rgb[r,c,:3] = [1,1,1]
gray = cv2.cvtColor((rgb*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
img_filter = scipy.ndimage.gaussian_filter(gray.astype(float), sigma=0.75)
_, bw_img = cv2.threshold(img_filter, 50, 255, cv2.THRESH_BINARY)
plt.imshow(bw_img, cmap="gray")
plt.axis('off')
plt.show()
plt.savefig(bw_image_path, bbox_inches='tight', pad_inches=0)
print(f"Saved B&W image: {bw_image_path}")
