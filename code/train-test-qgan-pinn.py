import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import torchvision.utils as vutils 
import pandas as pd
import random
import time
from models.models import IWAE, Discriminator, QuantumGenerator

#Get GPU Information
print("CUDA is available: {}".format(torch.cuda.is_available()))
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))

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

# Load the pretrained QVAE and train the QGAN

def train_qgan(generator, discriminator, iwae, num_samples, excelDataTensor, dataloader, optimizerG, optimizerD, criterion, num_epochs):
    generator.train()
    discriminator.train()
    iwae.eval()  # Pretrained IWAE in eval mode
    x = 0
    noise = torch.Tensor()
    noise2 = torch.Tensor()
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        x = 0
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size * num_samples,), real_label, device=device, dtype=torch.float)

            with torch.no_grad():
                recon_real, mu, logvar, latent_real = iwae(real_cpu, num_samples)

            noise = torch.Tensor()
            noise2 = torch.Tensor()

            for j in range(b_size):
                excelIndex = x * b_size + j
                try:
                    gotdata = excelDataTensor[excelIndex]
                except IndexError:
                    break

                for _ in range(num_samples):
                    tensorA = gotdata.view(1, label_dims)
                    noise2 = torch.cat((noise2, tensorA), 0)
                    tensor1 = torch.cat((gotdata, torch.rand(latent)), dim=0).view(1, n_qubits)
                    noise = torch.cat((noise, tensor1), 0)

            noise = noise.to(device)
            noise2 = noise2.to(device)
            noise3 = torch.rand(2*b_size, label_dims + latent).to(device)
            noise4 = torch.rand(2*b_size, label_dims).to(device)

            fake_latent = generator.forward(noise).to(device)

            output_real, _ = discriminator.forward(latent_real, noise2)
            errD_real = criterion(output_real.view(-1), label)

            label = torch.full((b_size * num_samples,), fake_label, device=device, dtype=torch.float)
            output_fake, _ = discriminator.forward(fake_latent, noise2)
            errD_fake = criterion(output_fake.view(-1), label)
            errD_gan = errD_real + errD_fake

            gen_phys_real = generator.forward(noise3).to(device)
            _, pred_phys = discriminator(gen_phys_real.detach(), noise4)
            physics_loss_D = physics_constraint_loss(pred_phys)
            errD = errD_gan + lambda_physics * physics_loss_D
            errD.backward(retain_graph=True)
            optimizerD.step()

            generator.zero_grad()
            label = torch.full((b_size * num_samples,), real_label, device=device, dtype=torch.float)
            output, _ = discriminator.forward(fake_latent, noise2)
            errG_gan = criterion(output.view(-1), label)

            gen_phys_fake = generator.forward(noise3).to(device)
            _, pred_phys = discriminator(gen_phys_real.detach(), noise4)
            physics_loss_G = physics_constraint_loss(pred_phys)
            errG = errG_gan + lambda_physics * physics_loss_G
            errG.backward(retain_graph=True)
            optimizerG.step()

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch+1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item()))

            x += 1
            noise = torch.Tensor()
            noise2 = torch.Tensor()

    print("QGAN Training Completed with Pretrained IWAE.")


# Testing function to check the generator's output using testTensor
def test_generator(generator, iwae, testTensor, device):
    generator.eval()
    iwae.eval()

    with torch.no_grad():
        fake_latent = generator(testTensor)
        fake_images = iwae.decoder(fake_latent)
        fake = fake_images.detach().cpu()
                
    img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))

    return img_list

def Excel_Tensor(spectra_path):
    # Location of excel data
    excelData = pd.read_csv(spectra_path, header = 0, index_col = 0)    
    excelDataSpectra = excelData.iloc[:,:4] #index until the last point of the spectra in the Excel file
    excelDataTensor = torch.tensor(excelDataSpectra.values).type(torch.FloatTensor)
    return excelData, excelDataSpectra, excelDataTensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
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
    workers = 1
    lrG = 1e-5  # Increase the learning rate for the generator
    lrD = 1e-4  # Keep the learning rate for the discriminator lower
    lambda_physics = 2.0
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
    # Establish convention for real and fake labels during training
    real_label = random.uniform(0.9,1.0)  # One-sided label smoothing for real labels
    fake_label = 0  # Fake label remains the same

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data preparation (Adjust the path as needed)
    img_path = 'C:/.../Images'
    spectra_path = 'C:/.../absorptionData_HybridGAN.csv'

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

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    # Initialize Quantum Generator and Discriminator
    generator = QuantumGenerator(n_qubits, q_depth, n_generators).to(device)
    discriminator = Discriminator(n_qubits, label_dims).to(device)

    # Apply the weights_init function to randomly initialize all weights
    discriminator.apply(weights_init)

    # Load the pretrained QVAE from a specific directory
    iwae = IWAE(n_qubits = n_qubits, nc = nc, beta = beta, num_samples = num_samples).to(device)
    iwae.load_state_dict(torch.load('pretrained_iwae.pth'))

    # Define optimizers
    optimizerG = optim.Adam(generator.parameters(), lr=lrG, betas=(beta1, beta2))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lrD, betas=(beta1, beta2))
    criterion = nn.BCELoss(reduction='mean')

    start_time = time.time()
    local_time = time.ctime(start_time)
    print('Start Time = %s' % local_time)

    # Train the QGAN
    train_qgan(generator, discriminator, iwae, num_samples, excelDataTensor, dataloader, optimizerG, optimizerD, criterion, num_epochs)

    local_time = time.ctime(time.time())
    print('End Time = %s' % local_time)
    run_time = (time.time()-start_time)/3600
    print('Total Time Lapsed = %s Hours' % run_time)
    
    # Save the final models
    torch.save(generator.state_dict(), 'final_generator_qgan_iwae.pth')
    torch.save(discriminator.state_dict(), 'final_discriminator_qgan_iwae.pth')

    # Test the generator using testTensor
    generator.load_state_dict(torch.load('final_generator_qgan_iwae.pth'))

    fixed_noise = torch.rand(64, nz, device=device)
    
    # Call the test function
    img_list = test_generator(generator, iwae, fixed_noise, device)
    
    # Plot and save G and D Training Losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="Generator Loss")
    plt.plot(D_losses,label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses-qgan-iwae.png')
    plt.show()

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
    plt.savefig('fake-and-real-iwae.png')
    plt.show()
