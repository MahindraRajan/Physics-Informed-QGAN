import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.models import IWAE

# Training the IWAE
def train_iwae(iwae, dataloader, beta, optimizer, num_epochs, num_samples, device):
    iwae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            optimizer.zero_grad()

            recon_real, mu, logvar, latent_real = iwae(real_cpu, num_samples)
            loss = iwae.iwae_loss_function(recon_real, real_cpu, mu, logvar, beta, num_samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_values.append(total_loss/len(dataloader.dataset))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}")

# Main script
if __name__ == '__main__':
    # Parameters
    latent_dim = 9
    beta = 2.0
    image_size = 64
    batch_size = 32
    num_epochs = 100
    num_samples = 5
    lr_vae = 1e-4
    loss_values = []
    beta1 = 0.5
    beta2 = 0.999
    nc = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data preparation (Adjust the path as needed)
    img_path = 'C:/.../Images'
    
    dataset = dset.ImageFolder(root=img_path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor()
                               ]))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the IWAE
    iwae = IWAE(n_qubits = latent_dim, nc = nc, beta = beta, num_samples = num_samples).to(device)
    optimizer = optim.Adam(iwae.parameters(), lr=lr_vae, betas=(beta1, beta2))

    # Train the IWAE
    train_iwae(iwae, dataloader, beta, optimizer, num_epochs, num_samples, device)

    # Save the pretrained IWAE model
    torch.save(iwae.state_dict(), 'pretrained_iwae.pth')

    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, color='b', marker='o', markevery=10, label='pretrained loss')
    plt.title(r'Training Loss over Epochs for pretrained IWAE on Absorption spectra dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.savefig('losses-iwae-abs.png')
    plt.show()