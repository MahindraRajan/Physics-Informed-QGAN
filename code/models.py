# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# Define IWAE model class
class IWAE(nn.Module):
    def __init__(self, n_qubits, nc, beta, num_samples):
        super(IWAE, self).__init__()
        self.n_qubits = n_qubits
        self.beta = beta
        self.num_samples = num_samples

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, n_qubits)
        self.fc_logvar = nn.Linear(1024, n_qubits)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),  # 4x4 -> 512 channels
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, num_samples):
        batch_size = x.size(0)
        
        x = x.unsqueeze(1).expand(-1, num_samples, -1, -1, -1).reshape(batch_size * num_samples, *x.size()[1:])
        
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded).view(batch_size, num_samples, -1)
        logvar = self.fc_logvar(encoded).view(batch_size, num_samples, -1)
        
        z = self.reparameterize(mu, logvar).view(batch_size * num_samples, -1)

        z = torch.tanh(z)
        
        # Ensure the decoder output matches [batch_size * num_samples, 3, 64, 64]
        reconstructed = self.decoder(z)
        
        return reconstructed.view(batch_size, num_samples, *x.size()[1:]), mu, logvar, z

    @staticmethod
    def iwae_loss_function(reconstructed, x, mu, logvar, beta, num_samples):
        batch_size = x.size(0)
        
        # Reshape the input tensor to match the expected batch_size and num_samples
        x = x.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
        
        # Compute Reconstruction Loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='none')
        reconstruction_loss = reconstruction_loss.view(batch_size, num_samples, -1).sum(dim=-1)
        
        # Compute KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        # IWAE Loss (log-sum-exp trick for numerical stability)
        log_weight = -reconstruction_loss - beta * kl_divergence
        max_log_weight = log_weight.max(dim=1, keepdim=True)[0]
        weight = torch.exp(log_weight - max_log_weight)

        # Compute Importance Weighted Loss
        loss = -max_log_weight.squeeze() - torch.log(weight.mean(dim=1) + 1e-10)
        return loss.mean()

# Define the VAE class
class BetaVAE(nn.Module):
    def __init__(self, n_qubits, nc, beta):
        super(BetaVAE, self).__init__()
        self.n_qubits = n_qubits
        self.beta = beta
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, n_qubits)
        self.fc_logvar = nn.Linear(1024, n_qubits)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),  # 4x4 -> 512 channels
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = torch.tanh(self.reparameterize(mu, logvar))
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z

    @staticmethod
    def loss_function(reconstructed, x, mu, logvar, beta):
        # Reconstruction loss (MSE or BCE)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='sum')
        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Combine losses with beta factor
        loss = reconstruction_loss + beta * kl_divergence
        return loss

# Define the Classical Discriminator
class Discriminator(nn.Module):
    def __init__(self, n_qubits, label_dims):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(label_dims, n_qubits, bias=False)
        self.model = nn.Sequential(
            nn.Linear(2*n_qubits, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64, bias=False),
        )
        # GAN head for real/fake classification
        self.gan_output = nn.Linear(64, 1)

        # Physics head for predicting the 4-dimensional physics parameters
        self.physics_output = nn.Linear(64, 4)

    def forward(self, inputs, label):
        x1 = inputs
        x2 = self.l1(label)
        combined_input = torch.cat((x1, x2), dim=1)
        x = F.leaky_relu(self.model(combined_input), 0.2)
        gan_pred = torch.sigmoid(self.gan_output(x))
        physics_pred = self.physics_output(x)
        return gan_pred.view(-1), physics_pred

# Define Quantum Generator class
class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits, q_depth, n_generators):
        super(QuantumGenerator, self).__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators
        
        # Initialize the quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize the weights for the orthogonal transformation
        self.q_params = nn.ParameterList([
            nn.Parameter(nn.init.orthogonal_(torch.rand(n_qubits, n_qubits) * 0.02 - 0.01),   requires_grad=True)
            for _ in range(n_generators)
        ])

        
        # Initialize the biases
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(n_qubits).uniform_(-0.01, 0.01), requires_grad=True)
            for _ in range(n_generators)
        ])
        
        # Linear layer to map to the final output size
        #self.fc = nn.Linear(n_generators * n_qubits, n_qubits)

   
    def efficient_su2_entanglement(self, params):
        """Parameterized Efficient SU2 with Circular Entanglement."""
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i % self.n_qubits)

        # Circular entanglement
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i % self.n_qubits)

    def quantum_generator(self, params):
        """Define the QNode."""
        @qml.qnode(self.dev, interface='torch')
        def circuit():
            # Efficient SU2-like ansatz with circular entanglement
            for _ in range(self.q_depth):
                self.efficient_su2_entanglement(params)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]      
        
        return circuit()

    def forward(self, x):
        batch_sizes = x.size(0) # Get the batch size
        batch_results = []

        for i in range(batch_sizes): # Loop over the batch size
            element_results = []
            for W, b in zip(self.q_params, self.bias):
                # Apply orthogonal transformation
                theta = torch.matmul(x[i], W) + b
                output = self.quantum_generator(theta)  # Pass each batch element through the quantum generator
                element_results.append(output.to(x.device, dtype=torch.float32))
            batch_results.append(torch.cat(element_results, dim=0))  # Concatenate results for this element

        x = torch.stack(batch_results) # Stack to reform the batch dimension
        
        return x
