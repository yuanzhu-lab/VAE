# %%
from __future__ import print_function
# Set-up
import os

path = r"./"
if os.path.exists(path):
    print("ok")

# %%
import os
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt

if not os.path.exists('results'):
    os.mkdir('results')

batch_size = 100
latent_size = 20

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)
        self.fc3 = nn.Linear(latent_size, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        # The encoder will take an input of size 784, and will produce two vectors of size latent_size (corresponding to the coordinatewise means and log_variances)
        # It should have a single hidden linear layer with 400 nodes using ReLU activations, and have two linear output layers (no activations)
        h1 = F.relu(self.fc1(x))
        means = self.fc21(h1)
        log_variances = self.fc22(h1)
        return means, log_variances

    def reparameterize(self, means, log_variances):
        # The reparameterization module lies between the encoder and the decoder
        # It takes in the coordinatewise means and log-variances from the encoder (each of dimension latent_size), and returns a sample from a Gaussian with the corresponding parameters
        std = torch.exp(0.5 * log_variances)
        eps = torch.randn_like(std)
        return means + eps * std

    def decode(self, z):
        # The decoder will take an input of size latent_size, and will produce an output of size 784
        # It should have a single hidden linear layer with 400 nodes using ReLU activations, and use Sigmoid activation for its outputs
        h3 = F.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(h3))
        return x_recon

    def forward(self, x):
        # Apply the VAE encoder, reparameterization, and decoder to an input of size 784
        # Returns an output image of size 784, as well as the means and log_variances, each of size latent_size (they will be needed when computing the loss)
        x = x.view(-1, 784)
        means, log_variances = self.encode(x)
        z = self.reparameterize(means, log_variances)
        reconstructed_x = self.decode(z)
        return reconstructed_x, means, log_variances


def vae_loss_function(reconstructed_x, x, means, log_variances):
    # Compute the VAE loss
    # The loss is a sum of two terms: reconstruction error and KL divergence
    # Use cross entropy loss between x and reconstructed_x for the reconstruction error (as opposed to L2 loss as discussed in lecture -- this is sometimes done for data in [0,1] for easier optimization)
    # The KL divergence is -1/2 * sum(1 + log_variances - means^2 - exp(log_variances)) as described in lecture
    # Returns loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    x = x.view(-1, 784)

    reconstruction_loss = F.binary_cross_entropy(
        reconstructed_x, x, reduction='sum'
    )

    kl_divergence = -0.5 * torch.sum(
        1 + log_variances - means.pow(2) - log_variances.exp()
    )

    loss = reconstruction_loss + kl_divergence
    return loss, reconstruction_loss


def train(model, optimizer):
    # Trains the VAE for one epoch on the training dataset
    # Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    model.train()
    train_loss = 0.0
    train_recon_loss = 0.0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        reconstructed_x, means, log_variances = model(data)
        loss, reconstruction_loss = vae_loss_function(
            reconstructed_x, data, means, log_variances
        )
        loss.backward()
        train_loss += loss.item()
        train_recon_loss += reconstruction_loss.item()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_reconstruction_loss = train_recon_loss / len(train_loader.dataset)
    return avg_train_loss, avg_train_reconstruction_loss


def test(model):
    # Runs the VAE on the test dataset
    # Returns the average (over the dataset) loss (reconstruction + KL divergence) and reconstruction loss only (both scalars)
    model.eval()
    test_loss = 0.0
    test_recon_loss = 0.0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed_x, means, log_variances = model(data)
            loss, reconstruction_loss = vae_loss_function(
                reconstructed_x, data, means, log_variances
            )
            test_loss += loss.item()
            test_recon_loss += reconstruction_loss.item()

    avg_test_loss = test_loss / len(test_loader.dataset)
    avg_test_reconstruction_loss = test_recon_loss / len(test_loader.dataset)
    return avg_test_loss, avg_test_reconstruction_loss


epochs = 50
avg_train_losses = []
avg_train_reconstruction_losses = []
avg_test_losses = []
avg_test_reconstruction_losses = []

vae_model = VAE().to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    avg_train_loss, avg_train_reconstruction_loss = train(vae_model, vae_optimizer)
    avg_test_loss, avg_test_reconstruction_loss = test(vae_model)
    
    avg_train_losses.append(avg_train_loss)
    avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
    avg_test_losses.append(avg_test_loss)
    avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

    with torch.no_grad():
        sample = torch.randn(64, latent_size).to(device)
        sample = vae_model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
        print('Epoch #' + str(epoch))
        display(Image('results/sample_' + str(epoch) + '.png'))
        print('\n')

plt.figure()
plt.plot(range(1, epochs + 1), avg_train_reconstruction_losses)
plt.title('Training Reconstruction Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, epochs + 1), avg_test_reconstruction_losses)
plt.title('Test Reconstruction Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.grid(True)
plt.show()
