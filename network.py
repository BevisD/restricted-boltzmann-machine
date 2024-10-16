import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Check if MPS is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the Restricted Boltzmann Machine class
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units

        # Initialize weights with small random values
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def sample_h(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return h_prob, torch.bernoulli(h_prob)

    def sample_v(self, h):
        v_prob = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return v_prob, torch.bernoulli(v_prob)

    def contrastive_divergence(self, v0, eta=0.01, k=5):  # CD-5 for more stability
        v = v0
        for _ in range(k):
            h_prob, h = self.sample_h(v)
            v_prob, v = self.sample_v(h)

        # Positive phase (using original data)
        positive_grad = torch.matmul(v0.t(), self.sample_h(v0)[0])

        # Negative phase (using reconstructed data)
        negative_grad = torch.matmul(v_prob.t(), self.sample_h(v_prob)[0])

        # Update weights and biases with more moderate learning rate
        self.W.data += (positive_grad - negative_grad) * eta
        self.v_bias.data += torch.sum((v0 - v_prob), dim=0) * eta
        self.h_bias.data += torch.sum((self.sample_h(v0)[0] - self.sample_h(v_prob)[0]), dim=0) * eta

        # Compute reconstruction loss (mean squared error)
        loss = torch.mean((v0 - v_prob) ** 2)
        return loss

# Function to train the RBM with Momentum
def train_rbm(rbm, train_loader, eta=0.01, epochs=10, k=5, device='cpu'):
    losses = []
    rbm.to(device)  # Move RBM to MPS (or CPU)
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.view(-1, 784).to(device)  # Flatten the data from 28x28 to 784 and move to device
            data = data.bernoulli()  # Convert to binary (if you want to binarize the data)
            loss = rbm.contrastive_divergence(data, eta=eta, k=k)
            batch_loss = loss.item()
            epoch_loss += batch_loss
            losses.append(batch_loss)
            # print(f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {batch_loss:.4f}')
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}')
    return losses

def mask_image(img, mask_fraction=0.5):
    mask = torch.bernoulli(torch.full(img.shape, 1 - mask_fraction))
    return img * mask

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize the RBM (784 visible units for 28x28 images, and 128 hidden units)
rbm = RBM(visible_units=784, hidden_units=256)

# Train the RBM on MNIST dataset using DataLoader and MPS
losses = train_rbm(rbm, train_loader, eta=0.0001, epochs=20, k=5, device=device)
torch.save(rbm.state_dict(), 'models/rbm_model.pth')
# plt.plot(losses)
# plt.show()

test_image = next(iter(train_loader))[0][0].view(-1, 784)  # Take the first image from the DataLoader
masked_image = mask_image(test_image, mask_fraction=0.5).to(device)  # Mask out 50% of the pixels

# Display the original, masked, and reconstructed images
def display_images(original, masked, reconstructed):
    original = original.cpu().detach()
    masked = masked.cpu().detach()
    reconstructed = reconstructed.cpu().detach()

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(original.view(28, 28), cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(masked.view(28, 28), cmap='gray')
    axs[1].set_title('Masked Image')
    axs[2].imshow(reconstructed.view(28, 28), cmap='gray')
    axs[2].set_title('Reconstructed Image')
    plt.show()

# Reconstruct the image by sampling the missing pixels using the trained RBM
_, hidden_representation = rbm.sample_h(masked_image)
reconstructed_image, _ = rbm.sample_v(hidden_representation)

# Show the results
display_images(test_image, masked_image, reconstructed_image)