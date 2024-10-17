import os

import torch
from torchvision import datasets, transforms
import numpy as np

from network import RBM
from utils import reconstruction_loss

ETA = 0.0002
K = 5
EPOCHS = 10
BATCH_SIZE = 64
HIDDEN_UNITS = 128

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
val_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = RBM(visible_units=784, hidden_units=HIDDEN_UNITS)
model.eval()
model.to(device)

losses = []
recon_train_losses = []
recon_val_losses = []

# Start training
for epoch in range(EPOCHS):
    # Train epoch
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        data = data.bernoulli()
        loss = model.contrastive_divergence(data, eta=ETA, k=K)
        batch_loss = loss.item()
        epoch_loss += batch_loss
    losses.append(epoch_loss / len(train_loader))

    # Evaluate Reconstruction Performance
    train_loss = reconstruction_loss(model, train_loader, device=device)
    val_loss = reconstruction_loss(model, val_loader, device=device)
    recon_train_losses.append(train_loss)
    recon_val_losses.append(val_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {epoch_loss / len(train_loader):.4f}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")


torch.save(model.state_dict(), os.path.join("runs", "rbm_model.pth"))
np.savez(os.path.join("runs", "logs"), losses, recon_train_losses, recon_val_losses)
