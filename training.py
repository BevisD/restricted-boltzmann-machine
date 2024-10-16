import torch
from torchvision import datasets, transforms

from network import RBM

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')


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


transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

rbm = RBM(visible_units=784, hidden_units=256)

losses = train_rbm(rbm, train_loader, eta=0.0001, epochs=20, k=5, device=device)
# torch.save(rbm.state_dict(), 'models/rbm_model.pth')
# plt.plot(losses)
# plt.show()
