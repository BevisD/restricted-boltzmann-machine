import torch
from torchvision import datasets, transforms
from torch.nn.functional import cross_entropy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from network import RBM
from utils import mask_image

colors = [
    (0, 0, 1),
    (0, 0, 0),
    (1, 0, 0)
]

cmap_name = 'blue_black_red'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


def display_images(original, masked, reconstructed):
    original = original.cpu().detach()
    masked = masked.cpu().detach()
    reconstructed = reconstructed.cpu().detach()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    for ax in axs.flatten():
        ax.axis('off')

    axs[0].imshow(original.view(28, 28), cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('Original Image')
    axs[1].imshow(masked.view(28, 28), cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Masked Image')
    axs[2].imshow(reconstructed.view(28, 28), cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('Reconstructed Image')
    plt.tight_layout()
    plt.savefig("figures/inference.png", dpi=200)

def display_weights(model):
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axs.flatten()):
        states = model.W[:, i].view(28, 28).detach().numpy()
        ax.imshow(states, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("figures/weights.png", dpi=200)


# Inference on test image
transform = transforms.Compose([transforms.ToTensor()])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

test_image = next(iter(test_loader))[0][0].view(-1, 784)
masked_image = mask_image(test_image, mask_fraction=0.4)

model = RBM(visible_units=784, hidden_units=128)
model.load_state_dict(torch.load('runs/rbm_model.pth'))

_, hidden_representation = model.sample_h(masked_image)
reconstructed_image, _ = model.sample_v(hidden_representation)

display_images(test_image, masked_image, reconstructed_image)
display_weights(model)
