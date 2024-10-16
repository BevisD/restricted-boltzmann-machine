import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from network import RBM


def mask_image(img, mask_fraction=0.5):
    mask = torch.bernoulli(torch.full(img.shape, 1 - mask_fraction))
    return img * mask


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


transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_image = next(iter(train_loader))[0][0].view(-1, 784)
masked_image = mask_image(test_image, mask_fraction=0.2)

rbm = RBM(visible_units=784, hidden_units=256)
rbm.load_state_dict(torch.load('models/rbm_model.pth'))

_, hidden_representation = rbm.sample_h(masked_image)
reconstructed_image, _ = rbm.sample_v(hidden_representation)

# Show the results
display_images(test_image, masked_image, reconstructed_image)