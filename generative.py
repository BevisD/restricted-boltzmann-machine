import torch
import numpy as np

from network import RBM
import imageio

def generate_new_image(rbm, num_iterations=100, device='cpu'):
    images = torch.zeros(num_iterations, rbm.visible_units)

    hidden_units = torch.bernoulli(torch.rand(1, rbm.hidden_units)).to(device)

    for i in range(num_iterations):
        visible_probs, visible_units = rbm.sample_v(hidden_units)

        hidden_probs, hidden_units = rbm.sample_h(visible_units)
        images[i] = visible_probs

    return images.detach().cpu().numpy()


model = RBM(visible_units=784, hidden_units=128)
model.load_state_dict(torch.load('runs/rbm_model.pth'))

images = generate_new_image(model, num_iterations=600)

frames = []
for i in range(0,images.shape[0], 4):
    frame = (images[i].reshape(28, 28) * 255).astype(np.uint8)
    frames.append(frame)

# Save the frames as a GIF
imageio.mimsave('rbm_evolution.gif', frames, fps=30)
