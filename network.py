import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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
