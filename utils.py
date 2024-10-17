import torch


def mask_image(img, mask_fraction=0.5):
    mask = torch.bernoulli(torch.full(img.shape, 1 - mask_fraction)).to(img.device)
    return img * mask


def reconstruction_loss(model, data_loader, device="cpu"):
    loss = 0.0
    
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.view(-1, 784).to(device)
        masked_image = mask_image(data)
        
        _, h = model.sample_h(masked_image)
        out, _ = model.sample_v(h)
        loss += torch.mean((out - data) ** 2).item()

    return loss / len(data_loader)