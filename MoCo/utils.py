from torchvision import transforms
import torch
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import os


def save_model(save_path, model, **kwargs):
    if not os.path.exists(os.path.dirname(save_path)):
        print(f"[Warning] Directory \"{save_path}\" does not exist, creating directory.")
        os.makedirs(os.path.dirname(save_path))
    save_dict = {'model': model.state_dict()}
    for key, value in kwargs.items():
        save_dict[key] = value
    torch.save(save_dict, save_path)

def load_model(load_path):
    checkpoint = torch.load(load_path)
    return {key: checkpoint[key] for key in checkpoint.keys()}

NOISE_SCALE = 1. / 256.
def add_noise(x):
    return x + NOISE_SCALE * torch.randn_like(x)

def linear_trans(x, min_, max_):
    x_scale, x_center = x.max() - x.min(), x.min() + (x.max() - x.min()) / 2.
    y_scale, y_center = max_ - min_, min_ + (max_ - min_) / 2.
    return (x - x_center) * (y_scale / x_scale) + y_center

def linear_trans_(x):
    EPS = 1e-3
    return linear_trans(x, EPS, 1 - EPS)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(add_noise),
    transforms.Lambda(linear_trans_),
])

train_set = CIFAR10(root='./data', train=True, download=True, transform=data_transform)
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def sample_data(h, w):
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=h*w, shuffle=True)
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    fig, axes = plt.subplots(h, w, figsize=(w, h))
    for i, ax in enumerate(axes.flat):
        img = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
        ax.imshow(img)
        ax.set_title(f'{label_names[labels[i].item()]}')
        ax.axis('off')
    plt.tight_layout()
    plt.axis('off')
    plt.show()