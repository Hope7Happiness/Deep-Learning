from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from utils import save_model, load_model, train_set
from model import CustomResNet, MoCo, create_resnet_encoder, ClassificationHead
import yaml
from torch.utils.tensorboard import SummaryWriter

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

if config['device'] == 'cuda' and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=config['aug']['random_crop']['padding']),
    transforms.RandomHorizontalFlip() if config['aug']['random_horizontal_flip'] else lambda x: x,
    transforms.RandomApply([transforms.ColorJitter(config['aug']['color_jitter']['brightness'], config['aug']['color_jitter']['contrast'], config['aug']['color_jitter']['saturation'], config['aug']['color_jitter']['hue'])], p=config['aug']['color_jitter']['probability']),
    transforms.RandomGrayscale(p=config['aug']['random_grayscale']['probability']),
    # transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class AugmentedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        im_q = self.transform(img)
        im_k = self.transform(img)
        return im_q, im_k, target

# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
train_dataset = AugmentedDataset(train_set, train_transform)
train_loader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], shuffle=True, num_workers=config['dataloader']['num_workers'])
post_train_loader = DataLoader(train_set, batch_size=config['dataloader']['batch_size'], shuffle=True, num_workers=config['dataloader']['num_workers'])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=config['dataloader']['batch_size'], shuffle=False, num_workers=config['dataloader']['num_workers'])

encoder_q = create_resnet_encoder()
encoder_k = create_resnet_encoder()
model = MoCo(encoder_q, encoder_k, dim=config['moco']['dim'], queue_size=config['moco']['queue_size'], momentum=config['moco']['momentum'], tau=config['moco']['tau']).to(device)
save_model(f'./checkpoints/moco_epoch_0.pth', model)
print(f"Model saved at epoch 0")

optimizer = torch.optim.SGD(model.parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'])

criterion = nn.CrossEntropyLoss()

num_epochs = config['train']['num_epochs']
save_interval = config['train']['save_interval']

writer = SummaryWriter(log_dir='./logs')

def valid(model):
    classification_head = ClassificationHead(dim=config['moco']['dim'], num_classes=10).to(device)
    post_optimizer = torch.optim.SGD(classification_head.parameters(), lr=config['post_train']['optimizer']['lr'], momentum=config['post_train']['optimizer']['momentum'], weight_decay=config['post_train']['optimizer']['weight_decay'])

    for param in model.encoder_q.parameters():
        param.requires_grad = False

    for epoch in range(config['post_train']['num_epochs']):
        classification_head.train()
        pbar = tqdm(total=len(post_train_loader.dataset), desc=f"epoch {epoch+1}/{config['post_train']['num_epochs']}")
        for inputs, targets in post_train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.encoder_q(inputs)
            outputs = classification_head(features)
            loss = criterion(outputs, targets)
            post_optimizer.zero_grad()
            loss.backward()
            post_optimizer.step()
            pbar.update(inputs.size(0))
        pbar.close()

    classification_head.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(total=len(test_loader.dataset), desc="Validating Classification Head")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            features = model.encoder_q(inputs)
            outputs = classification_head(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.update(inputs.size(0))
    pbar.close()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    for param in model.encoder_q.parameters():
        param.requires_grad = True
    return avg_loss, accuracy

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.
    length = 0
    pbar = tqdm(total=len(train_loader.dataset))
    for _, (im_q, im_k, _) in enumerate(train_loader):
        length += 1
        im_q, im_k = im_q.to(device), im_k.to(device)
        
        logits, labels = model(im_q, im_k)
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.update(im_q.shape[0])
        pbar.set_description(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / length:.4f}")
        
    pbar.close()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    writer.add_scalar('Loss/train', avg_loss, epoch + 1)

    if (epoch + 1) % save_interval == 0:
        save_model(f'./checkpoints/moco_epoch_{epoch+1}.pth', model)
        print(f"Model saved at epoch {epoch+1}")
        print(f"Validating model at epoch {epoch+1}")
        test_loss, test_accuracy = valid(model)
        print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy:.2f}%")
        writer.add_scalar('Loss/valid', test_loss, epoch + 1)
        writer.add_scalar('Accuracy/valid', test_accuracy, epoch + 1)

writer.close()