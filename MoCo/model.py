import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += res
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, dim=128):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * 4 * 4, dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def create_resnet_encoder():
    return CustomResNet(dim=128)

class MoCo(nn.Module):
    def __init__(self, encoder_q, encoder_k, dim=128, queue_size=2000, momentum=0.999, tau=0.07):
        super(MoCo, self).__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.tau = tau
        self.dim = dim
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0
        print(f"MoCo Info: queue size {queue_size}, momentum {momentum}, tau {tau}, dim {dim}.")
        print(f"MoCo total params: {sum(p.numel() for p in self.parameters())}")

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _push(self, keys):
        batch_size = keys.shape[0]
        assert self.queue_size % batch_size == 0, f"queue size should be divisible by batch size, got {self.queue_size} and {batch_size}."
        if keys.shape == (batch_size, self.dim):
            self.queue[:, self.queue_ptr : self.queue_ptr + batch_size] = keys.transpose(0, 1)
        elif keys.shape == (self.dim, batch_size):
            self.queue[:, self.queue_ptr : self.queue_ptr + batch_size] = keys
        else:
            raise ValueError(f"Invalid shape of keys: {keys.shape}, expected {(batch_size, self.dim)} or {(self.dim, batch_size)} to match the queue size.")
        self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size

    def forward(self, im_q, im_k):
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
            self._push(k)
        
        l_pos = torch.bmm(q.view(q.shape[0], 1, -1), k.view(k.shape[0], -1, 1)).squeeze(-1)
        l_neg = torch.mm(q, self.queue.clone().detach())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return logits, labels
    
class ClassificationHead(nn.Module):
    def __init__(self, dim=128, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)