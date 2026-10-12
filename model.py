import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, n_classes=3, channels=62, samples=400, F1=8, D=2, F2=16, kernel_len=64, dropout=0.25):

        super().__init__()
        
        # Temporal Filter
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Spatial Filter
        self.depthwise = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        
        # Separable Conv
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            out = self.pool2(self.act2(self.bn3(self.separable(self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(dummy)))))))))))
            flatten_dim = out.shape[1] * out.shape[2] * out.shape[3]
            
        self.fc = nn.Linear(flatten_dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.separable(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.flatten(x)
        return self.fc(x)
