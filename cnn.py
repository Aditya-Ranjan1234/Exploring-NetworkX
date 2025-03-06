import torch
import networkx as nx
import matplotlib.pyplot as plt
from torchviz import make_dot
import torch.nn as nn
import torch.nn.functional as F
import pydot

class CNNModel(nn.Module):
    def __init__(self, num_classes=149):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.3)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.35)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.4)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop5 = nn.Dropout(0.5)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.drop6 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = F.relu(self.bn4(self.conv7(x)))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        x = self.drop4(x)

        x = F.relu(self.bn5(self.conv9(x)))
        x = F.relu(self.conv10(x))
        x = self.pool5(x)
        x = self.drop5(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop6(x)
        x = self.fc2(x)
        
        return x


model = CNNModel(num_classes=149)
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))

dot = pydot.graph_from_dot_data(dot.source)[0]  
G = nx.nx_pydot.from_pydot(dot)  

plt.figure(figsize=(12, 8))
nx.draw(G, with_labels=True, node_size=100, font_size=10, font_color="red")
plt.show()
