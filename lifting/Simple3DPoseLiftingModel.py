import torch.nn as nn


class Simple3DPoseLiftingModel(nn.Module):
    """model architecture for 2D-to-3D pose lifting"""

    def __init__(self, num_keypoints=133):
        super(Simple3DPoseLiftingModel, self).__init__()
        self.upscale=nn.Linear(num_keypoints * 2, 1024)
        self.fc1=nn.Linear(1024, 1024)
        self.bn1=nn.BatchNorm1d(1024)
        self.fc2=nn.Linear(1024, 1024)
        self.bn2=nn.BatchNorm1d(1024)
        self.fc3=nn.Linear(1024, 1024)
        self.bn3=nn.BatchNorm1d(1024)
        self.fc4=nn.Linear(1024, 1024)
        self.bn4=nn.BatchNorm1d(1024)
        self.outputlayer=nn.Linear(1024, num_keypoints * 3)
        self.dropout=nn.Dropout(p=0.5)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.upscale(x)

        x1=self.dropout(self.relu(self.bn1(self.fc1(x))))
        x1=self.dropout(self.relu(self.bn2(self.fc2(x1))))
        x=x + x1

        x1=self.dropout(self.relu(self.bn3(self.fc3(x))))
        x1=self.dropout(self.relu(self.bn4(self.fc4(x1))))
        x=x + x1

        x=self.outputlayer(x)
        return x
