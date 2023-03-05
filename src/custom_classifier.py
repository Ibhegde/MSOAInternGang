import torch
import torch.nn.functional as F


class SimpleFCs(torch.nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(SimpleFCs, self).__init__()

        self.fc1 = torch.nn.Linear(hidden_size, 2048)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(2048, 1024)
        self.fc3 = torch.nn.Linear(1024, num_labels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.nn.Dropout(0.2)(x)
        x = F.relu(self.fc2(x))
        x = torch.nn.Dropout(0.2)(x)
        x = self.fc3(x)
        return x
