import torch
import torch.nn.functional as F


class SimpleFCs(torch.nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(SimpleFCs, self).__init__()

        self.fc1 = torch.nn.Linear(hidden_size, 1024)
        self.fc3 = torch.nn.Linear(1024, num_labels)

    def forward(self, x):
        x = torch.nn.Dropout(0.5)(x)
        x = F.relu(self.fc1(x))
        x = torch.nn.Dropout(0.8)(x)
        x = self.fc3(x)
        return x
