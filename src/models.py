import torch.nn as nn

class MLP_celeba(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(12288, 2048),
      nn.ReLU(),
      nn.Linear(2048, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, 2)
    )

  def forward(self, x):
    return self.fc(x)
