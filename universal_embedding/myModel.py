import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalLinkPredModel(nn.Module):
    def __init__(self, D):
        super(UniversalLinkPredModel, self).__init__()
        self.fc1 = nn.Linear(2 * D, D)
        self.fc2 = nn.Linear(D, 1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

if __name__ == '__main__':

    D = 10
    model = UniversalLinkPredModel(D)

    input1 = torch.randn(1, D)
    input2 = torch.randn(1, D)

    output = model(input1, input2)
    print(output)
