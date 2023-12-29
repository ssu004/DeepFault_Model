import torch
from torch import nn
import torch.nn.functional as F

class WDCNN3(nn.Module):
    def __init__(self, first_kernel: int=64, n_classes: int=10) -> None:
        super(WDCNN3, self).__init__()
        self.conv_layers = nn.Sequential(
            #Conv1
            torch.nn.Conv1d(1, 16, first_kernel, stride=16, padding=24),
            torch.nn.BatchNorm1d(16),
            torch.nn.ReLU(),
            #Pool1
            torch.nn.MaxPool1d(2, 2),
            #Conv2
            torch.nn.Conv1d(16, 32, 3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            #Pool2
            torch.nn.MaxPool1d(2, 2),
            #Conv3
            torch.nn.Conv1d(32, 32, 3, stride=1, padding='same'),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            #Pool3
            torch.nn.MaxPool1d(2, 2),
            #Conv4
            torch.nn.Conv1d(32, 32, 3, stride=1, padding=0),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            #Pool4
            torch.nn.MaxPool1d(2, 2)
        )

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        self.linear_layers = nn.Sequential(
            torch.nn.Linear(32, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(100, n_classes)

        # self.reset_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.global_avg_pooling(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.linear_layers(x)
        x = self.head(x)
 
        return x

    # def _init_weight(self, m):
    #     if isinstance(m, nn.Conv1d):
    #         m.reset_parameters()
    #     elif isinstance(m, nn.BatchNorm1d):
    #         m.reset_parameters()
    #     elif isinstance(m, nn.Linear):
    #         m.reset_parameters()

    # def reset_weights(self):
    #     self.apply(self._init_weight)