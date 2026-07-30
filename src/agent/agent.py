import torch
from torch import nn, Tensor


class Agent(nn.Module):
    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(15 + grid_size ** 2), 15)
        self.fc2 = nn.Linear(15, 4)

    def forward(self, observation: Tensor, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        new_hidden_state = torch.tanh(self.fc1(torch.cat([observation, hidden_state])))

        logits = self.fc2(new_hidden_state)

        return logits, new_hidden_state
