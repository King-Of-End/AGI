import torch
from torch import nn, Tensor

from .layer import Layer

class Agent(nn.Module):
    def __init__(self, grid_size: int, device) -> None:
        super().__init__()
        self.fc1 = Layer(int(15 + grid_size ** 2 + 1), 15,device=device)
        self.fc2 = Layer(15, 4 + 15, device=device)
        self.hidden = torch.zeros(15, device=device)

        self.grid_size = grid_size

        self.device = device

    @torch.no_grad()
    def forward(self, observation: Tensor) -> Tensor:
        model_input = torch.cat([observation, self.hidden])

        inb_hidden_state = torch.tanh(self.fc1(model_input))

        out = torch.tanh(self.fc2(inb_hidden_state))

        logits = out[:4]
        self.hidden = out[4:]

        return logits

    def mutate(self) -> 'Agent':
        new_agent = Agent(self.grid_size, self.device)
        new_agent.fc1 = self.fc1.mutate()
        new_agent.fc2 = self.fc2.mutate()
        return new_agent
