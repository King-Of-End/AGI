import torch
from torch import nn, Tensor

LEARNING_RATE = 1e-3

class Agent(nn.Module):
    def __init__(self, grid_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(int(15 + grid_size ** 2 + 1), 15)
        self.fc2 = nn.Linear(15, 4 + 15)

        self.lr = LEARNING_RATE

    @torch.no_grad()
    def forward(self, observation: Tensor, hidden_state: Tensor) -> tuple[Tensor, Tensor]:
        model_input = torch.cat([observation, hidden_state])

        inb_hidden_state = torch.tanh(self.fc1(model_input))

        out = self.fc2(inb_hidden_state)

        delta_w1_oja = self.oja(model_input, inb_hidden_state, self.fc1.weight.data)
        self.fc1.weight.data += delta_w1_oja

        delta_w2_oja = self.oja(inb_hidden_state, out, self.fc2.weight.data)
        self.fc2.weight.data += delta_w2_oja

        logits = out[:4]
        new_hidden_state = out[4:]

        return logits, new_hidden_state

    @torch.no_grad()
    def hebb(self, layer_input, layer_output):
        delta_w = self.lr * torch.outer(layer_input, layer_output)
        delta_w = delta_w.T
        return delta_w

    @torch.no_grad()
    def oja(self, layer_input, layer_output, layer_weights):
        delta_w = self.lr * (torch.outer(layer_output, layer_input) - torch.outer(layer_output, layer_output) @ layer_weights)
        return delta_w