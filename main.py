import time

import gymnasium as gym
import numpy as np
import torch

from agent import Agent

SIZE = 5
LIVES = 100

gym.register(id="GridWorld-v0", entry_point="env:GridWorld")

env = gym.make("GridWorld-v0", size=SIZE, render_mode='human')

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Используется: {device}")

while LIVES > 0:
    truncated = terminated = False
    obs, info = env.reset()
    model = Agent(SIZE, device=device)

    while not (truncated or terminated):
        obs, _, terminated, truncated, info = env.step(torch.argmax(model(torch.tensor(obs).to(device))))

    LIVES -= 1
    print(f"LIFE{100 - LIVES}, {info['step']=}")