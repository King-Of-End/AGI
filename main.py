import time

import gymnasium as gym
import numpy as np
import torch

from agent import Agent

SIZE = 5

gym.register(id="GridWorld-v0", entry_point="env:GridWorld")

env = gym.make("GridWorld-v0", size=SIZE, render_mode=None)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Используется: {device}")

model = Agent(SIZE)
model.to(device)

steps = np.empty(0)

obs, info = env.reset(seed=1)
hidden = torch.zeros(15, device=device)
for step in range(5000):
    action, hidden = model(torch.tensor(obs.reshape(SIZE ** 2 + 1), device=device, dtype=torch.float32), hidden)
    obs, reward, terminated, truncated, info = env.step(torch.argmax(action))

    if step % 100:
        print(f"step {step} | w1 norm: {model.fc1.weight.norm():.4f} | w2 norm: {model.fc2.weight.norm():.4f}")

    if terminated:
        model = Agent(SIZE)
        model.to(device)
        hidden = torch.zeros(15, device=device)
        steps = np.append(steps, info['step'])
        obs, info = env.reset()
env.close()
print(steps[-1])