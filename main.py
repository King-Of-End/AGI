import time

import gymnasium as gym
import torch

from agent import Agent

SIZE = 10

gym.register(id="GridWorld-v0", entry_point="env:GridWorld")

env = gym.make("GridWorld-v0", size=SIZE)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f"Используется: {device}")

model = Agent(SIZE)
model.to(device)

max_steps = 0

st = time.time()
obs, info = env.reset(seed=1)
hidden = torch.zeros(15, device=device)
for _ in range(5000):
    action, hidden = model(torch.tensor(obs.reshape(SIZE ** 2 + 1), device=device, dtype=torch.float32), hidden)
    obs, reward, terminated, truncated, info = env.step(torch.argmax(action))

    if terminated:
        max_steps = max(info['step'], max_steps)
        obs, info = env.reset()
env.close()
print(time.time() - st)
print(max_steps)