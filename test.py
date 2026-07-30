import random
import gymnasium as gym
import torch

from agent import Agent

SIZE = 10

gym.register(id="GridWorld-v0", entry_point="env:GridWorld")

env = gym.make("GridWorld-v0", render_mode="human", size=SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется: {device}")

model = Agent(SIZE)
model.to(device)

obs, info = env.reset(seed=1)
hidden = torch.zeros(15, device=device)
for _ in range(5000):
    action, hidden = model(torch.tensor(obs.reshape(SIZE ** 2), device=device), hidden)
    obs, reward, terminated, truncated, info = env.step(torch.argmax(action))

    if terminated:
        print(info)
        obs, info = env.reset()
env.close()
