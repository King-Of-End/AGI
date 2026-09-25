"""Минимальный прогон мира через gym без рендера.

Здесь виден весь каркас подключения разума к миру: кто держит умы, кто
заводит их при рождении и удаляет при смерти. Сегодня на этом месте
заглушка GreedyMind — верхняя отметка: так выглядит мир, в котором таксис
уже найден. Настоящий Mind, когда он появится, встанет ровно сюда —
в фабрику ниже и в ветку рождения через spawn(). Мост от агента с
forward()/mutate() к контракту act()/spawn() — world.adapters.MindAdapter.

    python3 main.py --world forage --steps 200000
    python3 main.py --mind random --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gymnasium as gym
import numpy as np

from world import list_worlds, register_all
from world.gym_env import gym_id
from world.stubs import GreedyMind, RandomMind

STUBS = {"greedy": GreedyMind, "random": RandomMind}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="forage", choices=list_worlds())
    ap.add_argument("--mind", default="greedy", choices=sorted(STUBS))
    ap.add_argument("--size", type=int, default=20)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    register_all()
    env = gym.make(gym_id(args.world), size=args.size, seed=args.seed)
    obs, info = env.reset(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    stub = STUBS[args.mind]

    # Разумы держит вызывающий — мир их не хранит и не создаёт.
    minds = {i: stub(env.unwrapped.world.action_size, rng) for i in info["ids"]}

    for _ in range(args.steps):
        actions = [minds[i].act(o) for i, o in zip(info["ids"], obs)]
        obs, _reward_unusable, terminated, truncated, info = env.step(actions)

        for parent, child in info["born"]:
            minds[child] = minds[parent].spawn(rng)
        for died in info["died"]:
            del minds[died]

        if terminated or truncated:
            break

    print(
        f"t={info['tick']} популяция={info['population']} "
        f"еда={info['food']} "
        f"поколение={int(info['generation'].max()) if info['population'] else 0}"
    )
    env.close()


if __name__ == "__main__":
    main()
