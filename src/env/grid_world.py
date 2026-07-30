from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

EMPTY = 0.0
FOOD = 1.0
AGENT = 2.0

MAX_HUNGER = 10
FOOD_AMOUNT = 5

CELL = 64   # px на клетку
WALL = 20   # толщина внешних стен

COLOR_WALL  = (18, 7, 26)      # стены — почти чёрный пурпур
COLOR_BG    = (46, 18, 60)     # поле — тёмно-пурпурный
COLOR_LINE  = (62, 26, 80)     # линии сетки
COLOR_FOOD  = (96, 214, 118)   # еда — зелёная
COLOR_AGENT = (245, 245, 250)  # агент — белый
COLOR_FOOD_DARK = (46, 140, 70)   # обводка еды


def constraint(val, min_val, max_val) -> int:
    return min(max(val, min_val), max_val)


class AgentPos:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    @property
    def pos(self):
        return self.x, self.y

    @property
    def pos_np(self):
        return self.y, self.x

    def correct(self):
        self.x = max(0, min(self.x, self.size - 1))
        self.y = max(0, min(self.y, self.size - 1))

    def copy(self):
        return AgentPos(self.x, self.y, self.size)

    def __str__(self):
        return f"({self.x}, {self.y})"


class GridWorld(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 4000
    }

    def __init__(
            self,
            render_mode: str | None = None,
            size: int = 5
    ) -> None:
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")

        self.render_mode: str | None = render_mode

        self.observation_space: spaces.Box = spaces.Box(
            low=0,
            high=2,
            shape=(size, size),
            dtype=np.float32
        )

        self.action_space: spaces.Discrete = spaces.Discrete(4)

        if size <= 0:
            raise ValueError("size must be > 0")
        self.size: int = size

        self.grid: np.ndarray | None = None
        self.hunger: int | None = None
        self.agent_pos: AgentPos | None = None
        self.step_count: int = 0

        self.window = None  # arcade.Window, создаётся лениво
        self._last_frame = 0.0  # для ограничения FPS

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        options = options or {}

        self.grid = np.zeros((self.size, self.size), dtype=np.float32)
        self.step_count = 0
        self.hunger = 0

        cells = [
            (y, x)
            for y in range(self.size)
            for x in range(self.size)
        ]

        food_count = constraint(options.get('food', FOOD_AMOUNT), 0, len(cells) - 1)
        food_idx = self.np_random.choice(len(cells), size=food_count, replace=False)

        for idx in food_idx:
            y, x = cells[idx]
            self.grid[y, x] = FOOD

        empty_idx = [
            idx
            for idx, (y, x) in enumerate(cells)
            if self.grid[y, x] == EMPTY
        ]

        agent_idx = self.np_random.choice(empty_idx)
        agent_y, agent_x = cells[agent_idx]

        self.agent_pos = AgentPos(int(agent_x), int(agent_y), size=self.size)

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        self.step_count += 1

        match action:  # 0 - up, 1 - down, 2 - left, 3 - right
            case 0:
                self.agent_pos.y += 1
            case 1:
                self.agent_pos.y -= 1
            case 2:
                self.agent_pos.x -= 1
            case 3:
                self.agent_pos.x += 1
            case _:
                pass
        self.agent_pos.correct()
        if self.grid[self.agent_pos.pos_np] == FOOD:
            self.hunger = 0
            self.grid[self.agent_pos.pos_np] = EMPTY
        else:
            self.hunger += 1

        terminated = bool(self.hunger >= MAX_HUNGER)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), 0.0, terminated, False, self._get_info()

    def _get_obs(self):
        """Формирует наблюдение. Всегда возвращает копию."""
        grid = self.grid.copy()
        grid[self.agent_pos.pos_np] = AGENT
        return grid

    def _get_info(self):
        """Формирует диагностику."""
        return {
            "agent_pos": str(self.agent_pos.copy()),
            "step": self.step_count,
            "hunger": self.hunger,
        }

    def render(self):
        if self.render_mode is None:
            return None
        if self.grid is None:
            raise RuntimeError("Call reset() before render()")

        import arcade
        import time

        side = self.size * CELL + 2 * WALL

        if self.window is None:
            self.window = arcade.Window(
                side, side,
                f"GridWorld {self.size}x{self.size}",
                visible=(self.render_mode == "human"),
                vsync=False,
            )

        if self.render_mode == "human":
            self.window.dispatch_events()

        self.window.activate()
        self.window.clear(COLOR_WALL)

        # поле
        arcade.draw_rect_filled(
            arcade.XYWH(side / 2, side / 2, self.size * CELL, self.size * CELL),
            COLOR_BG,
        )

        # линии сетки
        for i in range(1, self.size):
            p = WALL + i * CELL
            arcade.draw_line(p, WALL, p, side - WALL, COLOR_LINE, 1)
            arcade.draw_line(WALL, p, side - WALL, p, COLOR_LINE, 1)

        # еда — зелёные круги с тёмной обводкой
        for y in range(self.size):
            for x in range(self.size):
                if self.grid[y, x] == FOOD:
                    cx = WALL + x * CELL + CELL / 2
                    cy = side - WALL - y * CELL - CELL / 2  # ряд 0 сверху
                    arcade.draw_circle_filled(cx, cy, CELL * 0.30, COLOR_FOOD)
                    arcade.draw_circle_outline(cx, cy, CELL * 0.30, COLOR_FOOD_DARK, 3)

        # агент — белый скруглённый квадрат
        cx = WALL + self.agent_pos.x * CELL + CELL / 2
        cy = side - WALL - self.agent_pos.y * CELL - CELL / 2
        arcade.draw_rect_filled(
            arcade.XYWH(cx, cy, CELL - 14, CELL - 14),
            COLOR_AGENT,
        )

        self.window.flip()

        if self.render_mode == "human":
            wait = 1.0 / self.metadata["render_fps"] - (time.perf_counter() - self._last_frame)
            if wait > 0:
                time.sleep(wait)
            self._last_frame = time.perf_counter()
            return None

        # rgb_array: get_image() сам переворачивает кадр в «верхний левый угол»
        return np.asarray(arcade.get_image())[:, :, :3]  # RGBA -> RGB

    def close(self):
        if self.window is not None:
            self.window.close()
            self.window = None
