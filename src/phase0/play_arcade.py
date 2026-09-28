"""Графический вид и игра человеком. Часть 10.4, вариант --view arcade.

Терминальный вид (play.py) даёт все три обязательных вида и работает без
дисплея. Этот модуль добавляет то, чего терминал дать не может: нормальную
частоту кадров, цвет, различение двух видов предметов не одной буквой, и
режим «только сетчатка» на весь экран.

Двухслойная структура — не украшение, а условие проверяемости:

  PlaySession  — мир, темп, человек-агент. НЕ знает про arcade и про
                 дисплей, проверяется тестами на голой машине.
  PlayWindow   — тонкая оболочка: клавиши -> press()/release(), таймер ->
                 advance(dt), рисование. Логики не содержит.

Человек подключён через ТОТ ЖЕ событийный контракт, что и агент: клавиши
превращаются в события outbox, моторика мира удерживает значение до
следующего события, молчание — легальное состояние с последствиями.
Разница с терминальным режимом одна: события шлются при ИЗМЕНЕНИИ нажатий,
а не каждый кадр, — семантически это то же самое (тракт всё равно
удерживает), но outbox не засоряется повторами.
"""

from __future__ import annotations

import math
import os
from collections import deque

import numpy as np

import arcade

from .config import Config, Kind
from .events import Event, Motor
from .world import World

# Дефолтный шрифт arcade ('calibri', 'arial') на Linux не существует, и
# fallback не обязан содержать кириллицу. Регистрируем DejaVu Sans явно;
# там, где файла нет, остаёмся на дефолте — гарантий по кириллице нет.
_FONT: str | None = None
_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if os.path.exists(_DEJAVU):
    try:
        arcade.load_font(_DEJAVU)
        _FONT = "DejaVu Sans"
    except Exception:
        _FONT = None
FONT_NAME: str | tuple[str, ...] = _FONT if _FONT else ("calibri", "arial")

# --------------------------------------------------------------- палитра
COLOR_BG = (18, 20, 28)
COLOR_ARENA_BG = (13, 15, 21)
COLOR_BORDER = (88, 98, 118)
COLOR_TEXT = (206, 212, 222)
COLOR_DIM = (128, 136, 150)
COLOR_A = (222, 228, 240)      # вид A — светлый
COLOR_B = (214, 158, 64)       # вид B — тёмный, но различимый на тёмном
COLOR_OBSTACLE = (72, 76, 92)
COLOR_FOV = (66, 80, 108)
COLOR_TRAIL = (58, 68, 90)
COLOR_BODY_LOW = (220, 70, 70)
COLOR_BODY_MID = (228, 168, 62)
COLOR_BODY_OK = (94, 208, 110)
COLOR_PAUSE = (226, 88, 88)
COLOR_EMPTY_CELL = (38, 42, 52)
COLOR_HELP_BG = (10, 12, 18, 232)

STATUS_H = 36
HINT_H = 30

_KEY_TO_SESSION: dict[int, str] = {}
if hasattr(arcade, "key"):  # на случай странных сборок без key-модуля
    _KEY_TO_SESSION = {
        arcade.key.W: "w", arcade.key.UP: "w",
        arcade.key.S: "s", arcade.key.DOWN: "s",
        arcade.key.A: "a", arcade.key.LEFT: "a",
        arcade.key.D: "d", arcade.key.RIGHT: "d",
    }

MAX_SPEED = 16.0
MIN_SPEED = 0.125


class PlaySession:
    """Мир + человек-агент + темп. Без графики, проверяется без дисплея.

    Человек играет тем же контрактом, что и будущий агент: press/release
    превращаются в события outbox с t == текущий тик, мир применяет их в
    начале шага. Между событиями моторика удерживает значение — «зажатая
    клавиша» из спецификации это именно удержание трактом, а не поток
    повторов от окна.
    """

    def __init__(self, cfg: Config, speed: float = 1.0,
                 blind: bool = False) -> None:
        self.cfg = cfg
        self.world = World(cfg)
        self.speed = float(speed)
        self.blind = bool(blind)
        self.paused = False
        self.last_record = None        # TruthRecord последнего тика (для истины)
        self.trail: deque[tuple[float, float]] = deque(maxlen=1200)
        self._acc = 0.0                # накопленные (неотигранные) секунды
        self._pressed: set[str] = set()
        self._thrust = 0.0
        self._turn = 0.0

    # ------------------------------------------------ ввод человека
    def press(self, key: str) -> None:
        """Зажата клавиша движения (w/s/a/d)."""
        if key in ("w", "s", "a", "d"):
            self._pressed.add(key)
            self._emit_changed()

    def release(self, key: str) -> None:
        """Отпущена клавиша движения."""
        if key in self._pressed:
            self._pressed.discard(key)
            self._emit_changed()

    def stop(self) -> None:
        """Пробел: нейтраль по обоим моторным каналам."""
        if self._pressed:
            self._pressed.clear()
            self._emit_changed()

    def _emit_changed(self) -> None:
        """События шлются ТОЛЬКО при изменении цели.

        W+S и A+D взаимно гасятся: цель считается по одному нажатию,
        конфликт разрешается в пользу прямого хода (w, d). Мир не знает,
        откуда пришло событие — для него это обычный моторный вход.
        """
        thrust = 1.0 if "w" in self._pressed else \
            (-1.0 if "s" in self._pressed else 0.0)
        turn = 1.0 if "d" in self._pressed else \
            (-1.0 if "a" in self._pressed else 0.0)
        tick = self.world.tick
        if thrust != self._thrust:
            self.world.outbox.put(Event(tick, Motor.THRUST, thrust))
            self._thrust = thrust
        if turn != self._turn:
            self.world.outbox.put(Event(tick, Motor.TURN, turn))
            self._turn = turn

    # ------------------------------------------------ команды
    def toggle_blind(self) -> None:
        self.blind = not self.blind

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def faster(self) -> None:
        self.speed = min(MAX_SPEED, self.speed * 2.0)

    def slower(self) -> None:
        self.speed = max(MIN_SPEED, self.speed / 2.0)

    def restart(self) -> None:
        """Новый мир с тем же Config (сижив тем же: воспроизводимость)."""
        self.world = World(self.cfg)
        self.trail.clear()
        self.last_record = None
        self._acc = 0.0
        self._pressed.clear()
        self._thrust = self._turn = 0.0

    # ------------------------------------------------ время
    def advance(self, dt: float) -> int:
        """Шагнуть мир на тики, соответствующие dt реальных секунд.

        speed=1 — мир идёт в реальном времени (tick_hz тиков в секунду).
        Возвращает число сделанных тиков. Защита от спирали: если окно не
        рисовало долго, исполнить всё накопленное нельзя — лишнее сгорает,
        иначе после паузы человек получит прыжок на тысячи тиков.
        """
        if self.paused:
            return 0
        self._acc += max(0.0, dt) * self.speed
        ticks = int(self._acc * self.cfg.tick_hz)
        if ticks <= 0:
            return 0
        cap = max(4, int(self.cfg.tick_hz * self.speed * 2))
        if ticks > cap:
            ticks = cap
            self._acc = 0.0
        else:
            self._acc -= ticks / self.cfg.tick_hz

        w = self.world
        for _ in range(ticks):
            self.last_record = w.step()
            # Человек события inbox не читает, а очередь расти не должна:
            # осушаем, как это делал бы честный агент после разбора.
            w.inbox.drain()
            self.trail.append((w.body.x, w.body.y))
        return ticks


# ----------------------------------------------------------- утилиты вида
def _ray_to_rect(x: float, y: float, dx: float, dy: float,
                 arena: tuple[float, float]) -> float:
    """Расстояние от точки до границы арены вдоль луча (dx, dy)."""
    w, h = arena
    best = math.inf
    if abs(dx) > 1e-12:
        for wx in (0.0, w):
            t = (wx - x) / dx
            if t > 0:
                best = min(best, t)
    if abs(dy) > 1e-12:
        for wy in (0.0, h):
            t = (wy - y) / dy
            if t > 0:
                best = min(best, t)
    return best if math.isfinite(best) else 0.0


def _body_color(energy: float) -> tuple[int, int, int]:
    """Красный (голод) -> янтарный -> зелёный (сытость)."""
    e = min(1.0, max(0.0, energy))
    if e < 0.5:
        f = e / 0.5
        return tuple(int(a + (b - a) * f)
                     for a, b in zip(COLOR_BODY_LOW, COLOR_BODY_MID))
    f = (e - 0.5) / 0.5
    return tuple(int(a + (b - a) * f)
                 for a, b in zip(COLOR_BODY_MID, COLOR_BODY_OK))


def _lum_color(v: float) -> tuple[int, int, int]:
    """Яркость рецептора: 0 -> тёмный, ~3.1 -> белый.

    Верх шкалы — максимум фотометрии: I -> 1 при d -> 0 даёт
    L = ln((1+i_floor)/i_floor) = ln(21) = 3.04 (см. Config.d_ref/i_floor).
    """
    f = min(1.0, max(0.0, v / 3.1))
    g = int(28 + 220 * f)
    return (g, g, min(255, g + 8))


def _col_color(v: float) -> tuple[int, int, int]:
    """Цветооппонентная ось: плюс — вид A (светлый), минус — вид B.

    Эпсилон гасит шум фильтра: значение 0.001 не должно светиться так же,
    как настоящий сигнал. Пол 0.25 оставляет слабый сигнал различимым.
    """
    if abs(v) < 0.02:
        return COLOR_EMPTY_CELL
    a = min(1.0, abs(v))
    base = COLOR_A if v > 0 else COLOR_B
    f = 0.25 + 0.75 * a
    return tuple(int(c * f) for c in base)


def _lerp_box(v: float, lo: float, hi: float) -> float:
    return min(1.0, max(0.0, (v - lo) / (hi - lo))) if hi > lo else 0.0


def _draw_bar(x: float, y: float, w: float, h: float,
              values, color_fn) -> None:
    """Полоса рецепторов: n ячеек, цвет по color_fn(v)."""
    n = len(values)
    cw = w / n
    for i, v in enumerate(values):
        arcade.draw_rect_filled(arcade.LBWH(x + i * cw, y, max(1.0, cw - 1), h),
                                color_fn(float(v)))
    arcade.draw_rect_outline(arcade.LBWH(x - 2, y - 2, w + 4, h + 4),
                             COLOR_BORDER, 1)


class PlayWindow(arcade.Window):
    """Оболочка: клавиши -> сессия, таймер -> advance, рисование.

    Позиции и размеры считаются каждый кадр из текущих width/height:
    окно изменяемое, а хранить layout ради пары чисел незачем.
    """

    def __init__(self, session: PlaySession) -> None:
        super().__init__(1280, 800, "Фаза 0 — игра человеком", resizable=True)
        self.session = session
        self.background_color = COLOR_BG
        self._help = False
        self._texts: dict[str, arcade.Text] = {}

    # --------------------------------------------------------- ввод
    def on_key_press(self, symbol: int, modifiers: int) -> None:
        s = self.session
        key = _KEY_TO_SESSION.get(symbol)
        if key:
            s.press(key)
        elif symbol in (arcade.key.SPACE,):
            s.stop()
        elif symbol in (arcade.key.B,):
            s.toggle_blind()
        elif symbol in (arcade.key.P,):
            s.toggle_pause()
        elif symbol in (arcade.key.R,):
            s.restart()
        elif symbol in (arcade.key.H,):
            self._help = not self._help
        elif symbol in (arcade.key.EQUAL, arcade.key.NUM_ADD,
                        arcade.key.PLUS):
            s.faster()
        elif symbol in (arcade.key.MINUS, arcade.key.NUM_SUBTRACT):
            s.slower()
        elif symbol in (arcade.key.Q, arcade.key.ESCAPE):
            self.close()

    def on_key_release(self, symbol: int, modifiers: int) -> None:
        key = _KEY_TO_SESSION.get(symbol)
        if key:
            self.session.release(key)

    def on_update(self, dt: float) -> None:
        self.session.advance(dt)

    # --------------------------------------------------------- текст
    def _text(self, key: str, value: str, x: float, y: float,
              size: float = 13, color=COLOR_TEXT,
              anchor_x: str = "left", bold: bool = False) -> arcade.Text:
        t = self._texts.get(key)
        if t is None:
            t = arcade.Text(value, x, y, color, size, font_name=FONT_NAME,
                            anchor_x=anchor_x, bold=bold)
            self._texts[key] = t
        else:
            t.value = value
            t.position = (x, y)
            t.color = color
            t.font_size = size
            t.anchor_x = anchor_x
        return t

    # --------------------------------------------------------- кадр
    def on_draw(self) -> None:
        self.clear()
        w, h = self.width, self.height
        self._draw_status(w, h)
        self._draw_hint(w, h)
        if self.session.blind:
            self._draw_retina(40, STATUS_H + 10, w - 80,
                              h - STATUS_H - HINT_H - 20, blind=True)
        else:
            side = self._draw_world(w, h)
            self._draw_retina(16 + side + 36, STATUS_H + 8, w - 16 - side - 52,
                              h - STATUS_H - HINT_H - 16, blind=False)
        if self.session.paused:
            self._text("pause", "ПАУЗА", w - 16, h - 26, 16, COLOR_PAUSE,
                       anchor_x="right", bold=True).draw()
        if self._help:
            self._draw_help(w, h)

    # ---- статусная строка -------------------------------------------
    def _draw_status(self, w: float, h: float) -> None:
        s = self.session
        b = s.world.body
        counts = list(s.world.eaten_counts.values())
        flags = []
        if s.blind:
            flags.append("СЕТЧАТКА")
        if s.paused:
            flags.append("ПАУЗА")
        status = (f"тик {s.world.tick}   E={b.energy:.3f}   "
                  f"смертей={s.world.deaths}   "
                  f"съедено A/B={counts[0]}/{counts[1]}   "
                  f"режим {s.cfg.mode.name}/{s.cfg.difficulty}   "
                  f"x{s.speed:g}   "
                  f"событий/тик={s.world.input_rate:.0f}   "
                  f"потери={s.world.dropped_events}")
        self._text("status", status, 16, h - 24, 14).draw()
        bar_w, bar_h = 150.0, 12.0
        bx, by = w - bar_w - 16, h - 30
        arcade.draw_rect_filled(arcade.LBWH(bx, by, bar_w, bar_h),
                                COLOR_EMPTY_CELL)
        f = _lerp_box(b.energy, 0.0, s.cfg.e_max)
        arcade.draw_rect_filled(arcade.LBWH(bx, by, bar_w * f, bar_h),
                                _body_color(b.energy))
        arcade.draw_rect_outline(arcade.LBWH(bx, by, bar_w, bar_h),
                                 COLOR_BORDER, 1)
        # Подпись живёт ВНУТРИ шкалы: за правым краем окна её обрезает.
        self._text("bar", f"E {b.energy:.2f}", bx + bar_w - 6, by + 1, 11,
                   COLOR_TEXT, anchor_x="right").draw()
        if flags:
            self._text("flags", "  ".join(flags), bx - 8, by + 1, 12,
                       COLOR_PAUSE, anchor_x="right", bold=True).draw()

    def _draw_hint(self, w: float, h: float) -> None:
        hint = ("WASD/стрелки — тяга и поворот   пробел — стоп   "
                "B — только сетчатка   P — пауза   R — заново   "
                "+/− — скорость   H — помощь   Q — выход")
        self._text("hint", hint, 16, 10, 13, COLOR_DIM).draw()

    # ---- вид сверху ---------------------------------------------------
    def _draw_world(self, w: float, h: float) -> float:
        """Левая панель: арена сверху. Возвращает размер стороны квадрата."""
        s = self.session
        cfg, world = s.cfg, s.world
        side = min(w - 420, h - STATUS_H - HINT_H - 24)
        side = max(200.0, side)
        left, bottom = 16.0, HINT_H + 8
        k = side / cfg.arena[0]

        arcade.draw_rect_filled(arcade.LBWH(left, bottom, side, side),
                                COLOR_ARENA_BG)

        pts = [(left + x * k, bottom + y * k)
               for x, y in list(s.trail)[-400:]]
        if pts:
            arcade.draw_points(pts, COLOR_TRAIL, 2.0)

        body = world.body
        bx, by = left + body.x * k, bottom + body.y * k
        half = math.radians(cfg.fov_deg) / 2.0
        for dth in (-half, half):
            ang = body.theta + dth
            t = _ray_to_rect(body.x, body.y, math.cos(ang), math.sin(ang),
                             cfg.arena)
            arcade.draw_line(bx, by,
                             left + (body.x + math.cos(ang) * t) * k,
                             bottom + (body.y + math.sin(ang) * t) * k,
                             COLOR_FOV, 1)

        for ob in world.obstacles.obstacles:
            arcade.draw_circle_filled(left + ob.x * k, bottom + ob.y * k,
                                      ob.radius * k, COLOR_OBSTACLE)

        # Вид предмета = внешность, НЕ питательность: на A3 вид B ядовит,
        # но выглядит так же, как на A1. Это и есть задача игрока.
        for it in world.items.visible_items:
            color = COLOR_A if it.kind is Kind.A else COLOR_B
            arcade.draw_circle_filled(left + it.x * k, bottom + it.y * k,
                                      cfg.r_item * k, color)

        arcade.draw_circle_filled(bx, by, cfg.r_body * k,
                                  _body_color(body.energy))
        heading = 2.5 * k
        arcade.draw_line(bx, by,
                         bx + math.cos(body.theta) * heading,
                         by + math.sin(body.theta) * heading,
                         COLOR_TEXT, 2)
        arcade.draw_rect_outline(arcade.LBWH(left, bottom, side, side),
                                 COLOR_BORDER, 1)
        return side

    # ---- сетчатка ------------------------------------------------------
    def _draw_retina(self, x: float, y: float, w: float, h: float,
                     blind: bool) -> None:
        """Полоса рецепторов: L, C, sL, sC и строка истины.

        Данные — те же, что у терминального вида: проекция этого тика,
        фильтрованный устойчивый канал и ретинальные отрезки из канала
        истины (агенту они недоступны, человеку — обязательно: без них
        не увидеть расхождение восприятия и реальности).
        """
        s = self.session
        cfg, world = s.cfg, s.world
        proj = world._last_projection
        n = cfg.retina_n
        g = n // cfg.sustained_n
        sus_l = np.repeat(world.retina.sustained[:, 0], g)
        sus_c = np.repeat(world.retina.sustained[:, 1], g)

        owner = [""] * n
        rec = s.last_record
        if rec is not None:
            for it in rec.items:
                if it.retinal_span is None or not it.visible:
                    continue
                lo, hi = it.retinal_span
                for i in range(max(0, lo), min(n - 1, hi) + 1):
                    owner[i] = it.kind

        rows = [
            ("L — транзиентный, яркость", [proj.l[i] for i in range(n)], _lum_color),
            ("C — транзиентный, цвет", [proj.c[i] for i in range(n)], _col_color),
            ("sL — устойчивый, яркость", [float(v) for v in sus_l], _lum_color),
            ("sC — устойчивый, цвет", [float(v) for v in sus_c], _col_color),
            ("истина — что где лежит (агенту НЕ виден)", owner, None),
        ]

        label_h = 20.0
        bar_h = min(48.0, (h - len(rows) * label_h) / len(rows) - 12.0)
        bar_h = max(14.0, bar_h)
        cw = w / n

        def truth_color(kind: str):
            if kind == "A":
                return (170, 178, 196)
            if kind == "B":
                return (158, 116, 48)
            return COLOR_EMPTY_CELL

        top = y + h
        for idx, (label, values, color_fn) in enumerate(rows):
            ly = top - (idx + 1) * (bar_h + label_h + 14.0) + 12.0
            self._text(f"rl{idx}", label, x, ly + bar_h + 6, 12,
                       COLOR_DIM).draw()
            if color_fn is not None:
                _draw_bar(x, ly, w, bar_h, values, color_fn)
            else:
                for i in range(n):
                    arcade.draw_rect_filled(
                        arcade.LBWH(x + i * cw, ly, max(1.0, cw - 1), bar_h),
                        truth_color(values[i]))
                arcade.draw_rect_outline(
                    arcade.LBWH(x - 2, ly - 2, w + 4, bar_h + 4),
                    COLOR_BORDER, 1)

    # ---- помощь ---------------------------------------------------------
    def _draw_help(self, w: float, h: float) -> None:
        arcade.draw_rect_filled(arcade.LBWH(0, 0, w, h), COLOR_HELP_BG)
        lines = [
            ("Фаза 0 — игра человеком", 18, COLOR_TEXT),
            ("", 10, COLOR_TEXT),
            ("Цель: не умереть с голоду. E падает сама и за движение;", 15, COLOR_TEXT),
            ("еда восполняет. Вид предмета НЕ говорит, питателен ли он:", 15, COLOR_TEXT),
            ("на уровнях A2/A3 вид B бесполезен или ядовит — правило", 15, COLOR_TEXT),
            ("меняет режим B без предупреждения. Пробуйте и наблюдайте.", 15, COLOR_TEXT),
            ("", 10, COLOR_TEXT),
            ("WASD / стрелки — тяга и поворот (действие удерживается:", 15, COLOR_TEXT),
            ("    отпустили клавишу — шлётся нейтраль, тело едет по инерции)", 15, COLOR_TEXT),
            ("пробел — мгновенная нейтраль     B — только сетчатка (проверка 0.6)", 15, COLOR_TEXT),
            ("P — пауза     R — новый мир     +/− — скорость мира", 15, COLOR_TEXT),
            ("H / Q — помощь / выход", 15, COLOR_TEXT),
            ("", 10, COLOR_TEXT),
            ("Левая панель — мир сверху; серые лучи — поле зрения 120°.", 15, COLOR_TEXT),
            ("Правая панель — сетчатка: транзиентный (события) и", 15, COLOR_TEXT),
            ("устойчивый (ФНЧ ~100 мс) каналы + строка истины для сравнения.", 15, COLOR_TEXT),
            ("Играя только по сетчатке (B), вы живёте тем, что живёт агент.", 15, COLOR_TEXT),
        ]
        y = h - 60
        for text, size, color in lines:
            if text:
                self._text(f"h{size}{y}", text, 48, y, size, color).draw()
            y -= size + 12


def run(cfg: Config, speed: float = 1.0, blind: bool = False) -> None:
    """Точка входа для play.py: собрать сессию и запустить цикл arcade."""
    PlayWindow(PlaySession(cfg, speed=speed, blind=blind))
    arcade.run()
