"""Игра человеком: сессия без графики. Часть 10.4.

PlaySession сознательно не знает про дисплей, поэтому проверяется на
голой машине. Окно (PlayWindow) тестируется вручную под X: в автотестах
ему нечего ловить, кроме того, что уже покрыто здесь.

Контракт тот же, что у будущего агента: события в outbox, мир никогда не
спрашивает действие, молчание легально. Нарушение любого из трёх в этих
тестах — регрессия человеческого бейзлайна.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

arcade = pytest.importorskip("arcade", reason="arcade — опциональная зависимость")

from phase0.config import Config
from phase0.events import Motor
from phase0.play_arcade import MAX_SPEED, MIN_SPEED, PlaySession


def _peek_outbox(s: PlaySession):
    """Посмотреть outbox НЕ выгребая: события должны дойти до мира."""
    return [(e.channel, e.value) for e in s.world.outbox.peek()]


def test_press_emits_thrust_and_motor_holds() -> None:
    s = PlaySession(Config())
    s.press("w")
    assert (Motor.THRUST, 1.0) in _peek_outbox(s)
    # Событие адресовано текущему тику: мир применит его в начале шага.
    s.world.step()
    assert s.world.motor.thrust == 1.0


def test_release_returns_to_neutral() -> None:
    s = PlaySession(Config())
    s.press("w")
    s.world.step()
    assert s.world.motor.thrust == 1.0
    s.release("w")
    s.world.step()
    assert s.world.motor.thrust == 0.0


def test_conflicting_keys_last_direct_wins() -> None:
    """A+D не обнуляются: прямой ход (d) перекрывает левый, как в терминале.

    Отпускание d возвращает поворот к ещё зажатому a — удержание держится
    ровно до тех пор, пока зажата хоть одна клавиша направления.
    """
    s = PlaySession(Config())
    s.press("a")
    s.world.step()
    assert s.world.motor.turn == -1.0
    s.press("d")
    s.world.step()
    assert s.world.motor.turn == 1.0
    s.release("d")
    s.world.step()
    assert s.world.motor.turn == -1.0
    s.release("a")
    s.world.step()
    assert s.world.motor.turn == 0.0


def test_no_events_without_change() -> None:
    """Зажатая и не отпущенная клавиша не спамит outbox повторами."""
    s = PlaySession(Config())
    s.press("w")
    s.world.step()
    assert len(s.world.outbox) == 0
    for _ in range(5):
        s.press("w")
        s.world.step()
    assert len(s.world.outbox) == 0


def test_stop_sends_neutral_to_both_channels() -> None:
    s = PlaySession(Config())
    s.press("w")
    s.press("a")
    s.stop()
    events = _peek_outbox(s)
    assert (Motor.THRUST, 0.0) in events
    assert (Motor.TURN, 0.0) in events


def test_advance_runs_in_realtime_at_speed_one() -> None:
    s = PlaySession(Config())
    ticks = s.advance(1.0)
    assert ticks == s.cfg.tick_hz
    assert s.world.tick == ticks


def test_advance_scales_with_speed() -> None:
    s = PlaySession(Config(), speed=2.0)
    assert s.advance(0.5) == s.cfg.tick_hz
    s2 = PlaySession(Config(), speed=0.5)
    assert s2.advance(1.0) == s.cfg.tick_hz // 2


def test_advance_covers_fractional_accumulation() -> None:
    """Дробные доли тика не теряются, а копятся."""
    s = PlaySession(Config())
    assert s.advance(0.01) == 0    # 0.6 тика — копится
    assert s.advance(0.01) == 1    # 0.6 + 0.6 = 1.2 -> тик исполнился
    assert s.advance(0.005) == 0   # хвост 0.2 + 0.3 = 0.5 -> снова копится


def test_advance_caps_after_long_freeze() -> None:
    """Спираль смерти: накопленное за «зависший» час не исполняется."""
    s = PlaySession(Config())
    ticks = s.advance(3600.0)
    assert ticks <= s.cfg.tick_hz * 2 + 4


def test_pause_freezes_world() -> None:
    s = PlaySession(Config())
    s.toggle_pause()
    assert s.advance(1.0) == 0
    assert s.world.tick == 0
    s.toggle_pause()
    assert s.advance(1.0) > 0


def test_events_queued_while_paused_apply_on_resume() -> None:
    s = PlaySession(Config())
    s.toggle_pause()
    s.press("w")
    assert (Motor.THRUST, 1.0) in _peek_outbox(s)
    s.toggle_pause()
    s.advance(1.0)
    assert s.world.motor.thrust == 1.0


def test_silent_session_runs_forever() -> None:
    """Инвариант 3: молчание агента — легальное состояние."""
    s = PlaySession(Config(difficulty="A3"))
    for _ in range(10):
        s.advance(0.5)
    assert s.world.tick == 10 * 30
    assert len(s.world.inbox) == 0  # inbox осушается, очередь не растёт
    assert s.last_record is not None


def test_restart_fresh_world() -> None:
    s = PlaySession(Config())
    s.press("w")
    s.advance(1.0)
    assert s.world.tick > 0
    s.restart()
    assert s.world.tick == 0
    assert len(s.trail) == 0
    assert s.world.outbox.drain() == []
    # Удержание моторики сброшено: старое «газ в пол» не переживает рестарт.
    assert s.world.motor.thrust == 0.0


def test_speed_bounds() -> None:
    s = PlaySession(Config(), speed=1.0)
    for _ in range(10):
        s.faster()
    assert s.speed == MAX_SPEED
    for _ in range(20):
        s.slower()
    assert s.speed == MIN_SPEED


def test_blind_toggle_only_changes_view() -> None:
    s = PlaySession(Config())
    s.toggle_blind()
    assert s.blind is True
    assert len(s.world.outbox) == 0  # вид — не событие, мир об этом не знает
