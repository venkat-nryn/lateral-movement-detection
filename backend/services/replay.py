"""Replay clock for the real W2 event stream.

LANL events are historical, so "live" in the dashboard means a replay of the
real W2 events in their recorded order, on a clock the analyst controls. The
clock only maps wall time to W2 time; it never alters, reorders or invents an
event.
"""

from __future__ import annotations

import time
from typing import Callable

SPEEDS: tuple[float, ...] = (1.0, 5.0, 20.0, 60.0, 300.0)


class ReplayClock:
    """Maps wall-clock time onto the W2 timeline at a chosen speed."""

    def __init__(
        self,
        start: float,
        end: float,
        *,
        speed: float = 1.0,
        playing: bool = True,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if end < start:
            raise ValueError("end must not be earlier than start")
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        self.start = float(start)
        self.end = float(end)
        self._now = now
        self._speed = float(speed)
        self._playing = playing
        self._base_position = self.start
        self._base_wall = now()
        self.generation = 0

    def position(self) -> float:
        if not self._playing:
            return self._base_position
        elapsed = (self._now() - self._base_wall) * self._speed
        return min(self.end, self._base_position + elapsed)

    def _rebase(self) -> None:
        self._base_position = self.position()
        self._base_wall = self._now()

    def play(self) -> None:
        if self.position() >= self.end:
            self.seek(self.start)
        self._rebase()
        self._playing = True

    def pause(self) -> None:
        self._rebase()
        self._playing = False

    def seek(self, position: float) -> None:
        self._base_position = min(max(float(position), self.start), self.end)
        self._base_wall = self._now()
        self.generation += 1

    def set_speed(self, speed: float) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {SPEEDS}")
        self._rebase()
        self._speed = float(speed)

    def state(self) -> dict:
        position = self.position()
        return {
            "position": position,
            "start": self.start,
            "end": self.end,
            "speed": self._speed,
            "speeds": list(SPEEDS),
            "playing": self._playing and position < self.end,
            "finished": position >= self.end,
            "generation": self.generation,
            "progress": (position - self.start) / (self.end - self.start) if self.end > self.start else 1.0,
        }
