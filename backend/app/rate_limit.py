"""Sliding-window request limiter for NVIDIA NIM (40 requests/minute on the free tier).

One process-wide window shared by every NVIDIA call — direct answers, query
rewrites, and each ADK model turn — so bursts wait here instead of getting 429s.
"""
import threading
import time
from collections import deque

from . import config

WINDOW_SECONDS = 60.0


class RpmLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = WINDOW_SECONDS / rpm  # pace requests evenly; bursts trigger 429s
        self._stamps: deque = deque()
        self._last = float("-inf")
        self._lock = threading.Lock()

    def _prune(self, now: float):
        while self._stamps and now - self._stamps[0] >= WINDOW_SECONDS:
            self._stamps.popleft()

    def acquire(self):
        """Block until one more request is allowed: at most `rpm` in any 60 seconds,
        and no two closer together than the pacing interval."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                pace_wait = self._last + self.interval - now
                window_wait = (WINDOW_SECONDS - (now - self._stamps[0])
                               if len(self._stamps) >= self.rpm else 0.0)
                wait = max(pace_wait, window_wait)
                if wait <= 0:
                    self._stamps.append(now)
                    self._last = now
                    return
            time.sleep(wait)

    def used(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return len(self._stamps)


limiter = RpmLimiter(config.NVIDIA_RPM_LIMIT)
