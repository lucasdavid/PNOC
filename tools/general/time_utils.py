# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import time


def get_today():
  now = time.localtime()
  s = "%04d-%02d-%02d-%02dh%02dm%02ds" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
  return s


class Timer:

  def __init__(self):
    self.start_time = 0.0
    self.end_time = 0.0

    self.tik()

  def tik(self):
    self.start_time = time.time()

  def tok(self, ms=False, clear=False):
    self.end_time = time.time()

    if ms:
      duration = int((self.end_time - self.start_time) * 1000)
    else:
      duration = int(self.end_time - self.start_time)

    if clear:
      self.tik()

    return duration


class BlockTimer:
  _TRACKERS = {}

  def __init__(self, title: str = None, enabled: bool = True):
    self.title = title or "TimeTracker"
    self.enabled = enabled
    self.runs = 0
    self._time_total = 0
    self._start_e = None

  def __enter__(self):
    if self.enabled:
      self._start_e = time.time()
    return self

  def __exit__(self, exc_type, exc, exc_tb):
    if self.enabled:
      self._time_total += time.time() - self._start_e
      self._start_e = None
      self.runs += 1

  def description(self) -> str:
    # return f"Block U2PL repeated 10 times, taking 0.421 seconds on average."
    return f"Block {self.title} repeated {self.runs} times, taking {self._time_total / max(1, self.runs):.3f} seconds on average."

  @classmethod
  def scope(cls, name, enabled: bool = True):
    if name in cls._TRACKERS:
      tracker = cls._TRACKERS[name]
    else:
      tracker = cls(name)
      cls._TRACKERS[name]

    tracker.enabled = enabled
    return tracker

  @classmethod
  def trackers(cls):
    return cls._TRACKERS
