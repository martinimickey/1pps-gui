"""Utilities for 1PPS measurements."""

from __future__ import annotations
from typing import List
from datetime import datetime
import numpy
from numba.types import int32, int64, b1
from numba.experimental import jitclass


@jitclass([("max_length", int32),
           ("period", int64),
           ("channel", int32),
           ("clock_tags", int64[:]),
           ("delayed_clock_tag", int64),
           ("timestamp_storage", int64[:]),
           ("channel_storage", int32[:]),
           ("storage_start", int32),
           ("storage_end", int32),
           ("clock_time", int64),
           ("clock_timestamps", int64[:]),
           ("y_sum", int64),
           ("xy_sum", int64),
           ("i_sum", int64),
           ("last_tag", int64),
           ("clock_tag_pointer", int32),
           ("full", b1)])
class LinearClockApproximation:
    """Calculate an approximated value based on linear fitting."""

    def __init__(self, length: int, period: int, channel):
        self.max_length = length
        self.period = period
        self.channel = channel
        self.clock_tags = numpy.zeros(length, dtype=numpy.int64)
        self.delayed_clock_tag = -1
        self.timestamp_storage = numpy.zeros(1000, dtype=numpy.int64)
        self.channel_storage = numpy.zeros(1000, dtype=numpy.int32)
        self.storage_start = 0
        self.storage_end = 0
        self.clock_time = -period
        self.clock_timestamps = numpy.zeros(3, dtype=numpy.int64)
        self.y_sum: int
        self.xy_sum: int
        self.i_sum: int
        self.last_tag: int
        self.clock_tag_pointer: int
        self.full: bool
        self._reset()

    def _reset(self):
        """Reset calculated values to start values."""
        self.i_sum = 0
        self.y_sum = 0
        self.xy_sum = 0
        self.last_tag = 0
        self.clock_tag_pointer = 0
        self.full = False
        self.delayed_clock_tag = -1

    def _skip(self, number_of_skipped_tags: int, reset: bool):
        self.clock_time += number_of_skipped_tags * self.period
        if reset:
            self._reset()

    def process_tags(self, tags: numpy.array, rescaled_tags: numpy.array) -> int:
        rescaled_tags_index = 0
        for tag in tags:
            if tag["type"] == 0:
                if tag["channel"] == self.channel:
                    if self.delayed_clock_tag >= 0:
                        self._process_clock_tag()
                        if self.storage_end != self.storage_start:
                            rescaled_tags_index = self._rescale(rescaled_tags, rescaled_tags_index)
                    else:
                        self._skip(1, reset=False)
                    self.delayed_clock_tag = tag["time"]
                else:
                    self.timestamp_storage[self.storage_end] = tag["time"]
                    self.channel_storage[self.storage_end] = tag["channel"]
                    self.storage_end = self._increment_storage_pointer(self.storage_end)
                    if self.storage_end == len(self.channel_storage):
                        self.storage_end = 0
            else:
                if tag["type"] == 4 and tag["channel"] == self.channel:
                    self.clock_timestamps[:2] = self.clock_timestamps[1:]
                    rescaled_tags_index = self._rescale(rescaled_tags, rescaled_tags_index)
                    self._skip(tag["missed_events"], reset=True)
                else:
                    rescaled_tags[rescaled_tags_index] = tag
                    rescaled_tags_index += 1
        return rescaled_tags_index

    def _increment_storage_pointer(self, pointer):
        pointer += 1
        if pointer == len(self.channel_storage):
            return 0
        return pointer

    def _process_clock_tag(self):
        """Add new clock tag and adjust calculated values."""
        if self.storage_start != self.storage_end:
            if self.full or self.clock_tag_pointer > 1:
                length = self.max_length if self.full else self.clock_tag_pointer
                offset = (((length << 1) - 1) * self.y_sum + 3 * self.xy_sum) // self.i_sum
                # print(offset, self.xy_sum)
                self.clock_timestamps[:2] = self.clock_timestamps[1:]
                self.clock_timestamps[2] = self.last_tag + offset
            else:
                self.clock_timestamps[2] = self.last_tag
        tag = self.delayed_clock_tag
        step = tag - self.last_tag - self.period
        self.last_tag = tag
        if self.full:
            front_to_end = tag - self.clock_tags[self.clock_tag_pointer] - self.max_length * self.period
            self.xy_sum += -self.y_sum + step * self.i_sum - front_to_end * self.max_length  # order is crucial here: First calculate xy_sum, then y_sum!
            self.y_sum += front_to_end - self.max_length * step
        else:
            self.xy_sum += -self.y_sum + step * self.i_sum
            self.y_sum -= self.clock_tag_pointer * step
            self.i_sum += self.clock_tag_pointer + 1
        self.clock_tags[self.clock_tag_pointer] = tag
        self.clock_time += self.period
        self.clock_tag_pointer += 1
        if self.clock_tag_pointer == self.max_length:
            self.clock_tag_pointer = 0
            self.full = True

    def _rescale(self, rescaled_tags: numpy.ndarray, rescaled_tags_index) -> int:
        cycle_length = self.clock_timestamps[1] - self.clock_timestamps[0]
        while self.storage_start != self.storage_end and self.timestamp_storage[self.storage_start] < self.clock_timestamps[1]:
            if cycle_length > 0:
                rescaled_tags[rescaled_tags_index]["type"] = 0
                rescaled_tags[rescaled_tags_index]["missed_events"] = 0
                rescaled_tags[rescaled_tags_index]["time"] = self.clock_time + (self.period * (self.timestamp_storage[self.storage_start] - self.clock_timestamps[0])) // cycle_length
                rescaled_tags[rescaled_tags_index]["channel"] = self.channel_storage[self.storage_start]
                rescaled_tags_index += 1
            self.storage_start = self._increment_storage_pointer(self.storage_start)
        return rescaled_tags_index


class TimeTagGroupBase:
    def __init__(self, time: datetime):
        self.time = time

    def get_channel_tags(self, channels: List[int]) -> List[float]: ...


class TimeTagGroup(TimeTagGroupBase):
    def __init__(self, index, reference_tag: int, time: datetime, debug_data: List[str]):
        super().__init__(time)
        self.reference_tag = reference_tag
        self.channel_tags = dict()
        self.time = time
        self.index = index
        self.debug_data = debug_data

    def add_tag(self, tag: TimeTag):
        self.channel_tags[tag.channel] = tag.time

    def get_missing_channels(self, channels: List[int]):
        missing = list(channels)
        try:
            for channel in self.channel_tags:
                missing.remove(channel)
        except ValueError:
            pass
        return missing

    def get_channel_tags(self, channels: List[int]) -> List[float]:
        return [self.channel_tags[channel] - self.reference_tag if channel in self.channel_tags else numpy.nan for channel in channels]


class MissingTimeTagGroup(TimeTagGroupBase):

    def get_channel_tags(self, channels: List[int]) -> List[float]:
        return [numpy.nan] * len(channels)


class TimeTag:
    def __init__(self, tag) -> None:
        self.time: int = tag["time"]
        self.channel: int = tag["channel"]

    def __repr__(self):
        return f"({self.channel}: {self.time})"


class Clock:
    """Information on the external clock."""

    def __init__(self, channel: int, period: int, dtype: numpy.dtype, average_length=1000):
        self.average_length = average_length
        self.data = LinearClockApproximation(
            self.average_length, period, channel)
        self.rescaled_tags = numpy.zeros(16*period//2000, dtype=dtype)

    def process_tags(self, tags: numpy.array) -> numpy.ndarray:
        """Process a set of incoming tags."""
        index = self.data.process_tags(tags, self.rescaled_tags)
        return self.rescaled_tags[:index]


if __name__ == "__main__":
    from numpy.random import random
    N_POINTS = 1000
    SIN_SIZE = 10000000
    RANDOM = 0
    clk = LinearClockApproximation(2, 100001, 7)
    clk_ticks = [i*100000 + int((random()-0.5)*RANDOM) for i in range(N_POINTS)]
    timetags = [i*100000 + 5000 for i in range(N_POINTS)]
    stream = [(tag, 7) for tag in clk_ticks] + [(tag, 1) for tag in timetags]
    stream.sort()

    from matplotlib import pyplot as plt

    tags = numpy.zeros([2*N_POINTS], dtype=numpy.dtype({'names': ['type', 'missed_events', 'channel', 'time'],
                                                        'formats': ['u1', '<u2', '<i4', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}, align=True))
    a, b = tuple(zip(*stream))
    tags["time"] = a
    tags["channel"] = b
    corrected = numpy.zeros([N_POINTS], dtype=numpy.dtype({'names': ['type', 'missed_events', 'channel', 'time'],
                                                           'formats': ['u1', '<u2', '<i4', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}, align=True))
    clk.process_tags(tags, corrected)

    plt.figure()
    plt.plot(numpy.diff(clk_ticks) - 100000)
    plt.plot(-(numpy.diff(corrected["time"])[:-4] - 100000))
    plt.show()
