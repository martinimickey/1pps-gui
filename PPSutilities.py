"""Utilities for 1PPS measurements."""

from __future__ import annotations
from typing import List, Tuple
from datetime import datetime
import numpy
from numba.types import int32, int64, b1
from numba.experimental import jitclass
import TimeTagger


@jitclass([("max_length", int64),
           ("period", int32),
           ("channel", int32),
           ("tags", int64[:]),
           ("timestamp_storage", int64[:]),
           ("channel_storage", int32[:]),
           ("storage_start", int32),
           ("storage_end", int32),
           ("clock_time", int64),
           ("clock_timestamps", int64[:]),
           ("denominator", int64),
           ("y_sum", int64),
           ("xy_sum", int64),
           ("last_tag", int64),
           ("position", int32),
           ("full", b1),
           ("offset", int64)])
class LinearClockApproximation:
    """Calculate an approximated value based on linear fitting."""

    def __init__(self, length: int, period: int, channel):
        self.max_length = length
        self.period = period
        self.channel = channel
        self.tags = numpy.zeros(length, dtype=numpy.int64)
        self.timestamp_storage = numpy.zeros(1000, dtype=numpy.int64)
        self.channel_storage = numpy.zeros(1000, dtype=numpy.int32)
        self.storage_start = 0
        self.storage_end = 0
        self.clock_time = -period
        self.clock_timestamps = numpy.zeros(3, dtype=numpy.int64)
        self.denominator: int
        self.y_sum: int
        self.xy_sum: int
        self.last_tag: int
        self.position: int
        self.full: bool
        self.offset: int
        self._reset()

    def _reset(self):
        """Reset calculated values to start values."""
        self.denominator = 1
        self.y_sum = 0
        self.xy_sum = 0
        self.last_tag = 0
        self.position = 0
        self.full = False
        self.offset = 0

    def skip(self, number_of_skipped_tags: int):
        self.clock_time += number_of_skipped_tags * self.period
        self._reset()

    def process_tags(self, tags: numpy.array, rescaled_tags: numpy.array) -> int:
        rescaled_tags_index = 0
        for tag in tags:
            if tag["type"] == 0:
                if tag["channel"] == self.channel:
                    self.new_tag(tag["time"])
                    cycle_length = self.clock_timestamps[1] - self.clock_timestamps[0]
                    while self.storage_start != self.storage_end and self.timestamp_storage[self.storage_start] < self.clock_timestamps[1]:
                        if cycle_length > 0:
                            rescaled_tags[rescaled_tags_index]["time"] = self.rescale(cycle_length)
                            rescaled_tags[rescaled_tags_index]["channel"] = self.channel_storage[self.storage_start]
                            rescaled_tags_index += 1
                        self.storage_start = self._increment_storage_pointer(self.storage_start)
                else:
                    self.timestamp_storage[self.storage_end] = tag["time"]
                    self.channel_storage[self.storage_end] = tag["channel"]
                    self.storage_end = self._increment_storage_pointer(self.storage_end)
                    if self.storage_end == len(self.channel_storage):
                        self.storage_end = 0
            else:
                if tag["type"] == 4 and tag["channel"] == self.channel:
                    self.skip(tag["missed_events"])
                    while self.storage_start != self.storage_end:
                        rescaled_tags[rescaled_tags_index]["type"] = 4
                        rescaled_tags[rescaled_tags_index]["channel"] = self.channel_storage[self.storage_start]
                        rescaled_tags_index += 1
                        self.storage_start = self._increment_storage_pointer(self.storage_start)
                else:
                    rescaled_tags[rescaled_tags_index] = tag
                    rescaled_tags[rescaled_tags_index]
                    rescaled_tags_index += 1
        return rescaled_tags_index

    def _increment_storage_pointer(self, pointer):
        pointer += 1
        if pointer == len(self.channel_storage):
            return 0
        return pointer

    def new_tag(self, tag: int):
        """Add new clock tag and adjust calculated values."""
        step = tag - self.last_tag - self.period
        self.last_tag = tag
        if self.full:
            front_to_end = tag - self.tags[self.position] - self.max_length * self.period
            self.xy_sum += -self.y_sum + step * ((self.max_length * (self.max_length + 1)) >> 1) - front_to_end * self.max_length
            self.y_sum += front_to_end - self.max_length * step
            self.offset = (((self.max_length << 1) - 1) * self.y_sum + 3 * self.xy_sum) // self.denominator
        else:
            if self.position:
                self.xy_sum += -self.y_sum + step * (((self.position + 1) * self.position) >> 1)
                self.y_sum -= self.position * step
            self.denominator = ((self.position + 1) * (self.position + 2)) >> 1
            self.offset = (((self.position << 1) + 1) * self.y_sum + 3 * self.xy_sum) // self.denominator
            if self.position + 1 == self.max_length:
                self.full = True
        self.tags[self.position] = tag
        self.clock_time += self.period
        self.position = (self.position + 1) % self.max_length
        if self.full or self.position > 1:
            self.clock_timestamps[:2] = self.clock_timestamps[1:]
            self.clock_timestamps[2] = tag + self.offset

    def rescale(self, length: int) -> int:
        return self.clock_time + (self.period * (self.timestamp_storage[self.storage_start] - self.clock_timestamps[0])) // length


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

    def add_tag(self, channel: int, timetag: int):
        self.channel_tags[channel] = timetag

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


class Clock:
    """Information on the external clock."""

    def __init__(self, channel: int, period: int, dtype: numpy.dtype, average_length=1000):
        self.average_length = average_length
        self.data = LinearClockApproximation(
            self.average_length, period, channel)
        self.channel = channel
        self.period = period
        self.last_time_tag = 0
        self.stored_clock_tags: List[numpy.array] = list()
        self._last_tags_stored = False
        self.rescaled_tags = numpy.zeros(16*period//2000, dtype=dtype)

    def rescale_tag(self, tag: int, channel: int) -> int:
        """Get a the tag time based on the clock input."""
        # print("rescale")
        return self.data.rescale_tag(tag)

    def process_tags(self, tags: numpy.array) -> numpy.ndarray:
        """Process a set of incoming tags."""
        index = self.data.process_tags(tags, self.rescaled_tags)
        return self.rescaled_tags[:index]
        # if tags[0]["channel"] != self.channel:
        #     return 0
        # clock_until = numpy.argmax(tags["channel"] != self.channel)
        # if clock_until == 0:  # all tags are clock tags
        #     self.data.print("a")
        #     self.to_storage(tags)
        #     return len(tags)
        # elif clock_until > self.average_length:
        #     self.data.print("b")
        #     start_index = clock_until - self.average_length
        #     self.data.skip(self.number_of_tags_in_storage() + start_index)
        #     self.stored_clock_tags.clear()
        #     return self.data.process_tags(tags, start_index)
        # else:
        #     self.data.print("c")
        #     self.process_storage(clock_until)
        #     return self.data.process_tags(tags)

    def to_storage(self, tags: numpy.array):
        storage = [tags]
        length = len(tags)
        tags_in_storage = self.number_of_tags_in_storage()
        for stored_tags in self.stored_clock_tags:
            if length >= self.average_length:
                break
            length += len(stored_tags)
            storage.append(stored_tags)
        self.data.skip(tags_in_storage + len(tags) - length)
        self.stored_clock_tags = storage

    def number_of_tags_in_storage(self):
        return sum([len(tags) for tags in self.stored_clock_tags])

    def process_storage(self, clock_until: int):
        if self.stored_clock_tags:
            tags_in_storage = self.number_of_tags_in_storage()
            offset = tags_in_storage + clock_until - self.average_length
            self.data.skip(offset)
            for tags in self.stored_clock_tags[::-1]:
                if len(tags) <= offset:
                    offset -= len(tags)
                    continue
                self.data.process_tags(tags, offset)
                offset = 0
            self.stored_clock_tags.clear()


if __name__ == "__main__":
    from numpy.random import random
    N_POINTS = 1000
    SIN_SIZE = 10000000
    RANDOM = 1000
    clk = LinearClockApproximation(1000, 100000, 7)
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
    plt.plot(numpy.diff(clk_ticks))
    plt.plot(numpy.diff(corrected["time"])[:-3])
    plt.show()
