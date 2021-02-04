"""Utilities for 1PPS measurements."""

from __future__ import annotations
from typing import List, Tuple
from datetime import datetime
import numpy
from numba.types import int32, int64, b1
from numba.experimental import jitclass


@jitclass([("max_length", int64),
           ("period", int32),
           ("channel", int32),
           ("tags", int64[:]),
           ("timestamp_storage", int64[:]),
           ("channel_storage", int32[:]),
           ("storage_start", int32),
           ("storage_end", int32),
           ("clock_time", int64),
           ("_clock_timestamps", int64[:]),
           ("_y_sum_factor", int64),
           ("_period_x_denominator", int64),
           ("_step", int64),
           ("_denominator", int64),
           ("y_sum", int64),
           ("xy_sum", int64),
           ("last_tag", int64),
           ("position", int32),
           ("full", b1),
           ("slope", int64),
           ("offset", int64),
           ("print_active", b1),
           ("_reset_done", b1)])
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
        self._clock_timestamps = numpy.zeros(3, dtype=numpy.int64)
        self._denominator: int
        self.y_sum: int
        self.xy_sum: int
        self._y_sum_factor: int
        self._period_x_denominator: int
        self._step: int
        self.last_tag: int
        self.position: int
        self.full: bool
        self.slope: int
        self.offset: int
        self.print_active = True
        self._reset_done = False
        self._reset()

    def _reset(self):
        """Reset calculated values to start values."""
        self._denominator = 1
        self.y_sum = 0
        self.xy_sum = 0
        self.last_tag = 0
        self.position = 0
        self.full = False
        self.slope = self.period
        self.print("reset")
        self._period_x_denominator = self.period
        self.offset = 0
        self._reset_done = True

    def skip(self, number_of_skipped_tags: int):
        self.clock_time += number_of_skipped_tags * self.period
        if not self._reset_done:
            self._reset()

    def process_tags(self, tags: numpy.array, rescaled_tags: numpy.array) -> int:
        rescaled_tags_index = 0
        for i in range(len(tags)):
            if tags[i]["channel"] == self.channel:
                self.new_tag(tags[i]["time"])
                cycle_length = self._clock_timestamps[1] - self._clock_timestamps[0]
                # print(self.timestamp_storage[self.storage_start], self._clock_timestamps[1])
                while self.storage_start != self.storage_end and self.timestamp_storage[self.storage_start] < self._clock_timestamps[1]:
                    if cycle_length > 0:
                        rescaled_tags[rescaled_tags_index]["time"] = self.rescale(cycle_length)
                        rescaled_tags[rescaled_tags_index]["channel"] = self.channel_storage[self.storage_start]
                        rescaled_tags_index += 1
                    # print("take out")
                    self.storage_start += 1
                    if self.storage_start == len(self.channel_storage):
                        self.storage_start = 0
            else:
                # break
                # print("put in", self.storage_start, self.storage_end)
                self.timestamp_storage[self.storage_end] = tags[i]["time"]
                self.channel_storage[self.storage_end] = tags[i]["channel"]
                self.storage_end += 1
                if self.storage_end == len(self.channel_storage):
                    self.storage_end = 0
        return rescaled_tags_index

    # def process_tags_new(self, tags: numpy.array):
    #     for tag in tags:
    #         if tag["channel"] == self.channel:
    #             self.new_tag(tag["time"])
    #         else:
    #             tag["time"] = self.rescale_tag(tag["time"])

    def new_tag(self, tag: int):
        """Add new clock tag and adjust calculated values."""
        self._reset_done = False
        self._step = tag - self.last_tag - self.period
        self.last_tag = tag
        if self.full:
            front_to_end = tag - self.tags[self.position] - self.max_length * self.period
            self.xy_sum += -self.y_sum + self.max_length * ((self._step * (self.max_length + 1) >> 1) - front_to_end)
            self.y_sum += front_to_end - self.max_length * self._step
            self._y_sum_factor = (self.y_sum * (self.max_length - 1)) << 1
            self.slope = self._period_x_denominator + 3 * ((self.xy_sum << 2) + self._y_sum_factor)
            self.offset = self.period*((2 * self.max_length - 1) * self._y_sum_factor + 6 * (self.max_length - 1) * self.xy_sum) - (self.slope >> 1)
        else:
            # Here, self.position determines the size of the valid buffer
            if self.position:
                self.xy_sum += -self.y_sum + self._step * ((self.position + 1) * self.position) // 2
                self.y_sum -= self.position * self._step
            self._denominator = (self.position + 1) * (self.position ** 2 - 1)
            self._period_x_denominator = self.period * self._denominator
            self.slope = self._period_x_denominator + 12 * self.xy_sum + 6 * self.position * self.y_sum
            self.offset = self.period * (2 * self.position * (2 * self.position + 1) * self.y_sum + 6 * self.position * self.xy_sum) - (self.slope >> 1)
            if self.position + 1 == self.max_length:
                self.full = True
        self.tags[self.position] = tag
        self.clock_time += self.period
        self.position = (self.position + 1) % self.max_length
        # print(tag - self.offset//self.slope)
        self._clock_timestamps[:2] = self._clock_timestamps[1:]
        self._clock_timestamps[2] = tag - self.offset//self.slope

    def print(self, *text):
        return
        if self.print_active:
            print(*text)

    def rescale(self, length: int) -> int:
        return self.clock_time + (self.period * (self.timestamp_storage[self.storage_start] - self._clock_timestamps[0])) // length
    #     return self.clock_time - int((self._clock_timestamps - tag) / (self._clock_timestamps - self._previous_clock_timestamps))
        # def rescale_tag(self, tag: int) -> int:
        #     # if self._period_x_denominator == 5000000:
        #     #     print("STOP!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     #     self.print_active = False
        #     # self.print(self.position, self._period_x_denominator, self.slope, (self._period_x_denominator * (tag-self.last_tag)-self.offset) //
        #     #            self.slope - ((self._period_x_denominator//self.slope) * (tag-self.last_tag)-self.offset//self.slope))
        #     # if not self.full and self.position < 2:
        #     #     return self.clock_time + (tag-self.last_tag)
        #     # else:
        #     return self.clock_time  # + (self._period_x_denominator * (tag-self.last_tag)-self.offset)//self.slope


class TimeTagGroup:
    def __init__(self, index, reference_tag: int, time: datetime, debug_data: List[str]):
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

    def get_channel_tags(self, channels: List[int]):
        return [self.channel_tags[channel] - self.reference_tag if channel in self.channel_tags else numpy.nan for channel in channels]


class MissingTimeTagGroups:
    def __init__(self, periods: int, unused_tags):
        self.periods = periods
        self.unused_tags = unused_tags


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
    RANDOM = 2000000
    clk = LinearClockApproximation(100, 100000, 7)
    clk_ticks = [i*100000 + int(numpy.sin(2*numpy.pi*i/1000)*SIN_SIZE)
                 for i in range(N_POINTS)]
    # timetags = [int(tag + (random()-0.5)*RANDOM) for tag in clk_ticks]
    timetags = [tag + 5000 for tag in clk_ticks]
    stream = [(tag, 7) for tag in clk_ticks] + [(tag, 1) for tag in timetags]
    stream.sort()

    from matplotlib import pyplot as plt

    tags = numpy.zeros([2*N_POINTS], dtype=numpy.dtype({'names': ['type', 'missed_events', 'channel', 'time'],
                                                        'formats': ['u1', '<u2', '<i4', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}, align=True))
    a, b = tuple(zip(*stream))
    tags["time"] = a
    tags["channel"] = b
    corrected = numpy.zeros([N_POINTS*10], dtype=numpy.dtype({'names': ['type', 'missed_events', 'channel', 'time'],
                                                              'formats': ['u1', '<u2', '<i4', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}, align=True))
    clk.process_tags(tags, corrected)
    # for tag_, clk_tick in zip(timetags, clk_ticks):
    #     clk.new_tag(clk_tick)
    #     corrected.append(clk.rescale_tag(tag_))
    timetags.sort()
    plt.scatter(range(N_POINTS), clk_ticks, marker=".")
    plt.scatter(range(N_POINTS), timetags, marker=".")
    plt.scatter(range(N_POINTS), corrected[:N_POINTS]["time"], marker=".")

    plt.figure()
    plt.scatter(range(N_POINTS-2), numpy.diff(corrected[:(N_POINTS-1)]["time"]))
    plt.show()
