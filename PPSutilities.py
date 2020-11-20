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
           ("clock_time", int64),
           ("tags", int64[:]),
           ("x_sum", int64),
           ("x_sum_long", int64),
           ("x2_sum", int64),
           ("denominator", int64),
           ("y_sum", int64),
           ("xy_sum", int64),
           ("last_tag", int64),
           ("position", int32),
           ("full", b1)])
class LinearClockApproximation:
    """Calculate an approximated value based on linear fitting."""
    def __init__(self, length: int, period: int, channel):
        self.max_length = length
        self.period = period
        self.channel = channel
        self.clock_time = 0
        self.tags = numpy.zeros(length, dtype=numpy.int64)
        self.x_sum = 0
        self.x_sum_long = 0
        self.x2_sum = 0
        self.denominator = 1
        self.y_sum = 0
        self.xy_sum = 0
        self.last_tag = 0
        self.position = 0
        self.full = False

    def process_tags(self, tags: numpy.array) -> int:
        index = 0
        for tag in tags:
            if tag["channel"] == self.channel:
                self.new_tag(tag["time"])
                index += 1
                continue
            break
        return index

    def new_tag(self, tag: int):
        """Add new clock tag and adjust calculated values."""
        step = tag - self.last_tag
        self.last_tag = tag
        if self.full:
            front_to_end = tag - self.tags[self.position]
            self.xy_sum += self.y_sum + self.max_length * front_to_end - self.x_sum_long * step
            self.y_sum += front_to_end - self.max_length * step
        else:
            length = self.position
            self.x_sum += length
            self.x_sum_long += length + 1
            self.x2_sum += length**2
            if length:
                self.denominator = numpy.sum(numpy.array(list(range(length + 1)), dtype=numpy.int32))**2 - (length + 1) * self.x2_sum
                self.xy_sum += self.y_sum - self.x_sum * step
                self.y_sum -= length * step
        self.tags[self.position] = tag
        self.clock_time += self.period
        self.position = (self.position + 1) % self.max_length
        self.full = self.full or self.position == 0

    def rescale_tag(self, tag: int) -> int:
        if not self.full and self.position < 2:
            return tag - self.last_tag + self.clock_time
        slope, offset = self._slope_and_offset()
        return self.clock_time - int(self.period*(tag-self.last_tag-offset)/slope)

    def _slope_and_offset(self) -> Tuple[int, int]:
        slope = (self.x_sum * self.y_sum - len(self.tags) * self.xy_sum + self.denominator//2)//self.denominator
        offset = (self.xy_sum * self.x_sum - self.y_sum * self.x2_sum + self.denominator//2)//self.denominator
        return slope, offset


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
    def __init__(self, channel: int, period: int):
        self.data = LinearClockApproximation(2000, period, channel)
        self.channel = channel
        self.period = period
        self.last_time_tag = 0

    def rescale_tag(self, tag: int) -> int:
        """Get a the tag time based on the clock input."""
        return self.data.rescale_tag(tag)

    def process_tags(self, tags: numpy.array) -> int:
        """Process a set of incoming tags."""
        return self.data.process_tags(tags)

if __name__ == "__main__":
    from numpy.random import random
    clk = LinearClockApproximation(100, 100000, 7)
    clk_ticks = [i*100000 + int(numpy.sin(2*numpy.pi*i/1000)*10000000) for i in range(1000)]
    timetags = [tag + (random()-0.5)*2000000 for tag in clk_ticks]
    from matplotlib import pyplot as plt
    corrected = list()
    for tag_, clk_tick in zip(timetags, clk_ticks):
        clk.new_tag(clk_tick)
        corrected.append(clk.rescale_tag(tag_))
    plt.scatter(range(1000), clk_ticks, marker=".")
    plt.scatter(range(1000), timetags, marker=".")
    plt.scatter(range(1000), corrected, marker=".")
    plt.show()