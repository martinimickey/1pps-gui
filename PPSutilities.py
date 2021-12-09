"""Utilities for 1PPS measurements."""

from __future__ import annotations
from typing import List, Tuple, Dict
from datetime import datetime
import numpy


class TimeTagGroupBase:
    def __init__(self, time: datetime):
        self.time = time

    def get_channel_tags(self, channels: List[int]) -> List[float]: ...


class TimeTagGroup(TimeTagGroupBase):
    def __init__(self, index, reference_tag: int, time: datetime, debug_data: List[str]):
        super().__init__(time)
        self.reference_tag = reference_tag
        self.channel_tags: Dict[int, int] = dict()
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
