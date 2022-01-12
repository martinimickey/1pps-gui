"""Utilities for 1PPS measurements."""

from __future__ import annotations
from typing import List, Optional, Dict
from datetime import datetime
from abc import ABC, abstractmethod, abstractproperty
import numpy
import TimeTagger


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


class AbstractTimeTaggerProxy(ABC):
    def __init__(self):
        self._tagger: Optional[TimeTagger.TimeTaggerBase] = None

    def get_tagger(self):
        if not self._tagger:
            raise ValueError("No tagger connected")
        return self._tagger

    @abstractmethod
    def get_id(self): str

    @abstractmethod
    def get_resolution(self, channel: int) -> str: ...

    @abstractmethod
    def is_channel_allowed(self, channel: int) -> bool: ...

    @abstractmethod
    def _connect(self, **kwargs) -> TimeTagger.TimeTaggerBase: ...

    def connect(self, **kwargs):
        self._tagger = self._connect(**kwargs)

    def disconnect(self):
        if self._tagger:
            TimeTagger.freeTimeTagger(self._tagger)
            self._tagger = None

    def allows_setters(self): return True


class USBTimeTaggerProxy(AbstractTimeTaggerProxy):
    def __init__(self, serial):
        self.__serial = serial
        self.__allowed_channels = list()

    def get_id(self):
        return self.__serial + " (USB)"

    def get_resolution(self, channel: int):
        channel = abs(channel)
        if channel in self._tagger.getChannelList(type=TimeTagger.ChannelEdge.StandardRising):
            return "Standard"
        elif channel in self._tagger.getChannelList(type=TimeTagger.ChannelEdge.HighResRising):
            return "HighRes"
        return ""

    def is_channel_allowed(self, channel: int):
        return channel in self.__allowed_channels

    def _connect(self, resolution="", **kwargs):
        tagger = TimeTagger.createTimeTagger(self.__serial, TimeTagger.Resolution[resolution] if resolution else TimeTagger.Resolution.Standard)
        self.__allowed_channels = tagger.getChannelList()
        return tagger


class NetworkTimeTaggerProxy(AbstractTimeTaggerProxy):
    def __init__(self, address: str):
        self.__address = address
        self.__info = TimeTagger.getTimeTaggerServerInfo(address)
        self.__setters = False
        self.__allowed_channels = list()

    def get_id(self):
        return self.__info["configuration"]["serial"] + " (Network)"

    def get_resolution(self, channel: int):
        if not self._tagger or channel not in self.__info["exposed channels"]:
            return ""
        channel = abs(channel)
        if channel in self._tagger.getChannelList(type=TimeTagger.ChannelEdge.StandardRising):
            return "Standard"
        elif channel in self._tagger.getChannelList(type=TimeTagger.ChannelEdge.HighResRising):
            return "HighRes"
        return ""

    def allows_setters(self):
        return self.__setters

    def _connect(self, **kwargs):
        tagger = TimeTagger.createTimeTaggerNetwork(self.__address)
        try:
            tagger.setTestSignalDivider(tagger.getTestSignalDivider())
            self.__allowed_channels = tagger.getChannelList()
        except:
            self.__setters = False
            self.__allowed_channels = self.__info["exposed channels"]
        return tagger

    def is_channel_allowed(self, channel: int):
        return channel in self.__allowed_channels
