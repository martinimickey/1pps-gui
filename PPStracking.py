from io import TextIOWrapper
from typing import List, Dict, Optional, Tuple
import numpy
from numba import jit, jitclass
import csv
from os.path import isdir
from os import getcwd
from datetime import datetime, timedelta, timezone, time
import TimeTagger

NO_SIGNAL_THRESHOLD = timedelta(seconds=5)

class TimeTagGroup:
    def __init__(self, index, reference_tag: int, clock_offset: int, time: datetime, debug_data: List[str]):
        self.reference_tag = reference_tag
        self.channel_tags = dict()
        self.time = time
        self.index = index
        self.clock_offset = clock_offset
        self.debug_data = debug_data

    def add_tag(self, channel: int, timetag: int):
        self.channel_tags[channel] = timetag

    def get_reference_with_offset(self):
        return self.reference_tag - self.clock_offset

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
        self.channel = channel
        self.period = period
        self.last_time_tag = 0
        self._offset = 0

    def update_offset(self, elapsed_time: int, periods: int):
        self._offset += elapsed_time - periods * self.period

    def get_offset(self):
        return self._offset


class PpsTracking(TimeTagger.CustomMeasurement):
    def __init__(self,
                 tagger: TimeTagger.TimeTaggerBase,
                 channels: List[int],
                 reference: int,
                 clock: Optional[int] = None,
                 clock_period: int = 100000,
                 period: int = 1000000000000,
                 folder: str = None,
                 channel_names: Optional[List[str]] = None,
                 reference_name: str = "Reference",
                 debug_to_file: bool = False):
        super().__init__(tagger)
        self.data_file: Optional[TextIOWrapper] = None
        self._current_index = 0
        self._channel_tags = list()
        self._reference_tag = None
        self._last_signal_time = self._now()
        self._last_reference_time = None
        self._last_time_check = self._now()
        self._messages = list()
        self._timetags = list()
        self._max_timetags = 300
        self._timetag_index = 0
        self._message_index = 0
        self._last_clock_tag = 0
        self._number_of_clock_cycles = 0
        self._ps_last_ref_after_clock = 0
        # self._clock_offset = 0

        self.channels = channels
        self.clock = Clock(clock, clock_period) if clock else None
        self.period = period
        # self.clock_period = clock_period
        self.channel_names = []
        self.reference_name = reference_name
        for channel, name in zip(self.channels, channel_names if channel_names else [""] * len(channels)):
            self.channel_names.append(name if name else f"Channel {channel}")
        self.reference = reference
        if folder is None or not isdir(folder):
            self.folder = getcwd()
        else:
            self.folder = folder
        print(self.folder)
        self.new_file_time = time(hour=0, minute=0, second=0)
        try:
            if tagger.getModel() == "Time Tagger 20" and debug_to_file:
                self._new_message("Time Tagger 20 cannot add debug data")
                debug_to_file = False
        except AttributeError:
            debug_to_file = False
        self.debug_to_file = debug_to_file
        for channel in channels:
            self.register_channel(channel)
        self.register_channel(reference)
        if clock:
            self.register_channel(clock)
        self._open_file(self._last_time_check)
        self.clear_impl()
        self.finalize_init()

    def __del__(self):
        self._close_file()
        self.stop()

    def getData(self):
        data = numpy.empty(
            [len(self.channels), len(self._timetags)], dtype=float)
        for index, tag in enumerate(self._timetags):
            data[:, index] = tag.get_channel_tags(self.channels)
        return data

    def getIndex(self):
        return numpy.arange(self._timetag_index - len(self._timetags), self._timetag_index)

    def getMeasurementStatus(self):
        return self._timetag_index, self._message_index

    def getChannelNames(self):
        return self.channel_names

    def measure(self, incoming_tags, begin_time, end_time):
        """Method called by TimeTagger.CustomMeasurement for processing of incoming time tags."""
        current_time = self._now()
        tag_index = 0
        while len(incoming_tags) > tag_index:
            if self.clock:
                periods, last_clock_tag = PpsTracking.remove_clock(incoming_tags[tag_index:], self.clock.channel)
                if periods:
                    if self._last_clock_tag:
                        tag_index += periods
                        self.clock.update_offset(last_clock_tag - self._last_clock_tag, periods)
                    self._last_clock_tag = last_clock_tag
                    if tag_index == len(incoming_tags):
                        break
            channel = incoming_tags[tag_index]["channel"]
            timetag = incoming_tags[tag_index]["time"]
            tag_index += 1
            if channel == self.reference:
                ps_this_ref_after_clock = timetag - self._last_clock_tag
                if self._reference_tag is not None:
                    index = self._select_tags_within_range(timetag)
                    self._channel_tags = self._channel_tags[index:]
                self._ps_last_ref_after_clock = ps_this_ref_after_clock
                self._number_of_clock_cycles = 0
                self._last_reference_time = current_time
                self._next_tag_group(timetag, current_time)
            else:
                self._channel_tags.append((channel, timetag))
            self._last_signal_time = current_time
        if current_time - self._last_signal_time > NO_SIGNAL_THRESHOLD:
            self._new_message(
                "No incoming signals for more than " + str(NO_SIGNAL_THRESHOLD.seconds) + " s.", current_time)
            self._last_signal_time = current_time

    def _select_tags_within_range(self, this_timetag):
        lower_limit = self._reference_tag.reference_tag - self.period//2
        upper_limit = self._reference_tag.reference_tag + self.period//2
        index = 0
        for index, (channel, timetag) in enumerate(self._channel_tags):
            if timetag > upper_limit:
                break
            if timetag >= lower_limit:
                self._reference_tag.add_tag(channel, timetag)
            else:
                num_periods = (
                    this_timetag - self._reference_tag.reference_tag - self.period//2) // self.period
                if num_periods:
                    MissingTimeTagGroups(
                        num_periods, self._channel_tags[:index])
        return index

    def clear_impl(self):
        pass

    def setTimetagsMaximum(self, value: int):
        self._max_timetags = value if value > 0 else 0

    def setFolder(self, folder):
        self.folder = folder

    def setNewFileTime(self, value: time):
        self.new_file_time = value

    def getMessages(self, from_index: int):
        return self._messages[from_index:]

    def _now(self):
        return datetime.now(timezone.utc)

    def _new_message(self, message: str, time: Optional[datetime] = None):
        if time is None:
            time = self._now()
        message = f"{time.replace(microsecond=0).isoformat()}: {message}"
        self._messages.append(message)
        self._message_index += 1

    def _next_tag_group(self, timetag: int, current_time: datetime):
        if self._reference_tag is not None:
            if missing := self._reference_tag.get_missing_channels(self.channels):
                self._new_message(f"Tags missing: " +
                                  ", ".join([f"input {ch}" for ch in missing]))
            self._timetags.append(self._reference_tag)
            self._store_timetag(self._reference_tag)
            if len(self._timetags) > self._max_timetags:
                self._timetags = self._timetags[-self._max_timetags:]
        self._reference_tag = TimeTagGroup(
            self._timetag_index, timetag, self.clock.get_offset() if self.clock else 0, current_time, self.get_sensor_data(1) if self.debug_to_file else [])
        self._timetag_index += 1

    def get_sensor_data(self, col: int) -> list:
        return [row[col] for row in csv.reader(self.tagger.getSensorData().split("\n"), delimiter="\t") if row]

    def _open_file(self, current_time: datetime):
        self._close_file()
        filename = self.folder + "/" + current_time.strftime("%Y-%m-%d_%H-%M-%S.csv")
        self.data_file = open(filename, "w", newline="")
        writer = csv.writer(self.data_file, delimiter=",")
        debug_header = self.get_sensor_data(0) if self.debug_to_file else []
        writer.writerow(["Index", "UTC", self.reference_name] +
                        self.channel_names + debug_header)
        self._new_message("New file opened: "+filename)

    def _close_file(self):
        if self.data_file:
            filename = self.data_file.name
            self.data_file.close()
            self.data_file = None
            self._new_message("File closed: "+filename)

    def _store_timetag(self, tag: TimeTagGroup):
        if self.data_file is None:
            self._open_file(tag.time)
        else:
            if tag.time.day == self._last_time_check.day:
                if self._last_time_check.time() < self.new_file_time <= tag.time.time():
                    self._open_file(tag.time)
            else:
                if tag.time.time() >= self.new_file_time or self.new_file_time >= self._last_reference_time.time():
                    self._open_file(tag.time)
        writer = csv.writer(self.data_file, delimiter=",")
        writer.writerow([tag.index, tag.time.replace(microsecond=0).isoformat(), tag.get_reference_with_offset()] +
                        tag.get_channel_tags(self.channels) + tag.debug_data)
        self._last_time_check = tag.time

    @staticmethod
    @jit(nopython=True, nogil=True)
    def remove_clock(tags: numpy.array, clock_channel: int):
        index = 0
        last_clock = 0
        for tag in tags:
            if tag["channel"] == clock_channel:
                last_clock = tag["time"]
                index += 1
                continue
            break
        return index, last_clock
