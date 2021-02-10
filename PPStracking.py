"""Custom measurement class for tracking 1PPS signals."""

from io import TextIOWrapper
from typing import List, Optional, Union
import csv
from os.path import isdir
from os import getcwd
from datetime import datetime, timedelta, timezone, time
import numpy
from PPSutilities import Clock, TimeTagGroup, MissingTimeTagGroup, TimeTagGroupBase, TimeTag
import TimeTagger
from time import sleep

NO_SIGNAL_THRESHOLD = timedelta(seconds=5)
COLUMN_DELIMITER = ","


class PpsTracking(TimeTagger.CustomMeasurement):
    """Custom measurement class for tracking 1PPS signals."""

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
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.tagger = tagger
        self.data_file: Optional[TextIOWrapper] = None
        self._channel_tags: List[TimeTag] = list()
        self._reference_tag: Optional[TimeTagGroup] = None
        self._last_signal_time = self._now()
        self._last_reference_time = None
        self._last_time_check = self._now()
        self._messages = list()
        self._timetags: List[TimeTagGroupBase] = list()
        self._max_timetags = 300
        self._timetag_index = 0
        self._message_index = 0
        self._all_channels = list()
        self._channel_tag_offest = dict()

        self.channels = channels
        self.clock = Clock(clock, clock_period, self.dtype) if clock else None
        self.period = period
        self.channel_names = []
        self.reference_name = reference_name
        for channel, name in zip(self.channels, channel_names if channel_names else [""] * len(channels)):
            self.channel_names.append(name if name else f"Channel {channel}")
        self.reference = reference
        if folder is None or not isdir(folder):
            self.folder = getcwd()
        else:
            self.folder = folder
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

    def register_channel(self, channel: int):
        super().register_channel(channel)
        self._all_channels.append(channel)

    def getData(self):
        self._lock()
        data = numpy.empty(
            [len(self.channels), len(self._timetags)], dtype=float)
        for index, tag in enumerate(self._timetags):
            data[:, index] = tag.get_channel_tags(self.channels)
        self._unlock()
        # print("")
        # for channel in data:
        #     x = channel[~numpy.isnan(channel)]
        #     print(numpy.std(x), numpy.mean(x))
        return data

    def getIndex(self):
        return numpy.arange(self._timetag_index - len(self._timetags), self._timetag_index)

    def getMeasurementStatus(self):
        return self._timetag_index, self._message_index

    def getChannelNames(self):
        return self.channel_names

    def process(self, incoming_tags, begin_time, end_time):
        """Method called by TimeTagger.CustomMeasurement for processing of incoming time tags."""
        current_time = self._now()
        tag_index = 0
        incoming_tags = incoming_tags[numpy.logical_or(numpy.isin(incoming_tags["channel"], self._all_channels), incoming_tags["type"] > 0)]
        if self.clock:
            incoming_tags = self.clock.process_tags(incoming_tags[tag_index:])
        for tag in incoming_tags:
            if tag["channel"] == self.reference:
                self._process_reference_tag()
            if tag["type"] == 0:
                if tag["channel"] == self.reference:
                    self._last_reference_time = current_time
                    self._next_tag_group(tag["time"], current_time)
                else:
                    self._channel_tags.append(TimeTag(tag))
            else:
                if tag["type"] == 4:
                    if tag["channel"] == self.reference:
                        for i in range(tag["missed_events"]):
                            self._missing_tag_group(current_time=current_time)
            self._last_signal_time = current_time
        if current_time - self._last_signal_time > NO_SIGNAL_THRESHOLD:
            self._new_message(
                "No incoming signals for more than " + str(NO_SIGNAL_THRESHOLD.seconds) + " s.", current_time)
            self._last_signal_time = current_time

    def _select_tags_within_range(self):
        """Add the timetags for the last reference tag. Called when a new reference tag arrives."""
        reference_time = self._reference_tag.reference_tag
        lower_limit = reference_time - self.period
        self._determine_channel_tag_offset()
        remaining = list()
        minimum_distance = dict()
        for tag in self._channel_tags:
            if tag.channel not in self._channel_tag_offest:
                remaining.append(tag)
                continue
            if tag.time < lower_limit:
                continue
            distance = tag.time - reference_time - self._channel_tag_offest[tag.channel]
            if tag.channel not in minimum_distance or abs(distance) < minimum_distance[tag.channel]:
                minimum_distance[tag.channel] = abs(distance)
                self._reference_tag.add_tag(tag)
            else:
                remaining.append(tag)
        self._channel_tags = remaining

    def _determine_channel_tag_offset(self):
        if len(self._channel_tag_offest) < len(self.channels):
            reference_time = self._reference_tag.reference_tag
            for channel in self.channels:
                if channel not in self._channel_tag_offest:
                    distance = [tag.time - reference_time for tag in self._channel_tags if tag.channel == channel]
                    if len(distance) > 1:
                        abs_distance = [abs(d) for d in distance]
                        if min(abs_distance) < self.period:
                            self._channel_tag_offest[channel] = distance[abs_distance.index(min(abs_distance))]
            print(self._channel_tag_offest)

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

    def _new_message(self, message: str, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = self._now()
        message = f"{timestamp.replace(microsecond=0).isoformat()}: {message}"
        self._messages.append(message)
        self._message_index += 1

    def _process_reference_tag(self):
        if self._reference_tag is not None:
            self._select_tags_within_range()
            if missing := self._reference_tag.get_missing_channels(self.channels):
                self._new_message("Tags missing: " +
                                  ", ".join([f"input {ch}" for ch in missing]))
            self._timetags.append(self._reference_tag)
            self._store_timetag(self._reference_tag)
            if len(self._timetags) > self._max_timetags:
                self._timetags = self._timetags[-self._max_timetags:]

    def _next_tag_group(self, timetag: int, current_time: datetime):
        """Create a new group of tags for a given reference tag."""
        self._reference_tag = TimeTagGroup(
            self._timetag_index, timetag, current_time, self.get_sensor_data(1) if self.debug_to_file else [])
        self._timetag_index += 1

    def _missing_tag_group(self, current_time: datetime):
        self._reference_tag = None
        self._timetags.append(MissingTimeTagGroup(current_time))
        self._timetag_index += 1

    def get_sensor_data(self, col: int) -> list:
        return [row[col] for row in csv.reader(self.tagger.getSensorData().split("\n"), delimiter="\t") if row]

    def _open_file(self, current_time: datetime):
        self._close_file()
        filename = self.folder + "/" + current_time.strftime("%Y-%m-%d_%H-%M-%S.csv")
        self.data_file = open(filename, "w", newline="")
        writer = csv.writer(self.data_file, delimiter=COLUMN_DELIMITER)
        debug_header = self.get_sensor_data(0) if self.debug_to_file else []
        writer.writerow(["Index", "UTC", self.reference_name] +
                        self.channel_names + debug_header)
        self._new_message("New file opened: "+filename)

    def _close_file(self):
        if self.data_file:
            filename = self.data_file.name
            self.data_file.close()
            self.data_file = None
            self._new_message("File closed: " + filename)

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
        if self.data_file:
            writer = csv.writer(self.data_file, delimiter=COLUMN_DELIMITER)
            writer.writerow([tag.index, tag.time.replace(microsecond=0).isoformat(), tag.reference_tag] +
                            tag.get_channel_tags(self.channels) + tag.debug_data)
        self._last_time_check = tag.time
