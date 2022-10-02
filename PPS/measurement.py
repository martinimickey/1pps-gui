"""Custom measurement class for tracking 1PPS signals."""

from io import TextIOWrapper
from typing import List, Optional, Union
import csv
from os.path import isdir
from os import getcwd
from datetime import datetime, timedelta, timezone, time
import numpy
from .utilities import TimeTagGroup, MissingTimeTagGroup, TimeTagGroupBase, TimeTag
import TimeTagger
from .fast_processing import fast_process, reference_dtype, channel_dtype

NO_SIGNAL_THRESHOLD = timedelta(seconds=5)
COLUMN_DELIMITER = ","


class PpsData:
    def __init__(self, cyclic_data, cyclic_data_reftime, reference):
        start_index = reference["data_index"]
        self.count = reference["count"]
        self.__data = numpy.concatenate((cyclic_data[:, :, start_index:], cyclic_data[:, :, :start_index]), axis=2)
        self.__data_reftime = numpy.concatenate((cyclic_data_reftime[start_index:], cyclic_data_reftime[:start_index]))
        self.__offset = reference["count"]

    def getData(self):
        return self.__data[0, :, :]

    def getIndex(self):
        return numpy.arange(self.__offset-self.__data.shape[2], self.__offset)

    def getTime(self):
        return self.__data_reftime

    def getError(self):
        return self.__data[1, :, :]


class PpsTracking(TimeTagger.CustomMeasurement):
    """Custom measurement class for tracking 1PPS signals."""

    def __init__(self,
                 tagger: TimeTagger.TimeTaggerBase,
                 channels: List[int],
                 n_average: int,
                 reference_channel: Optional[int] = None,
                 period: int = 1000000000000,
                 folder: str = None,
                 channel_names: Optional[List[str]] = None,
                 reference_name: str = "Reference",
                 debug_to_file: bool = False):
        if not channels:
            raise ValueError("You need to provide channels")
        TimeTagger.CustomMeasurement.__init__(self, tagger)
        self.tagger = tagger
        self.data_file: Optional[TextIOWrapper] = None
        self._channel_tags: List[TimeTag] = list()
        self._reference_tag: Optional[TimeTagGroup] = None
        self.__last_reference_time = None
        self._last_time_check = self._now()
        self.__messages = list()
        self.__max_timetags = 300
        self._timetag_index = 0
        self.__message_index = 0
        self.__all_channels = [reference_channel] if reference_channel else []
        self.__all_channels += channels
        self._channel_tag_offest = dict()

        # Data for fast processing
        self.__reference = numpy.array([(reference_channel if reference_channel else 0, period, n_average, 0, 0, 0, 0, 0)], dtype=reference_dtype)
        self.__channels = numpy.array([(channel, 0, 0, -1, 0, 0) for channel in channels], dtype=channel_dtype)
        self.__stored_tags = numpy.empty([len(channels), 100], dtype=numpy.int64)
        self.__histograms = numpy.zeros([len(channels), n_average+10])  # min(period//10, 100000) if reference_channel else (n_average+1000)])
        self.__data = numpy.empty([2, len(channels), self.__max_timetags])
        self.__data[:, :] = numpy.nan
        self.__data_reftime = numpy.zeros(self.__max_timetags, dtype=numpy.int64)

        self.__current_data: Optional[PpsData] = None
        self.__storage_count = 0
        self.__utc_timestamps: list[tuple[int, datetime]] = list()
        # self.channels = channels
        # self.period = period
        self.channel_names = []
        self.reference_name = reference_name if reference_name else "Reference"
        for channel, name in zip(channels, channel_names if channel_names else [""] * len(channels)):
            self.channel_names.append(name if name else f"Channel {channel}")
        # self.reference = reference
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
        if reference_channel:
            self.register_channel(reference_channel)
        self._open_file(self._now())
        self.clear_impl()
        self.finalize_init()

    def __del__(self):
        for channel in self.__all_channels:
            self.unregister_channel(channel)
        self._close_file()
        self.stop()

    def getDataObject(self):
        if not self.__current_data or self.__current_data.count != self.__reference[0]["count"]:
            with self.mutex:
                self.__current_data = PpsData(self.__data, self.__data_reftime, self.__reference[0])
        return self.__current_data

    def getChannelNames(self):
        return self.channel_names

    def process(self, incoming_tags, begin_time, end_time):
        """Method called by TimeTagger.CustomMeasurement for processing of incoming time tags."""
        self.__utc_timestamps.append((end_time, self._now()))
        messages = fast_process(incoming_tags,
                                reference=self.__reference[0],
                                channels=self.__channels,
                                stored_tags=self.__stored_tags,
                                histograms=self.__histograms,
                                data=self.__data,
                                data_reftime=self.__data_reftime)
        for message in messages:
            self._new_message(message)
        self._store_timetags()

    def clear_impl(self):
        pass

    def setTimetagsMaximum(self, value: int):
        with self.mutex:
            self.__max_timetags = value if value > 0 else 0
            current_size = self.__data_reftime.size
            index = self.__reference[0]['data_index']
            if value > current_size:
                nan_data = numpy.empty([2, len(self.__channels), value-current_size], dtype=numpy.float64)
                nan_data[:, :, :] = numpy.nan
                self.__data_reftime = numpy.concatenate((self.__data_reftime[:index],
                                                         numpy.zeros(value-current_size, dtype=numpy.int64),
                                                         self.__data_reftime[index:]), axis=0)
                self.__data = numpy.concatenate((self.__data[:, :, :index], nan_data, self.__data[:, :, index:]), axis=2)
                self._new_message(f"increase by {value-current_size}")
            elif value < current_size:
                end = max(self.__reference[0]["data_index"], value)
                start = end - value
                self.__data_reftime = self.__data_reftime[start:end]
                self.__data = self.__data[:, :, start:end]
                self.__reference[0]["data_index"] -= start
                if self.__reference[0]["data_index"] == value:
                    self.__reference[0]["data_index"] = 0
                self._new_message(f"decrease from {start} to {end}, index {self.__reference[0]['data_index']}")

    def setFolder(self, folder):
        self.folder = folder

    def setNewFileTime(self, value: time):
        self.new_file_time = value

    def getMessages(self, from_index: int):
        return self.__messages[from_index:]

    def _now(self):
        return datetime.now(timezone.utc)

    def _new_message(self, message: str, timestamp: Optional[datetime] = None):
        if timestamp is None:
            timestamp = self._now()
        message = f"{message} ({timestamp.replace(microsecond=0).isoformat()})"
        self.__messages.append(message)
        self.__message_index += 1

    @property
    def message_index(self):
        return self.__message_index

    def _next_tag_group(self, timetag: int, current_time: datetime):
        """Create a new group of tags for a given reference tag."""
        self._reference_tag = TimeTagGroup(
            self._timetag_index, timetag, current_time, self.get_sensor_data(1) if self.debug_to_file else [])
        self._timetag_index += 1

    def _missing_tag_group(self, current_time: datetime):
        self._reference_tag = None
        self.__timetags.append(MissingTimeTagGroup(current_time))
        self._timetag_index += 1

    def get_sensor_data(self, col: int) -> list:
        return [row[col] for row in csv.reader(self.tagger.getSensorData().split("\n"), delimiter="\t") if row]

    def _open_file(self, timestamp):
        self._close_file()
        filename = self.folder + "/" + timestamp.strftime("%Y-%m-%d_%H-%M-%S.csv")
        self.data_file = open(filename, "w", newline="")
        writer = csv.writer(self.data_file, delimiter=COLUMN_DELIMITER)
        debug_header = self.get_sensor_data(0) if self.debug_to_file else []
        writer.writerow(["Index", "UTC", self.reference_name] +
                        [name + value_type for name in self.channel_names for value_type in [" Value", " Error"]] +
                        debug_header)
        self._new_message("New file opened: "+filename)

    def _close_file(self):
        if self.data_file:
            filename = self.data_file.name
            self.data_file.close()
            self.data_file = None
            self._new_message("File closed: " + filename)

    def _store_timetags(self):
        if self.data_file is None:
            self._open_file(self._now())
        if self.data_file:
            writer = csv.writer(self.data_file, delimiter=COLUMN_DELIMITER)
            rows_to_write = self.__reference[0]["count"] - self.__storage_count
            n_data = self.__data.shape[2]
            while rows_to_write > n_data:
                writer.writerow(["--- Missing Tag ---"])
                rows_to_write -= 1
            while rows_to_write:
                index = (self.__reference[0]["data_index"] - rows_to_write) % n_data
                reference = self.__data_reftime[index]
                while self.__utc_timestamps[1][0] < reference:
                    self.__utc_timestamps.pop(0)
                ((t1, utc1), (t2, utc2)) = self.__utc_timestamps[:2]
                timestamp = utc1 + (utc2-utc1)*(reference-t1)/(t2-t1)
                if timestamp.day == self._last_time_check.day:
                    if self._last_time_check.time() < self.new_file_time <= timestamp.time():
                        self._open_file(timestamp)
                        writer = csv.writer(self.data_file, delimiter=COLUMN_DELIMITER)
                else:
                    if timestamp.time() >= self.new_file_time or self.new_file_time >= self.__last_reference_time.time():
                        self._open_file(timestamp)
                        writer = csv.writer(self.data_file, delimiter=COLUMN_DELIMITER)
                self._last_time_check = timestamp
                writer.writerow([self.__storage_count, timestamp.replace(microsecond=0).isoformat(), reference] +
                                [self.__data[i, channel, index] for channel in range(len(self.__channels)) for i in range(2)] +
                                [])  # debug data
                self.__storage_count += 1
                rows_to_write -= 1
            self.data_file.flush()
