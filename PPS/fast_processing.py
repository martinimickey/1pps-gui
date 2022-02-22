from __future__ import annotations
import numba
import numpy
import TimeTagger

HISTOGRAM_WIDTH = 100_000
TAGS_TO_STORE = 10
N_DATA = 20


@numba.jit(nopython=True, nogil=True)
def fast_process(tags,
                 reference_channel: int,
                 channels: list[int],
                 period: int,
                 reference_count: int,
                 reference_average: int,
                 reference_last,
                 offsets,
                 stored_tags,
                 read_index,
                 write_index,
                 histograms,
                 data: numpy.ndarray,
                 data_index):
    histogram_width = histograms.shape[1]
    tags_to_store = stored_tags.shape[1]
    for tag in tags:
        if tag["channel"] == reference_channel:
            if reference_last > 0:
                for i in range(len(channels)):
                    while write_index[i] != read_index[i]:
                        difference = stored_tags[i, read_index[i]] - reference_last
                        if difference < 0:
                            pass
                        elif difference < histogram_width:
                            histograms[i, difference] += 1
                        else:
                            break
                        read_index[i] += 1
                        if read_index[i] == tags_to_store:
                            read_index[i] = 0
            reference_last = tag["time"]
            reference_count += 1
            if reference_count == reference_average:
                index = numpy.arange(histogram_width, dtype=numpy.float64)
                counts = numpy.sum(histograms, axis=1)
                for i in range(len(channels)):
                    if counts[i] == 0:
                        data[:, i, data_index] = numpy.nan
                        offsets[i] = -1
                    else:
                        normalized_histogram = histograms[i, :].astype(numpy.float64)/counts[i]
                        average = numpy.sum(normalized_histogram * index)
                        data[0, i, data_index] = average - histogram_width//2
                        data[1, i, data_index] = numpy.sqrt(numpy.sum(normalized_histogram*(index-average)**2))
                histograms[:, :] = 0
                data_index += 1
                if data_index == data.shape[2]:
                    data_index = 0
                reference_count = 0
        else:
            for i in range(len(channels)):
                if channels[i] == tag["channel"]:
                    if offsets[i] == -1:
                        difference = numpy.int32(tag["time"] - reference_last)
                        if difference > period // 2:
                            difference -= period
                        offsets[i] = numpy.int32(tag["time"] - reference_last) - histogram_width // 2
                    stored_tags[i, write_index[i]] = tag["time"] - offsets[i]
                    write_index[i] += 1
                    if write_index[i] == TAGS_TO_STORE:
                        write_index[i] = 0
    return reference_count, reference_last, data_index


if __name__ == "__main__":
    from time import sleep
    from matplotlib import pyplot as plt

    class PpsData:
        def __init__(self, cyclic_data, start_index, offset):
            self.__data = numpy.concatenate((cyclic_data[:, :, start_index:], cyclic_data[:, :, :start_index]), axis=2)
            self.__data
            self.__offset = offset

        def getData(self):
            return self.__data[0, :, :]

        def getIndex(self):
            return numpy.arange(self.__offset, self.__offset+self.__data.shape[2])

        def getError(self):
            return self.__data[1, :, :]

    class TestMeasurement(TimeTagger.CustomMeasurement):
        def __init__(self, tagger: TimeTagger.TimeTaggerBase, reference_channel: int, channels: list[int], period: int, average: int):
            super().__init__(tagger)
            self.__reference_channel = reference_channel
            self.__channels = numpy.array(channels)
            self.__period = period
            self.__average = average
            self.__offsets = -numpy.ones_like(channels, dtype=numpy.int32)
            self.__read_index = numpy.zeros_like(channels, dtype=numpy.int32)
            self.__write_index = numpy.zeros_like(channels, dtype=numpy.int32)
            self.__stored_tags = numpy.empty([len(channels), TAGS_TO_STORE], dtype=numpy.int64)
            self.__histograms = numpy.zeros([len(channels), min(period//10, 100000)])
            self.__reference_count = 0
            self.__reference_last = 0
            self.__data = numpy.empty([2, len(channels), N_DATA])
            self.__data[:, :] = numpy.nan
            self.__data_index = 0
            self.__data_offset = -N_DATA
            self.register_channel(reference_channel)
            for channel in channels:
                self.register_channel(channel)
            self.finalize_init()

        def getDataObject(self):
            with self.mutex:
                data = PpsData(self.__data, self.__data_index, self.__data_offset)
            return data

        def process(self, incoming_tags, begin_time, end_time):
            (self.__reference_count,
             self.__reference_last,
             data_index) = fast_process(incoming_tags,
                                        reference_channel=self.__reference_channel,
                                        channels=self.__channels,
                                        period=self.__period,
                                        reference_count=self.__reference_count,
                                        reference_average=self.__average,
                                        reference_last=self.__reference_last,
                                        offsets=self.__offsets,
                                        stored_tags=self.__stored_tags,
                                        read_index=self.__read_index,
                                        write_index=self.__write_index,
                                        histograms=self.__histograms,
                                        data=self.__data,
                                        data_index=self.__data_index)
            self.__data_offset += data_index - self.__data_index
            if data_index < self.__data_index:
                self.__data_offset += N_DATA
            self.__data_index = data_index
            # print(self.__data_index, self.__data_offset)

    ttv = TimeTagger.createTimeTaggerVirtual()
    ttv.setTestSignal(1, True)
    ttv.setTestSignal(2, True)
    ttv.setTestSignal(3, True)
    ttv.setReplaySpeed(1)
    test = TestMeasurement(ttv, 2, [1, 3], 1200000, 800000)
    count = TimeTagger.Countrate(ttv, [1, 2, 3])
    fig = plt.figure()
    while True:
        data = test.getDataObject()
        fig.clear()
        x = data.getIndex()
        for y, err in zip(data.getData(), data.getError()):
            plt.fill_between(x, y-err/2, y+err/2, color="C0", alpha=0.2)
            plt.scatter(x, y, marker=".")
        plt.pause(.01)
