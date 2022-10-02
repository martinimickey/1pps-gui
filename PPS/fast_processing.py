from __future__ import annotations
import numba
import numpy
import TimeTagger

channel_dtype = numpy.dtype([("channel", "<i4"),
                             ("buffer_read_index", "<i4"),
                             ("buffer_write_index", "<i4"),
                             ("offset", "<i8"),
                             ("index", "<i4"),
                             ("missed", "<i4")])
reference_dtype = numpy.dtype([("channel", "<i4"),
                              ("period", "<i8"),
                              ("n_average", "<i4"),
                              ("index", "<i4"),
                              ("count", "<i4"),
                              ("last", "<i8"),
                              ("data_index", "<i4"),
                              ("missed", "<i4")])


@ numba.jit(nopython=True, nogil=True)
def fast_process(tags: numpy.ndarray,
                 reference: numpy.ndarray,
                 channels: numpy.ndarray,
                 stored_tags: numpy.ndarray,
                 histograms: numpy.ndarray,
                 data: numpy.ndarray,
                 data_reftime: numpy.ndarray) -> list[str]:
    """Fast analysis of incoming tags.

    :param tags: An array of incoming tags
    :type tags: numpy.ndarray
    :param reference: Information on the reference channel
    :type reference: numpy.ndarray
    :param channels: Information on the signal channels
    :type channels: numpy.ndarray
    :param stored_tags: The collection of old unprocessed tags
    :type stored_tags: numpy.ndarray
    :param histograms: The accumulated histograms
    :type histograms: numpy.ndarray
    :param data: The resulting data
    :type data: numpy.ndarray
    :param data_reftime: Time axis of the resulting data
    :type data_reftime: numpy.ndarray
    :return: A list of messages
    :rtype: list[str]
    """
    messages = []
    tags_to_store = stored_tags.shape[1]
    histogram_width = histograms.shape[1]
    for tag in tags:
        if tag["type"] == 0:
            if reference["channel"] == 0:
                if reference["last"] == 0:
                    reference["last"] = tag["time"]
                    continue
                else:
                    threshold = tag["time"] - reference["period"]
                    while reference["last"] < threshold:
                        reference["last"] += reference["period"]
                        reference["index"] += 1
                    if reference["index"] >= reference["n_average"]:
                        for i in range(len(channels)):
                            data_set = histograms[i, :channels[i]["index"]]
                            average = numpy.sum(data_set)/channels[i]["index"]
                            data[0, i, reference["data_index"]] = average + reference["period"]
                            data[1, i, reference["data_index"]] = numpy.sqrt(numpy.sum((data_set-average)**2)/channels[i]["index"])
                            channels[i]["index"] = 0
                        data_reftime[reference["data_index"]] = reference["last"]
                        reference["data_index"] += 1
                        reference["count"] += 1
                        if reference["data_index"] == data.shape[2]:
                            reference["data_index"] = 0
                        reference["index"] = 0
                for i in range(len(channels)):
                    if channels[i]["channel"] == tag["channel"]:
                        if channels[i]["offset"] > 0 and channels[i]["index"] < histogram_width:
                            histograms[i, channels[i]["index"]] = tag["time"] - channels[i]["offset"] - reference["period"]
                            channels[i]["index"] += 1
                        channels[i]["offset"] = tag["time"]
                        break

            else:
                if tag["channel"] == reference["channel"]:
                    if reference["last"] > 0:
                        for i in range(len(channels)):
                            while channels["buffer_write_index"][i] != channels["buffer_read_index"][i]:
                                difference = stored_tags[i, channels["buffer_read_index"][i]] - reference["last"]
                                if difference < -reference["period"] // 2:
                                    pass
                                elif difference < reference["period"] // 2:
                                    histograms[i, channels[i]["index"]] = difference
                                    channels[i]["index"] += 1
                                else:
                                    break
                                # if difference < 0:
                                #     pass
                                # elif difference < histogram_width:
                                #     histograms[i, difference] += 1
                                # else:
                                #     break
                                channels["buffer_read_index"][i] += 1
                                if channels["buffer_read_index"][i] == tags_to_store:
                                    channels["buffer_read_index"][i] = 0
                    reference["last"] = tag["time"]
                    reference["index"] += 1
                    if reference["index"] >= reference["n_average"]:
                        for i in range(len(channels)):
                            if channels[i]["index"] == 0:
                                data[:, i, reference["data_index"]] = numpy.nan
                                # channels["offset"][i] = -1
                                messages.append(f"No events on channel {channels[i]['channel']}.")
                            else:
                                data_set = histograms[i, :channels[i]["index"]]
                                average = numpy.sum(data_set)/channels[i]["index"]
                                data[0, i, reference["data_index"]] = average + channels[i]["offset"]
                                data[1, i, reference["data_index"]] = numpy.sqrt(numpy.sum((data_set-average)**2)/channels[i]["index"])
                                channels[i]["index"] = 0
                        data_reftime[reference["data_index"]] = reference["last"]
                        histograms[:, :] = 0
                        reference["data_index"] += 1
                        reference["count"] += 1
                        if reference["data_index"] == data.shape[2]:
                            reference["data_index"] = 0
                        reference["index"] = 0
                elif reference["last"] > 0:
                    for i in range(len(channels)):
                        if channels[i]["channel"] == tag["channel"]:
                            if channels["offset"][i] == -1:
                                difference = tag["time"] - reference["last"]
                                if difference > reference["period"] // 2:
                                    difference = difference - reference["period"]
                                channels["offset"][i] = difference
                            stored_tags[i, channels["buffer_write_index"][i]] = tag["time"] - channels["offset"][i]
                            channels["buffer_write_index"][i] += 1
                            if channels["buffer_write_index"][i] == tags_to_store:
                                channels["buffer_write_index"][i] = 0
                            break
        else:
            print("overflow!", tag["type"])
        # elif tag["type"] == 4:
        #     if tag["channel"] == reference["channel"]:
        #         events_for_block = reference["n_average"] - reference["index"]
        #         missed_events = tag["missed_events"]
        #         while missed_events >= events_for_block:
        #             messages.append("Missed reference tags")
        #             reference["index"] = missed_events
        #             data[reference["data_index"]]
        #             missed_events -= events_for_block
        #             events_for_block = reference["n_average"]
        #     else:
        #         for channel in channels:
        #             if channel["channel"] == tag["channel"]:
        #                 channel["missed"] += tag["missed_events"]
    return messages


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    N_DATA = 5000

    class PpsData:
        def __init__(self, cyclic_data, cyclic_data_reftime, reference):
            start_index = reference["data_index"]
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

    AVERAGE = 1000
    CHANNELS = [1, 2]
    PERIODS = [1_000_000, 999_999]
    NOISES = [20, 20]
    REF_PERIOD = 1_000_000
    REF_CHANNEL = 0

    raw_tags = [(pos, REF_CHANNEL) for pos in numpy.arange(0, int(1E12), REF_PERIOD, dtype=numpy.int64)]
    for channel, period, noise in zip(CHANNELS, PERIODS, NOISES):
        clicks = numpy.arange(0, int(1E12), period, dtype=numpy.int64)
        raw_tags += [(pos, channel) for pos in clicks + numpy.array(numpy.random.normal(scale=noise, size=len(clicks)), dtype=numpy.int64)]
    raw_tags.sort()
    # tag_type = numpy.dtype({'names': ['type', 'missed_events', 'channel', 'time'], 'formats': ['u1', '<u2', '<i4', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}, align=True)
    incoming_tags = numpy.array([(0, 0, ch, pos) for (pos, ch) in raw_tags], dtype=TimeTagger.CustomMeasurement.INCOMING_TAGS_DTYPE)

    reference = numpy.array([(REF_CHANNEL if REF_CHANNEL else 0, REF_PERIOD, AVERAGE, 0, 0, 0, 0, 0)], dtype=reference_dtype)
    channels = numpy.array([(channel, 0, 0, -1, 0) for channel in CHANNELS], dtype=channel_dtype)
    stored_tags = numpy.empty([len(channels), 10], dtype=numpy.int64)
    histograms = numpy.zeros([len(channels), min(REF_PERIOD//10, 100000)])
    data = numpy.empty([2, len(channels), N_DATA])
    data[:, :] = numpy.nan
    data_reftime = numpy.zeros(N_DATA, dtype=numpy.int64)

    fast_process(incoming_tags, reference[0], channels, stored_tags, histograms, data, data_reftime)
    pps_data = PpsData(data, data_reftime, reference[0])

    fig = plt.figure()
    x = pps_data.getTime()
    for y, err in zip(pps_data.getData(), pps_data.getError()):
        plt.fill_between(x, y-err/2, y+err/2, color="C0", alpha=0.2)
        plt.scatter(x, y, marker=".")
    plt.show()
