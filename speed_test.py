import numpy
from PPSutilities import LinearClockApproximation
from time import time

dtype = numpy.dtype({'names': ['type', 'missed_events', 'channel', 'time'], 'formats': ['u1', '<u2', '<i4', '<i8'], 'offsets': [0, 2, 4, 8], 'itemsize': 16}, align=True)

LENGTH = 10000
SIGNALS_PER_SECOND = 1
clock = numpy.arange(10000000)

signal_time = numpy.random.rand(SIGNALS_PER_SECOND)
# tags = [(0, 0, 1, 100000*time+numpy.int64(1000*random)) for time, random in zip(numpy.arange(10000000, dtype=numpy.int64), numpy.random.random(10000000))]
tags = numpy.zeros(10000000 + SIGNALS_PER_SECOND, dtype=dtype)
tags["channel"][SIGNALS_PER_SECOND:] = 1
tags["channel"][:SIGNALS_PER_SECOND] = 2
tags["time"][:SIGNALS_PER_SECOND] = numpy.random.randint(0, 1000000000000, size=SIGNALS_PER_SECOND, dtype=numpy.int64)
tags["time"][SIGNALS_PER_SECOND:] = numpy.arange(10000000, dtype=numpy.int64) + numpy.random.randint(-10000, 10000, size=10000000, dtype=numpy.int64)
# numpy.sort(tags, "channel")
# print(len(tags), numpy.sort(tags, order="time")["time"])
tags = numpy.sort(tags, order="time")
print(len(tags))

appr = LinearClockApproximation(LENGTH, 100000, 1)
# result = list()
for i in range(10):
    start = time()
    appr.process_tags_new(tags)
    print(time() - start)
    tags["time"] += 1000000000000
# appr.process_tags()
