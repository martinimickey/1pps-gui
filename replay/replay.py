import sys
import os
import numpy
import TimeTagger

sys.path.append(os.path.abspath(""))
if True:
    from PPStracking import PpsTracking

ttv = TimeTagger.createTimeTaggerVirtual()
tracking = PpsTracking(ttv, [2, 3, 4, 5, 6], 1, 7, 100000, folder="dump")
ttv.replay("replay/test5.ttbin")
ttv.waitForCompletion()
for channel in tracking.getData():
    x = channel[~numpy.isnan(channel)]
    print(numpy.std(x), numpy.mean(x))
tracking.stop()
