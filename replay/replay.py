import sys
import os
import TimeTagger

sys.path.append(os.path.abspath(""))
if True:
    from PPStracking import PpsTracking

ttv = TimeTagger.createTimeTaggerVirtual()
tracking = PpsTracking(ttv, [2, 3, 4], 1, 8, 100000*50, folder="dump")
ttv.replay("replay/test2.ttbin")
ttv.waitForCompletion()
print(tracking.getData())
tracking.stop()
