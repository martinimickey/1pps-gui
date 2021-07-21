import TimeTagger
import os

if not os.path.exists("data"):
    os.mkdir("data")

tagger = TimeTagger.createTimeTagger()
sync = TimeTagger.SynchronizedMeasurements(tagger)
filewriter = TimeTagger.FileWriter(sync.getTagger(), "data/dump.ttbin", list(range(1, 9)))
sync.startFor(10000000000000)
sync.waitUntilFinished()
