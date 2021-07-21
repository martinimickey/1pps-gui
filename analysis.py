import pandas
from matplotlib import pyplot as plt
import numpy

file_name = "2021-06-18_13-14-53.csv"
file_name = "2021-06-18_12-31-29.csv"
channels = [f"Channel {i}" for i in range(2, 7)]

with open("data/" + file_name) as f:
    data = pandas.read_csv(f)

for channel in channels:
    values = data[channel][1:]
    values -= numpy.average(values)
    print(numpy.std(values))
    plt.plot(values)
plt.show()
