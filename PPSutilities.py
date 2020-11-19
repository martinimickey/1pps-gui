"""Utilities for 1PPS measurements."""

from collections import deque
import numpy

class LinearClockApproximation:
    """Calculate an approximated value based on linear fitting."""
    def __init__(self, length: int):
        self.deque = deque([0]*length, maxlen=length)
        self.x_sum = sum(range(length))
        self.x_sum_long = self.x_sum + length
        self.x2_sum = sum([x**2 for x in range(length)])
        self.denominator = sum(range(length))**2 - length * self.x2_sum
        self.y_sum = 0
        self.xy_sum = 0
        self.last_tag = 0

    def calculate(self, tag: int):
        step = tag - self.last_tag
        self.last_tag = tag
        full = len(self.deque) == self.deque.maxlen
        if full:
            front_to_end = tag - self.deque[0]
            self.xy_sum += self.y_sum + self.deque.maxlen * front_to_end - self.x_sum_long * step
            self.y_sum += front_to_end - self.deque.maxlen * step
            self.deque.append(tag)
        return tag + (self.xy_sum * self.x_sum - self.y_sum * self.x2_sum)//self.denominator

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    VALUES = 1000
    y = [int(i*100000 + (numpy.random.random()-0.5)*500000) for i in range(VALUES)]
    app = LinearClockApproximation(100)
    app_y = list()
    for v in y:
        app_y.append(app.calculate(v))
    plt.scatter(range(VALUES), y, marker=".")
    plt.scatter(range(VALUES), app_y, marker=".")
    plt.show()