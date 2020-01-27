import numpy as np
import time

class Timer:
    ''' Simple stopwatch-like timer object.
    '''

    def __init__(self):
        self.time_init = time.time()
        self.time_prev = self.time_init

    def lap(self):
        time_curr = time.time()
        time_diff = time_curr - self.time_prev

        self.time_prev = time_curr

        return time_diff

    def total(self):
        time_curr = time.time()
        time_diff = time_curr - self.time_init

        return time_diff

    def copy(self):
        copy = Timer()

        copy.time_init = self.time_init
        copy.time_prev = self.time_prev

        return copy

    def __call__(self):
        return self.lap()

