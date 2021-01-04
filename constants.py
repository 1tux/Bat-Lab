from threading import Thread

# PROBABILITIES:

MISSING_DATA_PROB = 0.1  # PER BAT

TARGET_P = 0.0  # 0.05 # 0.005 #; to pick a new target
HD_P = 0.2  # head direction change probability

START_W_P = 0.1  # probability to start walking
KEEP_W_P = 0.9  # probability to keep walking

SEED = 1337
N_BATS = 5
FRAME_RATE = 25  # frames per seconds
TOTAL_TIME_POINTS = 150000  # TOTAL_TIME_POINTS / FRAME_RATE / 60 = 100 minutes = 1 hour 40 minutes
SPIKE_RATE = 25  # 1

SPIKE_P = SPIKE_RATE / FRAME_RATE

SPIKE_NOISE_P = 0.25

HD_CHANGE_RANGE = 90
MIN_V, MAX_V = (0.4, 0.4)  # 10 cm/s for walking
MIN_FV, MAX_FV = (1.20, 3.00)  # flying speed 1.2m/s to 3m/s

# BOX
MIN_X, MAX_X = (0, 100)
MIN_Y, MAX_Y = (0, 100)
TIME_POINTS = TOTAL_TIME_POINTS

DATA_PATH = "data/"

GAUSSIAN_FILTER_SIGMA = 2.5
GAUSSIAN_FILTER_SIZE = 5 * (round(GAUSSIAN_FILTER_SIGMA) + 1)  # 3cm


# python 3 is stupid, so round converts 2.5 to 2, and 3.5 to 4.
def my_round(x):
    import math
    fx = math.floor(x)
    return fx + (round(1 + fx) - 1)


TIME_SPENT_THRESHOLD = my_round(FRAME_RATE / TOTAL_TIME_POINTS * TIME_POINTS) // 4

NET1_WIDTH = 100
NET1_HEIGHT = 50

NET3_WIDTH = 50
NET3_HEIGHT = 50


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return
