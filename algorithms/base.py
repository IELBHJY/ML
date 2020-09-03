import time
import numpy as np


class Timer:
    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self):
        self.end_time = time.time()
        print("This model running time:", self.end_time - self.start_time)


class Function:

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e(-x))

