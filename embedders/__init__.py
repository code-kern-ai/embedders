from abc import ABC, abstractmethod
import numpy as np


class Embedder(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, documents):
        pass

    def batch(self, documents):
        length = len(documents)
        for idx in range(0, length, self.batch_size):
            yield documents[idx : min(idx + self.batch_size, length)]

    def num_batches(self, documents):
        return int(np.ceil(len(documents) / self.batch_size))
