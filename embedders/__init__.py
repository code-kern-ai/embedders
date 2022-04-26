from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def batch_encode(self, documents):
        pass

    def batch(self, documents):
        length = len(documents)
        for idx in range(0, length, self.batch_size):
            yield documents[idx : min(idx + self.batch_size, length)]
