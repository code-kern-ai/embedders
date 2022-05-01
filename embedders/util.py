import numpy as np


def batch(documents, batch_size):
    length = len(documents)
    for idx in range(0, length, batch_size):
        yield documents[idx : min(idx + batch_size, length)]


def num_batches(documents, batch_size):
    return int(np.ceil(len(documents) / batch_size))
