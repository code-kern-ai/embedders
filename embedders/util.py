from typing import Any, Generator, List
import numpy as np


def batch(documents: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
    length = len(documents)
    for idx in range(0, length, batch_size):
        yield documents[idx : min(idx + batch_size, length)]


def num_batches(documents: List[Any], batch_size: int) -> int:
    return int(np.ceil(len(documents) / batch_size))
