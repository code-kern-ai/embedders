from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import PCA


class Embedder(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _encode(self, documents, fit_model):
        pass

    def encode(self, documents, as_generator=False, fit_model=True):
        if as_generator:
            return self._encode(documents, fit_model)
        else:
            embeddings = []
            for embedding_batch in self._encode(documents, fit_model):
                embeddings.extend(embedding_batch)
            return embeddings

    def batch(self, documents):
        length = len(documents)
        for idx in range(0, length, self.batch_size):
            yield documents[idx : min(idx + self.batch_size, length)]

    def num_batches(self, documents):
        return int(np.ceil(len(documents) / self.batch_size))


class PCAReducer(ABC):
    def __init__(self, embedder, n_components=8):
        self.embedder = embedder
        self.reducer = PCA(n_components=n_components)

    def transform(self, documents):
        embeddings = self.embedder.encode(documents, fit_model=False)
        return self.transform_batch(embeddings)

    @abstractmethod
    def transform_batch(self, embedding_batch):
        pass

    @abstractmethod
    def _fit_transform(self, documents, fit_batches):
        pass

    def fit_transform(self, documents, fit_batches=5, as_generator=False):
        if as_generator:
            return self._fit_transform(documents, fit_batches)
        else:
            embeddings = []
            for embedding_batch in self._fit_transform(documents, fit_batches):
                embeddings.extend(embedding_batch)
            return embeddings
