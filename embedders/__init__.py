from abc import ABC, abstractmethod
from sklearn.decomposition import PCA


class Transformer(ABC):
    @abstractmethod
    def fit_transform(self, documents, as_generator):
        pass

    @abstractmethod
    def transform(self, documents, as_generator):
        pass


class Embedder(Transformer):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _encode(self, documents, fit_model):
        pass

    def _encode_batch(self, documents, as_generator, fit_model):
        if as_generator:
            return self._encode(documents, fit_model)
        else:
            embeddings = []
            for embedding_batch in self._encode(documents, fit_model):
                embeddings.extend(embedding_batch)
            return embeddings

    def fit_transform(self, documents, as_generator=False):
        return self._encode_batch(documents, as_generator, True)

    def transform(self, documents, as_generator=False):
        return self._encode_batch(documents, as_generator, False)


class PCAReducer(Transformer):
    def __init__(self, embedder, n_components=8, **kwargs):
        self.embedder = embedder
        self.reducer = PCA(n_components=n_components, **kwargs)
        self.batch_size = self.embedder.batch_size

    @abstractmethod
    def _reduce(self, documents, fit_model, fit_batches):
        pass

    def _reduce_batch(
        self, documents, as_generator, fit_model, autocorrect_n_components, fit_batches
    ):
        if autocorrect_n_components:
            self.reducer.n_components = min(self.reducer.n_components, len(documents))
        if as_generator:
            return self._reduce(documents, fit_model, fit_batches)
        else:
            embeddings = []
            for embedding_batch in self._reduce(documents, fit_model, fit_batches):
                embeddings.extend(embedding_batch)
            return embeddings

    def fit_transform(
        self,
        documents,
        as_generator=False,
        fit_batches=5,
        autocorrect_n_components=True,
    ):
        return self._reduce_batch(
            documents, as_generator, True, autocorrect_n_components, fit_batches
        )

    def transform(self, documents, as_generator=False):
        return self._reduce_batch(documents, as_generator, False, False, 0)
