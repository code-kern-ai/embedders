from abc import ABC, abstractmethod
from typing import Any, List, Generator, Union
from sklearn.decomposition import PCA


class Transformer(ABC):
    @abstractmethod
    def fit_transform(
        self, documents: List[Any], as_generator: bool
    ) -> Union[List, Generator]:
        pass

    @abstractmethod
    def transform(
        self, documents: List[Any], as_generator: bool
    ) -> Union[List, Generator]:
        pass


class Embedder(Transformer):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _encode(self, documents: List[Any], fit_model: bool) -> Generator:
        pass

    def _encode_batch(
        self, documents: List[Any], as_generator: bool, fit_model: bool
    ) -> Union[List, Generator]:
        if as_generator:
            return self._encode(documents, fit_model)
        else:
            embeddings = []
            for embedding_batch in self._encode(documents, fit_model):
                embeddings.extend(embedding_batch)
            return embeddings

    def fit_transform(
        self, documents: List[Any], as_generator: bool = False
    ) -> Union[List, Generator]:
        return self._encode_batch(documents, as_generator, True)

    def transform(
        self, documents: List[Any], as_generator: bool = False
    ) -> Union[List, Generator]:
        return self._encode_batch(documents, as_generator, False)


class PCAReducer(Transformer):
    def __init__(self, embedder: Embedder, n_components: int = 8, **kwargs):
        self.embedder = embedder
        self.reducer = PCA(n_components=n_components, **kwargs)
        self.batch_size = self.embedder.batch_size

    @abstractmethod
    def _reduce(self, documents: List[Any], fit_model: bool, fit_after_n_batches: int):
        pass

    def _reduce_batch(
        self,
        documents: List[Any],
        as_generator: bool,
        fit_model: bool,
        autocorrect_n_components: bool,
        fit_after_n_batches: int,
    ) -> Union[List, Generator]:
        if autocorrect_n_components:
            self.reducer.n_components = min(self.reducer.n_components, len(documents))
        if as_generator:
            return self._reduce(documents, fit_model, fit_after_n_batches)
        else:
            embeddings = []
            for embedding_batch in self._reduce(
                documents, fit_model, fit_after_n_batches
            ):
                embeddings.extend(embedding_batch)
            return embeddings

    def fit_transform(
        self,
        documents: List[Any],
        as_generator: bool = False,
        fit_after_n_batches: int = 5,
        autocorrect_n_components: bool = True,
    ) -> Union[List, Generator]:
        return self._reduce_batch(
            documents, as_generator, True, autocorrect_n_components, fit_after_n_batches
        )

    def transform(self, documents, as_generator=False) -> Union[List, Generator]:
        return self._reduce_batch(documents, as_generator, False, False, 0)
