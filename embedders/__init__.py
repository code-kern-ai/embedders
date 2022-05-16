from abc import ABC, abstractmethod
from typing import List, Generator, Optional, Union
from spacy.tokens.doc import Doc
from sklearn.decomposition import PCA
from tqdm import tqdm
from embedders import util


class Transformer(ABC):
    @abstractmethod
    def fit_transform(
        self, documents: List[Union[str, Doc]], as_generator: bool
    ) -> Union[List, Generator]:
        """Trains the given algorithm to embed textual documents into semantic vector-spacy representations.

        Args:
            documents (List[Union[str, Doc]]): List of plain strings or spaCy documents.
            as_generator (bool): Embeddings are calculated batch-wise. If this is set to False, the results will be summarized in one list, else a generator will yield the values.

        Returns:
            Union[List, Generator]: List with all embeddings or generator that yields the embeddings.
        """
        pass

    @abstractmethod
    def transform(
        self, documents: List[Union[str, Doc]], as_generator: bool
    ) -> Union[List, Generator]:
        """Uses the trained algorithm to embed textual documents into semantic vector-spacy representations.

        Args:
            documents (List[Union[str, Doc]]): List of plain strings or spaCy documents.
            as_generator (bool): Embeddings are calculated batch-wise. If this is set to False, the results will be summarized in one list, else a generator will yield the values.

        Returns:
            Union[List, Generator]: List with all embeddings or generator that yields the embeddings.
        """
        pass


class Embedder(Transformer):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _encode(self, documents: List[Union[str, Doc]], fit_model: bool) -> Generator:
        pass

    def _encode_batch(
        self, documents: List[Union[str, Doc]], as_generator: bool, fit_model: bool, show_progress: Optional[bool] = True
    ) -> Union[List, Generator]:
        if as_generator:
            return self._encode(documents, fit_model)
        else:
            embeddings = []
            if show_progress:
                num_batches = util.num_batches(documents, self.batch_size)
                print("Initializing model, might take some time...")
                for embedding_batch in tqdm(self._encode(documents, fit_model), total=num_batches, desc="Encoding batches ..."):
                    embeddings.extend(embedding_batch)
            else:
                for embedding_batch in self._encode(documents, fit_model):
                    embeddings.extend(embedding_batch)
            return embeddings

    def fit_transform(
        self, documents: List[Union[str, Doc]], as_generator: bool = False
    ) -> Union[List, Generator]:
        return self._encode_batch(documents, as_generator, True)

    def transform(
        self, documents: List[Union[str, Doc]], as_generator: bool = False
    ) -> Union[List, Generator]:
        return self._encode_batch(documents, as_generator, False)


class PCAReducer(Transformer):
    """Wraps embedder into a principial component analysis to reduce the dimensionality.

    Args:
        embedder (Embedder): Algorithm to embed the documents.
        n_components (int, optional): Number of principal components to keep. Defaults to 8.
        autocorrect_n_components (bool, optional): If there are less data samples than specified components, this will automatically reduce the number of principial components. Defaults to True.
    """

    def __init__(
        self,
        embedder: Embedder,
        n_components: int = 8,
        autocorrect_n_components: bool = True,
        **kwargs
    ):
        self.embedder = embedder
        self.reducer = PCA(n_components=n_components, **kwargs)
        self.batch_size = self.embedder.batch_size
        self.autocorrect_n_components = autocorrect_n_components

    @abstractmethod
    def _reduce(
        self,
        documents: List[Union[str, Doc]],
        fit_model: bool,
        fit_after_n_batches: int,
    ):
        pass

    def _reduce_batch(
        self,
        documents: List[Union[str, Doc]],
        as_generator: bool,
        fit_model: bool,
        fit_after_n_batches: int,
    ) -> Union[List, Generator]:
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
        documents: List[Union[str, Doc]],
        as_generator: bool = False,
        fit_after_n_batches: int = 5,
    ) -> Union[List, Generator]:
        """Trains the given algorithm to embed textual documents into semantic vector-spacy representations.

        Args:
            documents (List[Union[str, Doc]]): List of plain strings or spaCy documents.
            as_generator (bool, optional): Embeddings are calculated batch-wise. If this is set to False, the results will be summarized in one list, else a generator will yield the values.. Defaults to False.
            fit_after_n_batches (int, optional): Maximal batch iteration, after which the PCA is fitted. Defaults to 5.

        Returns:
            Union[List, Generator]: List with all embeddings or generator that yields the embeddings.
        """

        return self._reduce_batch(
            documents,
            as_generator,
            True,
            fit_after_n_batches,
        )

    def transform(self, documents, as_generator=False) -> Union[List, Generator]:
        return self._reduce_batch(documents, as_generator, False, False, 0)
