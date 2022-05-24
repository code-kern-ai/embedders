from spacy.tokens.doc import Doc
from typing import Union, List, Generator
import numpy as np
from embedders import PCAReducer, util


class PCASentenceReducer(PCAReducer):
    def _transform(
        self, embeddings: List[List[Union[int, float]]]
    ) -> List[List[Union[float, int]]]:
        return self.reducer.transform(embeddings).tolist()

    def _reduce(
        self,
        documents: List[Union[str, Doc]],
        fit_model: bool,
        fit_after_n_batches: int,
    ) -> Generator[List[List[Union[float, int]]], None, None]:
        if fit_model:
            embeddings_training = []
            num_batches = util.num_batches(documents, self.embedder.batch_size)
            fit_after_n_batches = min(num_batches, fit_after_n_batches) - 1
            for batch_idx, batch in enumerate(
                self.embedder.fit_transform(documents, as_generator=True)
            ):
                if batch_idx <= fit_after_n_batches:
                    embeddings_training.append(batch)

                if batch_idx == fit_after_n_batches:
                    embeddings_training_flattened = []
                    for batch_training in embeddings_training:
                        embeddings_training_flattened.extend(batch_training)
                    embeddings_training_flattened = np.array(
                        embeddings_training_flattened
                    )
                    if (
                        embeddings_training_flattened.shape[1]
                        < self.reducer.n_components
                        and self.autocorrect_n_components
                    ):
                        self.reducer.n_components = embeddings_training_flattened.shape[
                            1
                        ]
                    self.reducer.fit(embeddings_training_flattened)

                    for batch_training in embeddings_training:
                        yield self._transform(batch_training)
                if batch_idx > fit_after_n_batches:
                    yield self._transform(batch)
        else:
            embeddings = self.embedder.transform(documents)
            yield self._transform(embeddings)
