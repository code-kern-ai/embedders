from typing import List, Generator, Union
import numpy as np
from embedders import PCAReducer, util


class PCATokenReducer(PCAReducer):
    def __init__(self, embedder, **kwargs):
        super().__init__(embedder=embedder, **kwargs)
        self.nlp = embedder.nlp

    def _transform(
        self, embeddings: List[List[List[Union[int, float]]]]
    ) -> List[List[List[Union[float, int]]]]:
        batch_concatenated = np.concatenate(embeddings)
        start_idx = 0
        batch_unsqueezed = []
        for length in [len(embedding) for embedding in embeddings]:
            end_idx = start_idx + length
            batch_reduced = self.reducer.transform(
                batch_concatenated[start_idx:end_idx]
            )
            batch_unsqueezed.append(batch_reduced.tolist())
            start_idx = end_idx
        return batch_unsqueezed

    def _reduce(
        self, documents, fit_model, fit_after_n_batches
    ) -> Generator[List[List[List[Union[float, int]]]], None, None]:
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
                        embeddings_training_flattened.extend(
                            np.concatenate(batch_training).tolist()
                        )
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
