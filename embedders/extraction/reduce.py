import numpy as np
from embedders import PCAReducer, util


class PCATokenReducer(PCAReducer):
    def __init__(self, embedder, **kwargs):
        super().__init__(embedder=embedder, **kwargs)
        self.nlp = embedder.nlp

    def _transform(self, embedding_batch):
        batch_concatenated = np.concatenate(embedding_batch)
        start_idx = 0
        batch_unsqueezed = []
        for length in [len(embedding) for embedding in embedding_batch]:
            end_idx = start_idx + length
            batch_reduced = self.reducer.transform(
                batch_concatenated[start_idx:end_idx]
            )
            batch_unsqueezed.append(batch_reduced.tolist())
            start_idx = end_idx
        return batch_unsqueezed

    def _reduce(self, documents, fit_model, fit_batches):
        if fit_model:
            embeddings_training = []
            num_batches = util.num_batches(documents, self.embedder.batch_size)
            fit_batches = min(num_batches, fit_batches) - 1
            for batch_idx, batch in enumerate(
                list(self.embedder.fit_transform(documents, as_generator=True))
            ):
                if batch_idx <= fit_batches:
                    embeddings_training.append(batch)

                if batch_idx == fit_batches:
                    embeddings_training_flattened = []
                    for batch_training in embeddings_training:
                        embeddings_training_flattened.extend(
                            np.concatenate(batch_training).tolist()
                        )
                    self.reducer.fit(np.array(embeddings_training_flattened))

                    for batch_training in embeddings_training:
                        yield self._transform(batch_training)
                if batch_idx > fit_batches:
                    yield self._transform(batch)
        else:
            embeddings = self.embedder.transform(documents)
            yield self._transform(embeddings)
