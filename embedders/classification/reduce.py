import numpy as np
from embedders import PCAReducer, util


class PCASentenceReducer(PCAReducer):
    def _transform(self, embedding_batch):
        return self.reducer.transform(embedding_batch).tolist()

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
                        embeddings_training_flattened.extend(batch_training)
                    self.reducer.fit(np.array(embeddings_training_flattened))

                    for batch_training in embeddings_training:
                        yield self._transform(batch_training)
                if batch_idx > fit_batches:
                    yield self._transform(batch)
        else:
            embeddings = self.embedder.transform(documents)
            yield self._transform(embeddings)
