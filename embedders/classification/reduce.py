import numpy as np
from embedders import PCAReducer


class PCASentenceReducer(PCAReducer):
    def transform_batch(self, embedding_batch):
        return self.reducer.transform(embedding_batch).tolist()

    def _fit_transform(self, documents, fit_batches):
        embeddings_training = []
        num_batches = self.embedder.num_batches(documents)
        fit_batches = min(num_batches, fit_batches) - 1
        for batch_idx, batch in enumerate(
            list(self.embedder.encode(documents, as_generator=True))
        ):
            if batch_idx <= fit_batches:
                embeddings_training.append(batch)

            if batch_idx == fit_batches:
                embeddings_training_flattened = []
                for batch_training in embeddings_training:
                    embeddings_training_flattened.extend(batch_training)
                self.reducer.fit(np.array(embeddings_training_flattened))

                for batch_training in embeddings_training:
                    yield self.transform_batch(batch_training)
            if batch_idx > fit_batches:
                yield self.transform_batch(batch)
