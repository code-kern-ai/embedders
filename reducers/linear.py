import numpy as np
from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, embedder, n_components=8):
        self.embedder = embedder
        self.reducer = PCA(n_components=n_components)

    def transform(self, documents):
        embeddings = self.embedder.encode(documents, fit_model=False)
        return self.transform_batch(embeddings)

    def transform_batch(self, embedding_batch):
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

    def fit_transform(self, documents, fit_batches=5, as_generator=False):
        if as_generator:
            return self._fit_transform(documents, fit_batches)
        else:
            embeddings = []
            for embedding_batch in self._fit_transform(documents, fit_batches):
                embeddings.extend(embedding_batch)
            return embeddings

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
                    embeddings_training_flattened.extend(
                        np.concatenate(batch_training).tolist()
                    )
                self.reducer.fit(np.array(embeddings_training_flattened))

                for batch_training in embeddings_training:
                    yield self.transform_batch(batch_training)
            if batch_idx > fit_batches:
                yield self.transform_batch(batch)
