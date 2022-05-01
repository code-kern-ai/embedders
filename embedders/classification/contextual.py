from sentence_transformers import SentenceTransformer
from embedders import util
from embedders.classification import SentenceEmbedder


class TransformerSentenceEmbedder(SentenceEmbedder):
    def __init__(self, config_string, batch_size=128):
        super().__init__(batch_size)
        self.model = SentenceTransformer(config_string)

    def _encode(self, documents, fit_model):
        for documents_batch in util.batch(documents, self.batch_size):
            yield self.model.encode(documents_batch, show_progress_bar=False).tolist()
