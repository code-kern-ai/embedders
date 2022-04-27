import spacy
from embedders import Embedder


class TokenEmbedder(Embedder):
    def __init__(self, language_code, precomputed_docs=False, batch_size=128):
        self.preloaded = precomputed_docs
        if precomputed_docs:
            self.nlp = spacy.blank(language_code)
        else:
            self.nlp = spacy.load(language_code)
        self.batch_size = batch_size

    def get_tokenized_document(self, document):
        if self.preloaded:
            return document
        else:
            return self.nlp(document)

    def encode(self, documents, as_generator=False, fit_model=True):
        if as_generator:
            return self._encode(documents, fit_model)
        else:
            embeddings = []
            for embedding_batch in self._encode(documents, fit_model):
                embeddings.extend(embedding_batch)
            return embeddings
