from abc import ABC, abstractmethod

from embedders import Embedder


class TokenEmbedder(Embedder):
    def get_tokenized_document(self, document):
        if self.preloaded:
            return document
        else:
            return self.nlp(document)
