import spacy
from embedders import Embedder
from spacy.tokens.doc import Doc
from typing import Union

class TokenEmbedder(Embedder):
    def __init__(self, language_code: str, precomputed_docs:bool=False, batch_size:int=128):
        self.preloaded = precomputed_docs
        if precomputed_docs:
            self.nlp = spacy.blank(language_code)
        else:
            self.nlp = spacy.load(language_code)
        self.batch_size = batch_size

    def _get_tokenized_document(self, document: Union[str, Doc]):
        if self.preloaded:
            return document
        else:
            return self.nlp(document)
            