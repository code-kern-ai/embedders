from typing import List, Generator, Union
from sklearn.feature_extraction.text import CountVectorizer
from embedders import util
from spacy.tokens.doc import Doc

from embedders.extraction import TokenEmbedder


class BagOfCharsTokenEmbedder(TokenEmbedder):
    def __init__(
        self, language_code: str, precomputed_docs: bool = False, batch_size: int = 128
    ):
        super().__init__(language_code, precomputed_docs, batch_size)
        self.model = CountVectorizer(analyzer="char", min_df=0.01)

    def _encode(
        self, documents: List[Union[str, Doc]], fit_model: bool
    ) -> Generator[List[List[List[int]]], None, None]:
        if fit_model:
            if self.preloaded:
                self.model.fit([doc.text for doc in documents])
            else:
                self.model.fit(documents)

        for documents_batch in util.batch(documents, self.batch_size):
            documents_batch_embedded = []
            for doc in documents_batch:
                documents_batch_embedded.append(
                    self.model.transform(
                        [tok.text for tok in self._get_tokenized_document(doc)]
                    )
                    .toarray()
                    .tolist()
                )
            yield documents_batch_embedded
