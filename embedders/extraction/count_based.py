from typing import List, Generator, Union
from sklearn.feature_extraction.text import CountVectorizer
from embedders import util
from spacy.tokens.doc import Doc

from embedders.extraction import TokenEmbedder


class BagOfCharsTokenEmbedder(TokenEmbedder):
    """Embeds documents using plain Bag of Characters approach.

    Args:
        language_code (str): Name of the spaCy language model
        precomputed_docs (bool, optional): If you have a large text corpus, it might make sense to precompute the data and input tokenized spaCy documents. Defaults to False.
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
    """

    def __init__(
        self,
        language_code: str,
        precomputed_docs: bool = False,
        batch_size: int = 128,
        **kwargs
    ):
        super().__init__(language_code, precomputed_docs, batch_size)
        self.model = CountVectorizer(analyzer="char", min_df=0.01, **kwargs)

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
