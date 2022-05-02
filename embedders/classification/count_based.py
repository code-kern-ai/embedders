from typing import List, Union, Generator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from embedders.classification import SentenceEmbedder
from embedders import util


class CountSentenceEmbedder(SentenceEmbedder):
    def __init__(self, batch_size: int, min_df: float, **kwargs):
        super().__init__(batch_size)

    def _encode(
        self, documents: List[str], fit_model: bool
    ) -> Generator[List[List[Union[float, int]]], None, None]:
        if fit_model:
            self.model.fit(documents)

        for documents_batch in util.batch(documents, self.batch_size):
            documents_batch_embedded = []
            for doc in documents_batch:
                documents_batch_embedded.append(
                    self.model.transform([doc]).toarray().tolist()[0]
                )
            yield documents_batch_embedded


class BagOfCharsSentenceEmbedder(CountSentenceEmbedder):
    """Embeds documents using plain Bag of Characters approach.

    Args:
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
        min_df (float, optional): When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. Defaults to 0.1.
    """

    def __init__(self, batch_size: int = 128, min_df: float = 0.1, **kwargs):
        super().__init__(batch_size, min_df)
        self.model = CountVectorizer(analyzer="char", min_df=min_df, **kwargs)


class BagOfWordsSentenceEmbedder(CountSentenceEmbedder):
    """Embeds documents using plain Bag of Words approach.

    Args:
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
        min_df (float, optional): When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. Defaults to 0.1.
    """

    def __init__(self, batch_size: int = 128, min_df: float = 0.1, **kwargs):
        super().__init__(batch_size, min_df)
        self.model = CountVectorizer(min_df=min_df, **kwargs)


class TfidfSentenceEmbedder(CountSentenceEmbedder):
    """Embeds documents using Term Frequency - Inverse Document Frequency (TFIDF) approach.

    Args:
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
        min_df (float, optional): When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float in range of [0.0, 1.0], the parameter represents a proportion of documents, integer absolute counts. Defaults to 0.1.
    """

    def __init__(self, batch_size: int = 128, min_df: float = 0.1, **kwargs):
        super().__init__(batch_size, min_df)
        self.model = TfidfVectorizer(min_df=min_df, **kwargs)
