from typing import List, Union, Generator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from embedders.classification import SentenceEmbedder
from embedders import util


class CountSentenceEmbedder(SentenceEmbedder):
    def __init__(self, batch_size: int, min_df: float):
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
            print(documents_batch_embedded)
            yield documents_batch_embedded


class BagOfCharsSentenceEmbedder(CountSentenceEmbedder):
    def __init__(self, batch_size: int = 128, min_df: float = 0.1):
        super().__init__(batch_size, min_df)
        self.model = CountVectorizer(analyzer="char", min_df=min_df)


class BagOfWordsSentenceEmbedder(CountSentenceEmbedder):
    def __init__(self, batch_size: int = 128, min_df: float = 0.1):
        super().__init__(batch_size, min_df)
        self.model = CountVectorizer(min_df=min_df)


class TfidfSentenceEmbedder(CountSentenceEmbedder):
    def __init__(self, batch_size: int = 128, min_df: float = 0.1):
        super().__init__(batch_size, min_df)
        self.model = TfidfVectorizer(min_df=min_df)
