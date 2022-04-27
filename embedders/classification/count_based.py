from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from embedders.classification import SentenceEmbedder


class CountSentenceEmbedder(SentenceEmbedder):
    def __init__(self, batch_size, min_df):
        super().__init__(batch_size)

    def _encode(self, documents, fit_model):
        if fit_model:
            self.model.fit(documents)

        for documents_batch in self.batch(documents):
            documents_batch_embedded = []
            for doc in documents_batch:
                documents_batch_embedded.append(
                    self.model.transform([doc]).toarray().tolist()[0]
                )
            yield documents_batch_embedded


class CharacterSentenceEmbedder(CountSentenceEmbedder):
    def __init__(self, batch_size=128, min_df=0.1):
        super().__init__(batch_size, min_df)
        self.model = CountVectorizer(analyzer="char", min_df=min_df)


class BagofWordsSentenceEmbedder(CountSentenceEmbedder):
    def __init__(self, batch_size=128, min_df=0.1):
        super().__init__(batch_size, min_df)
        self.model = CountVectorizer(min_df=min_df)


class TfidfSentenceEmbedder(CountSentenceEmbedder):
    def __init__(self, batch_size=128, min_df=0.1):
        super().__init__(batch_size, min_df)
        self.model = TfidfVectorizer(min_df=min_df)
