import spacy
from sklearn.feature_extraction.text import CountVectorizer

from embedders.extraction import TokenEmbedder


class CharacterTokenEmbedder(TokenEmbedder):
    def __init__(self, language_code, precomputed_docs=False, batch_size=128):
        self.model = CountVectorizer(analyzer="char", min_df=0.01)

        self.preloaded = precomputed_docs
        if precomputed_docs:
            self.nlp = spacy.blank(language_code)
        else:
            self.nlp = spacy.load(language_code)
        self.batch_size = batch_size

    def batch_encode(self, documents):
        if self.preloaded:
            self.model.fit([doc.text for doc in documents])
        else:
            self.model.fit(documents)
        for documents_batch in self.batch(documents):
            documents_batch_embedded = []
            for doc in documents_batch:
                documents_batch_embedded.append(
                    self.model.transform(
                        [tok.text for tok in self.get_tokenized_document(doc)]
                    )
                    .toarray()
                    .tolist()
                )
            yield documents_batch_embedded
