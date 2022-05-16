from typing import List, Tuple, Union, Generator
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from gensim.models import word2vec
from embedders import util
from spacy.tokens.doc import Doc


from embedders.extraction import TokenEmbedder


class SkipGramTokenEmbedder(TokenEmbedder):
    """Embeds documents using a word2vec approach from gensim.

    Args:
        language_code (str): Name of the spaCy language model
        precomputed_docs (bool, optional): If you have a large text corpus, it might make sense to precompute the data and input tokenized spaCy documents. Defaults to False.
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
    """

    def __init__(
        self, language_code: str, precomputed_docs: bool = False, batch_size: int = 128
    ):
        super().__init__(language_code, precomputed_docs, batch_size)
        self.model = None

    def _encode(
        self, documents: Union[List[str], List[Doc]], fit_model: bool
    ) -> Generator[List[List[List[float]]], None, None]:
        def lookup_w2v(text: str) -> List[float]:
            try:
                return self.model.wv[text].tolist()
            except KeyError:
                return [0.0 for _ in range(self.model.vector_size)]

        if not self.preloaded:
            documents = [self.nlp(doc) for doc in documents]
            vocabulary = []
            for doc in documents:
                vocabulary.append([tok.text for tok in doc])
            if fit_model:
                self.model = word2vec.Word2Vec(vocabulary, min_count=1)

        for documents_batch in util.batch(documents, self.batch_size):
            documents_batch_embedded = []
            for doc in documents_batch:
                documents_batch_embedded.append(
                    [lookup_w2v(tok.text) for tok in doc])
            yield documents_batch_embedded


class TransformerTokenEmbedder(TokenEmbedder):
    """Embeds documents using large, pre-trained transformers from https://huggingface.co

    Args:
        config_string (str): Name of the model listed on https://huggingface.co/models
        language_code (str): Name of the spaCy language model
        precomputed_docs (bool, optional): If you have a large text corpus, it might make sense to precompute the data and input tokenized spaCy documents. Defaults to False.
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
    """

    def __init__(
        self,
        config_string: str,
        language_code: str,
        precomputed_docs: bool = False,
        batch_size: int = 128,
    ):
        super().__init__(language_code, precomputed_docs, batch_size)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            config_string
        )
        self.model = AutoModel.from_pretrained(
            config_string, output_hidden_states=True
        ).to(self.device)

    def _encode(
        self, documents: Union[List[str], List[Doc]], fit_model: bool
    ) -> Generator[List[List[List[float]]], None, None]:
        for documents_batch in util.batch(documents, self.batch_size):
            documents_batch_embedded = []
            for doc in documents_batch:
                char_level_embs = self._get_char_level_embeddings(str(doc))
                doc = self._get_tokenized_document(doc)
                document_embedded = self._get_token_embedding_from_char_embedding(
                    char_level_embs, doc
                )
                documents_batch_embedded.append(document_embedded)
            yield documents_batch_embedded

    def _get_token_embedding_from_char_embedding(
        self,
        char_level_embeddings: List[List[Tuple[int, int, List[List[float]]]]],
        document_tokenized: Doc,
    ) -> List[List[float]]:
        embeddings = defaultdict(list)

        for index_start, index_end, char_embeddings in char_level_embeddings:
            span = document_tokenized.char_span(
                index_start, index_end, alignment_mode="expand"
            )
            if span is not None:
                token = span[0]
                embeddings[token.i].extend(char_embeddings)
        for key, values in embeddings.items():
            embeddings[key] = np.array(values).mean(0).tolist()
        return list(embeddings.values())

    def _get_char_level_embeddings(
        self, document: str
    ) -> List[List[Tuple[int, int, List[List[float]]]]]:
        encoded = self.transformer_tokenizer.encode_plus(
            document, return_tensors="pt")
        tokens = encoded.encodings[0]
        num_tokens = len(
            set(tokens.words[1:-1])
        )  # 1 and -1 are [CLS] tokens, and other tokens can be ##subwords
        with torch.no_grad():
            output = self.model(**encoded)

        # Get all hidden states
        states = output.hidden_states
        # Stack and sum last four layers
        layers = [-4, -3, -2, -1]
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

        token_embeddings = []

        for token_idx in range(num_tokens):
            index_begin, index_end = tokens.word_to_chars(token_idx)
            token_ids_word = np.where(
                np.array(encoded.word_ids()) == token_idx)
            # Only select the tokens that constitute the requested word
            word_tokens_output = output[token_ids_word]
            token_embeddings.append(
                [index_begin, index_end, word_tokens_output.tolist()]
            )
        return token_embeddings
