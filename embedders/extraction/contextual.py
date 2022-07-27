from typing import List, Tuple, Union, Iterator
import torch
import math
import numpy as np
import re
import string
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
    ) -> Iterator[List[List[List[float]]]]:
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
                documents_batch_embedded.append([lookup_w2v(tok.text) for tok in doc])
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(config_string)
        self.model = AutoModel.from_pretrained(
            config_string, output_hidden_states=True
        ).to(self.device)

    def _encode(
        self, documents: Union[List[str], List[Doc]], fit_model: bool
    ) -> Iterator[List[List[List[float]]]]:
        for documents_batch in util.batch(documents, self.batch_size):
            documents_batch_embedded = []
            for doc in documents_batch:
                doc = self._get_tokenized_document(doc)
                number_est_tokens = self._estimate_token_number(doc)
                # split the document if the estimated tokens are exceeding the
                # model's max input length
                if self.transformer_tokenizer.model_max_length < number_est_tokens:
                    transformer_embs = []
                    for doc_part, index_offset in self._split_document(
                        doc, number_est_tokens
                    ):
                        transformer_embs.extend(
                            self._get_transformer_embeddings(doc_part, index_offset)
                        )
                else:
                    transformer_embs = self._get_transformer_embeddings(doc.text)

                document_embedded = self._match_transformer_embeddings_to_spacy_tokens(
                    transformer_embs, doc
                )
                documents_batch_embedded.append(document_embedded)
            yield documents_batch_embedded

    def _match_transformer_embeddings_to_spacy_tokens(
        self,
        transformer_embeddings: List[List[Tuple[int, int, List[List[float]]]]],
        document_tokenized: Doc,
    ) -> List[List[float]]:
        embeddings = defaultdict(list)

        for index_start, index_end, char_embeddings in transformer_embeddings:
            span = document_tokenized.char_span(
                index_start, index_end, alignment_mode="expand"
            )
            if span is not None:
                token = span[0]
                embeddings[token.i].extend(char_embeddings)
        for key, values in embeddings.items():
            embeddings[key] = np.array(values).mean(0).tolist()
        return list(embeddings.values())

    def _get_transformer_embeddings(
        self,
        document: str,
        index_offset: int = 0,
    ) -> List[List[Tuple[int, int, List[List[float]]]]]:
        encoded = self.transformer_tokenizer(document, return_tensors="pt").to(
            self.device
        )
        tokens = encoded.encodings[0]
        with torch.no_grad():
            output = self.model(**encoded)

        # Get all hidden states
        states = output.hidden_states
        # Stack and sum last four layers
        layers = [-4, -3, -2, -1]
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

        token_embeddings = []
        # 1 and -1 are [CLS] tokens, and other tokens can be ##subwords
        for word_idx in sorted(set(tokens.words[1:-1])):
            index_begin, index_end = tokens.word_to_chars(word_idx)
            token_ids_word = np.where(np.array(encoded.word_ids()) == word_idx)
            # Only select the tokens that constitute the requested word
            word_tokens_output = output[token_ids_word]
            token_embeddings.append(
                [
                    index_begin + index_offset,
                    index_end + index_offset,
                    word_tokens_output.tolist(),
                ]
            )
        return token_embeddings

    def _estimate_token_number(self, doc: Doc) -> int:
        """
        Estimates the number of tokens which are generated by the transformer model.
        It is based on the rule of thumb that per spacy token 3 subtokens are created by
        the transformer tokenizer.
        Special Characters are handled seperately according to the assumption that each
        special character is treated as a token by the transformer tokenizer.
        """
        avg_subtokens_per_spacy_token = 3
        number_spacy_tokens = len(
            [token for token in doc if token.text not in string.punctuation]
        )
        number_special_characters = len(re.sub(r"[\w\s]+", "", doc.text))
        return (
            avg_subtokens_per_spacy_token * number_spacy_tokens
            + number_special_characters
        )

    def _split_document(
        self, document: Doc, estimated_tokens: int
    ) -> Iterator[Tuple[str, int]]:

        split_into = math.ceil(
            estimated_tokens / self.transformer_tokenizer.model_max_length
        )
        len_part = math.ceil(len(document) / split_into)

        prev_split_idx = 0
        for i in range(split_into):
            split_at_word = min(len(document) - 1, (i + 1) * len_part)
            current_split_idx = document[split_at_word].idx + len(
                document[split_at_word]
            )
            yield document.text[prev_split_idx:current_split_idx], prev_split_idx
            prev_split_idx = current_split_idx
