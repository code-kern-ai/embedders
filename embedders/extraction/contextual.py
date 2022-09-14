from typing import List, Tuple, Union, Iterator
import torch
import math
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from embedders import util
from spacy.tokens.doc import Doc


from embedders.enums import WarningType
from embedders.extraction import TokenEmbedder


class TransformerTokenEmbedder(TokenEmbedder):
    """Embeds documents using large, pre-trained transformers from https://huggingface.co

    Args:
        config_string (str): Name of the model listed on https://huggingface.co/models
        language_code (str): Name of the spaCy language model
        precomputed_docs (bool, optional): If you have a large text corpus, it might make sense to precompute the data and input tokenized spaCy documents. Defaults to False.
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
    """

    _NL_TOKEN = "[NL]"

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
        self.transformer_tokenizer.add_special_tokens(
            {"additional_special_tokens": [self._NL_TOKEN]}
        )

        self.model = AutoModel.from_pretrained(
            config_string, output_hidden_states=True
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.transformer_tokenizer))

    def _encode(
        self, documents: Union[List[str], List[Doc]], fit_model: bool
    ) -> Iterator[List[List[List[float]]]]:
        for batch_number, documents_batch in enumerate(
            util.batch(documents, self.batch_size)
        ):
            """
            Computation of embeddings for each spacy token of each document. For the
            embedding creation transformer models are used. Embeddings are calculated
            for the transformer tokens of the document, then these token embeddings are
            matched to the spacy tokens.

            Args:
                documents: list of strings or spacy documents
                fit_model: not used, required by base class
            Return:
                Token embeddings for each document
            """

            documents_batch_embedded = []
            for document_number, document in enumerate(documents_batch):
                doc = self._get_tokenized_document(document)

                # no spacy token.
                # set special token as text, so that an embedding
                # is created that can be processed by the PCA.
                if len(doc) == 0:
                    doc = self.nlp(self._NL_TOKEN)

                # spacy creates tokens which only contain whitespace characters.
                # the transformer tokenizer ignores these tokens.
                # in order to avoid problems while matching the tokens the text is
                # preprocessed.
                text, prep_offsets = self._preprocess_doc_text(doc)

                # transformer models have a maximum number of tokens which can be
                # processed at the same time.
                # in this case the text is splitted in mutliple subparts
                number_est_tokens = self._estimate_token_number(text)
                idx_document = batch_number * self.batch_size + document_number
                if self.model.config.max_position_embeddings < number_est_tokens:
                    if WarningType.DOCUMENT_IS_SPLITTED.value in self._warnings:
                        self._warnings[WarningType.DOCUMENT_IS_SPLITTED.value].append(
                            idx_document
                        )
                    else:
                        self._warnings[WarningType.DOCUMENT_IS_SPLITTED.value] = [
                            idx_document
                        ]

                    transformer_embs = []
                    for doc_part, index_offset in self._split_document(
                        text, number_est_tokens
                    ):
                        transformer_embs.extend(
                            self._get_transformer_embeddings(
                                doc_part, idx_document, index_offset
                            )
                        )
                else:
                    transformer_embs = self._get_transformer_embeddings(
                        text, idx_document
                    )

                document_embedded = self._match_transformer_embeddings_to_spacy_tokens(
                    transformer_embs, doc, prep_offsets
                )

                if len(document_embedded) != len(doc):
                    idx_document = batch_number * self.batch_size + document_number
                    if WarningType.TOKEN_MISMATCHING.value in self._warnings:
                        self._warnings[WarningType.TOKEN_MISMATCHING.value].append(
                            idx_document
                        )
                    else:
                        self._warnings[WarningType.TOKEN_MISMATCHING.value] = [
                            idx_document
                        ]

                documents_batch_embedded.append(document_embedded)
            yield documents_batch_embedded

    def _preprocess_doc_text(self, doc: Doc) -> Tuple[str, np.ndarray]:
        """Replaces the text of tokens which only consist of whitespace characters
        with the special token [NL] (new line).
        The special token and the whitespace string can consist of different number of
        chars. To match the tokens later these differences are saved as offsets.

        Args:
            doc: spacy document
        Returns:
            Preprocessed text in which whitespace tokens are
            replaced by a special token, an array containing the indices of replaced
            strings and the resulting offset
        """

        prep_text = ""
        idx_already_preprocessed = 0
        # pairs of the line number of the preprocessed text and the offset relative to
        # the original document, here, offset is the difference btw the preprocessed and
        # the original text.
        prep_offsets = [(0, 0)]

        for tkn in doc:
            if not re.sub(r"[\s]+", "", tkn.text):
                # indices of current token which will be replaced by the special token
                idx_start, idx_end = tkn.idx, tkn.idx + len(tkn)
                # append already processed text and the special token
                prep_text += doc.text[idx_already_preprocessed:idx_start]
                prep_text += self._NL_TOKEN

                additional_offset = len(tkn) - len(self._NL_TOKEN)
                prep_offsets.append(
                    (
                        len(prep_text),  # index to apply offset
                        additional_offset,  # offset to be applied
                    )
                )
                idx_already_preprocessed = idx_end

        # add remaining text
        prep_text += doc.text[idx_already_preprocessed:]

        return prep_text, np.array(prep_offsets)

    def _match_transformer_embeddings_to_spacy_tokens(
        self,
        transformer_embeddings: List[List[Tuple[int, int, List[List[float]]]]],
        document_tokenized: Doc,
        prep_offsets: np.ndarray = None,
    ) -> List[List[float]]:
        """
        Transformer and spacy tokens differ. Usual the transformer tokenizer splits
        splits the text into smaller subparts in comparison to the spacy tokenizer.
        To create embeddings for the spacy tokens the transformer embeddings must be
        matched. This is done by comparing the char spans of the tokens and matching the
        tokens which overlap.

        Args:
            transformer_embeddings: List of start and end indices for each transformer
                token and the corresponding embedding
            document_tokenized: spacy tokens
            prep_offsets: Indices and offsets to match the preprocessed text to the
                original document
        Returns:
            Embeddings for each spacy token in the tokenized document.
        """

        embeddings = defaultdict(list)

        for index_start, index_end, transformer_emb in transformer_embeddings:

            if prep_offsets is not None:
                index_start = self._add_offset(index_start, prep_offsets)
                index_end = self._add_offset(index_end, prep_offsets)

            span = document_tokenized.char_span(
                index_start, index_end, alignment_mode="expand"
            )
            if span is not None:
                # if a transformer token include multiple spacy tokens, the spacy
                # tokens get the same transformer embedding.
                for token in span:
                    embeddings[token.i].extend(transformer_emb)
        for key, values in embeddings.items():
            embeddings[key] = np.array(values).mean(0).tolist()
        return list(embeddings.values())

    def _add_offset(self, idx: int, offsets: np.ndarray) -> int:
        """
        Adds offset to index according to the offsets array.

        Args:
            idx: index to transform
            offsets: indices and the corresponding offsets
        Returns:
            Index customized according to the offset
        """
        return idx + np.sum(offsets[np.where(offsets[:, 0] <= idx)][:, 1])

    def _get_transformer_embeddings(
        self,
        document: str,
        idx_document: int,
        idx_offset: int = 0,
    ) -> List[List[Tuple[int, int, List[List[float]]]]]:
        """
        Calculates embeddings for the given document using a transformer model.
        First, the corresponding transformer tokens are computed. The next steps
        computes the embeddings. With each embedding the indices of the according
        chars are returned. idx_offset is used to return the correct indices if the
        document has been split.

        Args:
            document: plain document text
            idx_offset: offset if the document has been splitted
        Returns:
           Start and end index for each transformer token and the calculated
           embedding
        """
        encoded = self.transformer_tokenizer(document, return_tensors="pt").to(
            self.device
        )
        tokens = encoded.encodings[0]

        # fallback if the number of tokens is still too big
        if len(tokens) > self.model.config.max_position_embeddings:
            if WarningType.DOCUMENT_IS_SPLITTED.value in self._warnings:
                self._warnings[WarningType.DOCUMENT_IS_SPLITTED.value].append(
                    idx_document
                )
            else:
                self._warnings[WarningType.DOCUMENT_IS_SPLITTED.value] = [idx_document]

            token_embs = []
            for doc_part, additional_idx_offset in self._split_document(
                document, len(tokens)
            ):
                token_embs.extend(
                    self._get_transformer_embeddings(
                        doc_part, idx_document, idx_offset + additional_idx_offset
                    )
                )
            return token_embs

        with torch.no_grad():
            output = self.model(**encoded)

        # Get all hidden states
        states = output.hidden_states
        # Stack and sum last four layers
        layers = [-4, -3, -2, -1]
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

        token_embeddings = []
        # 1 and -1 are [CLS] tokens, and other tokens can be ##subwords
        for word_idx in set(tokens.word_ids[1:-1]):
            index_begin, index_end = tokens.word_to_chars(word_idx)
            token_ids_word = np.where(np.array(encoded.word_ids()) == word_idx)
            # Only select the tokens that constitute the requested word
            word_tokens_output = output[token_ids_word]
            token_embeddings.append(
                [
                    index_begin + idx_offset,
                    index_end + idx_offset,
                    word_tokens_output.tolist(),
                ]
            )
        return token_embeddings

    def _estimate_token_number(self, document: str) -> int:
        """
        Estimates the number of tokens which are generated by the transformer model.
        It is based on the rule of thumb that per token 3 subtokens are created by
        the transformer tokenizer. Tokens are created by splitting at every
        special and whitespace character.
        Special Characters are handled seperately according to the assumption that each
        special character is treated as a token by the transformer tokenizer.

        Args:
            document: plain text document
        Returns:
            Estimation for the number of transformer tokens included in the document
        """
        avg_subtokens_per_token = 3
        number_word_tokens = len(re.findall(r"\[NL\]|\w+", document))
        number_special_characters = len(re.sub(r"[\w\s]+", "", document))
        return avg_subtokens_per_token * number_word_tokens + number_special_characters

    def _split_document(
        self, document: str, estimated_tokens: int
    ) -> Iterator[Tuple[str, int]]:
        """
        Splits the documens into subparts, according to the model's max length and the
        number of estimated tokens.

        Args:
            document: plain text document
            estimated_tokens: estimation for the token number
        Returns:
            Yields subpart of the document, splitted depending on max model length and
            estimated number of tokens
        """
        # the regular expression matches the special token [NL], any word consiting of
        # numbers and chars and single characters which are no whitespace or word
        # character
        token_spans = [
            token.span() for token in re.finditer(r"\[NL\]|\w+|[^\w\s]+?", document)
        ]
        split_into = (
            round(estimated_tokens / self.model.config.max_position_embeddings) + 1
        )
        len_part = math.ceil(len(token_spans) / split_into)

        prev_split_idx = 0
        for i in range(split_into):
            current_split_idx = min(
                len(document),
                token_spans[min((i + 1) * len_part, len(token_spans) - 1)][1],
            )
            yield document[prev_split_idx:current_split_idx], prev_split_idx
            prev_split_idx = current_split_idx
