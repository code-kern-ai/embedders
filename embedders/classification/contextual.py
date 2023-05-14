from typing import List, Union, Generator
from sentence_transformers import SentenceTransformer
from embedders import util
from embedders.classification import SentenceEmbedder
from spacy.tokens.doc import Doc
import torch
import openai
from openai import error as openai_error
import cohere


class TransformerSentenceEmbedder(SentenceEmbedder):
    """Embeds documents using large, pre-trained transformers from https://huggingface.co

    Args:
        config_string (str): Name of the model listed on https://huggingface.co/models
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
    """

    def __init__(self, config_string: str, batch_size: int = 128):
        super().__init__(batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(config_string).to(self.device)

    def _encode(
        self, documents: List[Union[str, Doc]], fit_model: bool
    ) -> Generator[List[List[float]], None, None]:
        for documents_batch in util.batch(documents, self.batch_size):
            yield self.model.encode(documents_batch, show_progress_bar=False).tolist()


class HuggingFaceSentenceEmbedder(TransformerSentenceEmbedder):
    def __init__(self, config_string: str, batch_size: int = 128):
        super().__init__(config_string, batch_size)


class OpenAISentenceEmbedder(SentenceEmbedder):
    def __init__(self, openai_api_key: str, model_name: str, batch_size: int = 128):
        super().__init__(batch_size)
        self.model_name = model_name
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key

    def _encode(
        self, documents: List[Union[str, Doc]], fit_model: bool
    ) -> Generator[List[List[float]], None, None]:
        for documents_batch in util.batch(documents, self.batch_size):
            documents_batch = [doc.replace("\n", " ") for doc in documents_batch]
            try:
                response = openai.Embedding.create(
                    input=documents_batch, model=self.model_name
                )
                embeddings = [entry["embedding"] for entry in response["data"]]
                yield embeddings
            except openai_error.AuthenticationError:
                raise Exception(
                    "OpenAI API key is invalid. Please provide a valid API key in the constructor of OpenAISentenceEmbedder."
                )


class CohereSentenceEmbedder(SentenceEmbedder):
    def __init__(self, cohere_api_key: str, batch_size: int = 128):
        super().__init__(batch_size)
        self.cohere_api_key = cohere_api_key
        self.model = cohere.Client(self.cohere_api_key)

    def _encode(
        self, documents: List[Union[str, Doc]], fit_model: bool
    ) -> Generator[List[List[float]], None, None]:
        for documents_batch in util.batch(documents, self.batch_size):
            embeddings = self.model.embed(documents_batch).embeddings
            yield embeddings
