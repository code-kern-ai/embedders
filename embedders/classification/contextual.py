from typing import List, Union, Generator
from sentence_transformers import SentenceTransformer
from embedders import util
from embedders.classification import SentenceEmbedder
from spacy.tokens.doc import Doc
import torch


class TransformerSentenceEmbedder(SentenceEmbedder):
    """Embeds documents using large, pre-trained transformers from https://huggingface.co

    Args:
        config_string (str): Name of the model listed on https://huggingface.co/models
        batch_size (int, optional): Defines the number of conversions after which the embedder yields. Defaults to 128.
    """

    def __init__(self, config_string: str, batch_size: int = 128):

        super().__init__(batch_size)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model = SentenceTransformer(config_string).to(self.device)

    def _encode(
        self, documents: List[Union[str, Doc]], fit_model: bool
    ) -> Generator[List[List[float]], None, None]:
        for documents_batch in util.batch(documents, self.batch_size):
            yield self.model.encode(documents_batch, show_progress_bar=False).tolist()
