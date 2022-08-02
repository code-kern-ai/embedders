from embedders import Embedder


class SentenceEmbedder(Embedder):
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size
