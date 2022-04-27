# embedders
With embedders, you can easily convert your text into sentence- or token-level embeddings within a few lines of code. Use cases for this include similarity search between texts, information extraction such as named entity recognition, or basic text classification.

## Example
*Calculating sentence embeddings*
```python
from embedders.classification.contextual import TransformerSentenceEmbedder
from embedders.classification.reduce import PCASentenceReducer

corpus = [
    "I went to Cologne in 2009",
    "My favorite number is 41",
    ...
]

embedder = TransformerSentenceEmbedder("bert-base-cased")
embeddings = embedder.encode(corpus) # contains a list of shape [num_texts, embedding_dimension]

# if the dimension is too large, you can also apply dimensionality reduction
reducer = PCASentenceReducer(embedder)
embeddings_reduced = reducer.fit_transform(corpus)
```

*Calculating token embeddings*
```python
from embedders.extraction.count_based import CharacterTokenEmbedder
from embedders.extraction.reduce import PCATokenReducer

corpus = [
    "I went to Cologne in 2009",
    "My favorite number is 41",
    ...
]

embedder = CharacterTokenEmbedder("en_core_web_sm")
embeddings = embedder.encode(corpus) # contains a list of ragged shape [num_texts, num_tokens (text-specific), embedding_dimension]

# if the dimension is too large, you can also apply dimensionality reduction
reducer = PCATokenReducer(embedder)
embeddings_reduced = reducer.fit_transform(corpus)
```

## How to contribute
Currently, the best way to contribute is via adding issues for the kind of transformations you like and starring this repository :-)
