![embedders](https://uploads-ssl.webflow.com/61e47fafb12bd56b40022a49/626ee1c35a3abf0ca872486d_embedder-banner.png)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![pypi 0.1.3](https://img.shields.io/badge/pypi-0.1.3-red.svg)](https://pypi.org/project/embedders/0.1.3/)

# ⚗️ embedders

With `embedders`, you can easily convert your texts into sentence- or token-level embeddings within a few lines of code. Use cases for this include similarity search between texts, information extraction such as named entity recognition, or basic text classification.

## Prerequisites

This library uses [spaCy](https://github.com/explosion/spaCy) for tokenization; to apply it, please download the [respective language model](https://spacy.io/models) first.

## Installation

You can set up this library via either running `$ pip install embedders`, or via cloning this repository and running `$ pip install -r requirements.txt` in your repository.

A sample installation would be:

```
$ conda create --name embedders python=3.9
$ conda activate embedders
$ pip install embedders
$ python -m spacy download en_core_web_sm
```

## Usage

Once you installed the package, you can apply the embedders with a few lines of code. You can apply embedders on sentence- or token-level.

### Sentence embeddings

`"Wow, what a cool tool!"` is embedded to

```
[
    2.453, 8.325, ..., 3.863
]
```

Currently, we provide the following sentence embeddings:
| **Path** | **Name** | **Embeds documents using ...** |
| ------------------------------------ | --------------------------- | ------------------------------------------------------------ |
| embedders.classification.contextual | HuggingFaceSentenceEmbedder | large, pre-trained transformers from https://huggingface.co |
| embedders.classification.contextual | OpenAISentenceEmbedder | large, pre-trained transformers from https://openai.com |
| embedders.classification.contextual | CohereSentenceEmbedder | large, pre-trained transformers from https://cohere.com |
| embedders.classification.count_based | BagOfCharsSentenceEmbedder | plain Bag of Chars approach |
| embedders.classification.count_based | BagOfWordsSentenceEmbedder | plain Bag of Words approach |
| embedders.classification.count_based | TfidfSentenceEmbedder | Term Frequency - Inverse Document Frequency (TFIDF) approach |

### Token embeddings

`"Wow, what a cool tool!"` is embedded to

```
[
    [8.453, 1.853, ...],
    [3.623, 2.023, ...],
    [1.906, 9.604, ...],
    [7.306, 2.325, ...],
    [6.630, 1.643, ...],
    [3.023, 4.974, ...]
]
```

Currently, we provide the following token embeddings:

| **Path**                         | **Name**                 | **Embeds documents using ...**                              |
| -------------------------------- | ------------------------ | ----------------------------------------------------------- |
| embedders.extraction.contextual  | TransformerTokenEmbedder | large, pre-trained transformers from https://huggingface.co |
| embedders.extraction.count_based | BagOfCharsTokenEmbedder  | plain Bag of Characters approach                            |

You can choose the embedding category depending on your task at hand. To implement them, you can just grab one of the available methods and apply them to your text corpus as follows (shown for sentence embeddings, but the same is possible for token):

```python
from embedders.classification.contextual import TransformerSentenceEmbedder
from embedders.classification.reduce import PCASentenceReducer

corpus = [
    "I went to Cologne in 2009",
    "My favorite number is 41",
    # ...
]

embedder = TransformerSentenceEmbedder("bert-base-cased")
embeddings = embedder.fit_transform(corpus) # contains a list of shape [num_texts, embedding_dimension]
```

Sometimes, you want to reduce the size of the embeddings you received. To do so, you can easily wrap your embedder with some dimensionality reduction technique.

```python
# if the dimension is too large, you can also apply dimensionality reduction
reducer = PCASentenceReducer(embedder)
embeddings_reduced = reducer.fit_transform(corpus)
```

Currently, we provide the following dimensionality reductions:
| **Path** | **Name** | **Description** |
| ------------------------------- | ------------------- | -------------------------------------------------------------------------------- |
| embedders.classification.reduce | PCASentenceEmbedder | Wraps embedder into a principial component analysis to reduce the dimensionality |
| embedders.extraction.reduce | PCATokenEmbedder | Wraps embedder into a principial component analysis to reduce the dimensionality |

## Pre-trained embedders

With growing availability of large, pre-trained models such as provided by [🤗 Hugging Face](https://huggingface.co/), embedding complex sentences in a wide variety of languages and domains becomes much more applicable. If you want to make use of transformer models, you can just use the configuration string of the respective model, which will automatically pull the correct model for the [🤗 Hugging Face Hub](https://huggingface.co/models).

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

And please don't forget to leave a ⭐ if you like the work!

## License

Distributed under the Apache 2.0 License. See LICENSE.txt for more information.

## Contact

This library is developed and maintained by [kern.ai](https://github.com/code-kern-ai). If you want to provide us with feedback or have some questions, don't hesitate to contact us. We're super happy to help ✌️
