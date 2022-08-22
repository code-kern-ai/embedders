#!/usr/bin/env python
import os

from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md")) as file:
    long_description = file.read()

setup(
    name="embedders",
    version="0.0.16",
    author="Johannes Hötter",
    author_email="johannes.hoetter@kern.ai",
    description="High-level API for creating sentence and token embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/code-kern-ai/embedders",
    keywords=["kern", "machine learning", "representation learning", "python"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    package_dir={"": "."},
    packages=find_packages("."),
    install_requires=[
        "huggingface-hub",
        "nltk",
        "numpy",
        "scikit-learn",
        "scipy",
        "sentence-transformers",
        "sentencepiece",
        "spacy>=3.0.0",
        "tokenizers>=0.10.3",
        "torch>=1.6.0",
        "tqdm",
        "transformers>=4.6.0,<5.0.0",
    ],
)
