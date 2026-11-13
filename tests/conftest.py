# -*- coding: utf-8 -*-
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_law.corpus import Corpus
from rag_law.retrieval import LexicalRetriever
from rag_law.pipeline import RAGPipeline


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return Corpus()


@pytest.fixture(scope="session")
def retriever(corpus):
    return LexicalRetriever(corpus)


@pytest.fixture(scope="session")
def pipeline(corpus, retriever) -> RAGPipeline:
    return RAGPipeline(corpus, retriever)
