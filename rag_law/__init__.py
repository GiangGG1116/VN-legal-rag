"""RAG system for Vietnamese Laws & Codes (UTS_VLC corpus).

Core safety-oriented components:
- corpus:     load UTS_VLC 2026 split + vn_law.sqlite catalogue
- parsing:    structure-aware parsing (Chương / Điều / khoản)
- classify:   question classification (scope, intent)
- abstain:    "know when to say I don't know" decision engine
- safety:     input guard, output validator, audit log
- citations:  doc-number / article reference extraction & validation
- retrieval:  lexical retriever (stand-in until hybrid index is built)
- pipeline:   end-to-end query handling
"""

from rag_law.corpus import Corpus, DocMeta
from rag_law.classify import QuestionClassifier, Classification
from rag_law.abstain import decide, Decision, RetrievalHit
from rag_law.pipeline import RAGPipeline, PipelineResult

__all__ = [
    "Corpus",
    "DocMeta",
    "QuestionClassifier",
    "Classification",
    "decide",
    "Decision",
    "RetrievalHit",
    "RAGPipeline",
    "PipelineResult",
]

__version__ = "0.1.0"
