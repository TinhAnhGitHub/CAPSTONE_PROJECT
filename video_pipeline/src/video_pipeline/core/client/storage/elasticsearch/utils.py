"""OCR text utilities for Elasticsearch indexing."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────

_HYPHEN_RE = re.compile(r"(\w+)-\s+(\w+)")
_MULTI_WS = re.compile(r"\s{2,}")


def clean_ocr_text(raw: str) -> str:
    """
    Clean OCR text by:
    - Unicode-normalizing (NFC)
    - Removing control chars
    - De-hyphenating line-break splits
    - Collapsing whitespace
    """
    text = unicodedata.normalize("NFC", raw)
    text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t")
    text = _HYPHEN_RE.sub(r"\1\2", text)
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


# ─────────────────────────────────────────────
# Index Mapping
# ─────────────────────────────────────────────

def get_ocr_index_mapping(embedding_dim: int = 384) -> dict[str, Any]:
    """
    Get the OCR index mapping with:
    - Custom analyzer for OCR text (stemming, synonym filter for OCR misreads)
    - Dense vector field for semantic search
    - Metadata fields for filtering
    """
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "ocr_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "asciifolding",
                            "english_stemmer",
                            "ocr_synonym_filter"
                        ]
                    }
                },
                "filter": {
                    "english_stemmer": {
                        "type": "stemmer",
                        "language": "english"
                    },
                    "ocr_synonym_filter": {
                        "type": "synonym",
                        "synonyms": [
                            "0 => o, 0",
                            "1 => l, i, 1",
                            "rn => m, rn"
                        ]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                # Searchable text fields
                "raw_text": {
                    "type": "text",
                    "analyzer": "ocr_analyzer",
                    "term_vector": "with_positions_offsets"
                },
                "cleaned_text": {
                    "type": "text",
                    "analyzer": "ocr_analyzer"
                },
                # Semantic vector field
                "content_vector": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "cosine"
                },
                # Metadata / filter fields from ImageOCRArtifact
                "artifact_id": {"type": "keyword"},
                "user_id": {"type": "keyword"},
                "frame_index": {"type": "integer"},
                "timestamp": {"type": "text"},
                "timestamp_sec": {"type": "float"},
                "video_id": {"type": "keyword"},
                "related_video_fps": {"type": "float"},
                "image_minio_url": {"type": "keyword"},
                "image_id": {"type": "keyword"},
                "indexed_at": {"type": "date"}
            }
        }
    }