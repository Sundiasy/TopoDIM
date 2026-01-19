import os
from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=1)
def _get_sentence_transformer() -> SentenceTransformer:
    model_name_or_path = (
        os.getenv("SENTENCE_EMB_MODEL") or os.getenv("SENTENCE_TRANSFORMER_MODEL")
    )

    if not model_name_or_path:
        repo_root = Path(__file__).resolve().parents[2]
        local_model_dir = repo_root / "local"
        if local_model_dir.exists():
            model_name_or_path = str(local_model_dir)
        else:
            model_name_or_path = "sentence-transformers/all-MiniLM-L6-v2"

    return SentenceTransformer(model_name_or_path)


def get_sentence_embedding(sentence: str):
    model = _get_sentence_transformer()
    return model.encode(sentence)
