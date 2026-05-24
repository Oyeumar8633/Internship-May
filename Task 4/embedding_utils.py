"""Shared embedding factory — HuggingFaceEmbeddings with Colab-safe fallback."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
CONFIG_PATH = Path("embedding_config.json")


def repair_torch_stack() -> None:
    """Fix broken torch/torchvision pairs common after pip upgrades on Colab."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "torch",
            "torchvision",
            "torchaudio",
            "sentence-transformers",
        ],
        check=False,
    )


def _try_huggingface_embeddings() -> Any:
    import sentence_transformers  # noqa: F401
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _fastembed_embeddings() -> Any:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastembed"], check=False)
    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=FASTEMBED_MODEL)


def _save_config(backend: str, model: str) -> None:
    CONFIG_PATH.write_text(json.dumps({"backend": backend, "model": model}), encoding="utf-8")


def get_embeddings() -> Any:
    """
    Primary: HuggingFaceEmbeddings (all-MiniLM-L6-v2).
    Fallback: FastEmbed (ONNX) if sentence-transformers/torch is broken on Colab.
    """
    try:
        emb = _try_huggingface_embeddings()
        _save_config("huggingface", EMBEDDING_MODEL)
        logger.info("Embeddings: HuggingFaceEmbeddings (%s)", EMBEDDING_MODEL)
        return emb
    except Exception as exc:
        logger.warning("HuggingFaceEmbeddings failed (%s). Repairing torch stack…", exc)

    repair_torch_stack()
    try:
        emb = _try_huggingface_embeddings()
        _save_config("huggingface", EMBEDDING_MODEL)
        logger.info("Embeddings: HuggingFaceEmbeddings after torch repair")
        return emb
    except Exception as exc:
        logger.warning("Still failing (%s). Using FastEmbed fallback.", exc)

    emb = _fastembed_embeddings()
    _save_config("fastembed", FASTEMBED_MODEL)
    logger.info("Embeddings: FastEmbedEmbeddings (%s)", FASTEMBED_MODEL)
    return emb


def load_embeddings() -> Any:
    """Reload the same embedding backend used when building the FAISS index."""
    if not CONFIG_PATH.exists():
        return get_embeddings()

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    backend = cfg.get("backend", "huggingface")
    model = cfg.get("model", EMBEDDING_MODEL)

    if backend == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model)

    from langchain_community.embeddings import FastEmbedEmbeddings

    return FastEmbedEmbeddings(model_name=model)
