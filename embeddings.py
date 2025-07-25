import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from helpers import compute_hash
from models import Embedding


logger = logging.getLogger(__name__)

# disable parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def write_npz_embeddings(filepath: Union[str, Path], items: List[Embedding]) -> None:
    import numpy as np
    contents = np.array([item.content for item in items], dtype=object)
    embeddings = np.array([item.embedding for item in items], dtype=np.float32)
    hashes = np.array([item.hash_val for item in items], dtype=object)
    models = np.array([item.model for item in items], dtype=object)

    np.savez_compressed(filepath, contents=contents, embeddings=embeddings, hashes=hashes, models=models)


def read_npz_embeddings(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    import numpy as np
    data = np.load(filepath, allow_pickle=True)
    contents = data["contents"]
    embeddings = data["embeddings"]
    hashes = data["hashes"]
    models = data["models"]

    return [
        Embedding(
            content=content,
            embedding=embedding,
            hash_val=hash_val,
            model=model
		)
        for content, embedding, hash_val, model in zip(contents, embeddings, hashes, models)
    ]


def check_and_update_embedding_items(
    raw_items: List[str],
    filepath: Union[str, Path],
    model: str,
    embed_func: Callable[[str], List[float]],
) -> List[Embedding]:
    filepath = Path(filepath)

    def compute_hashes(texts):
        return [compute_hash(text) for text in texts]

    if not filepath.exists():
        logger.info("No cached embeddings found. Generating embeddings...")
        embeddings = [
            Embedding(
				content=text,
				embedding=embed_func(text),
				hash_val=compute_hash(text),
				model=model
			)
            for text in raw_items
        ]
        write_npz_embeddings(filepath, embeddings)
        return embeddings

    try:
        cached = read_npz_embeddings(filepath)
        if (
            len(cached) != len(raw_items)
            or [c.hash_val for c in cached] != compute_hashes(raw_items)
            or any(e.model != model for e in cached)
            or any(not e.embedding for e in cached)
        ):
            logger.info("Detected changes or inconsistencies, regenerating embeddings.")
            embeddings = [
				Embedding(
					content=text,
					embedding=embed_func(text),
					hash_val=compute_hash(text),
					model=model
				)
				for text in raw_items
            ]
            write_npz_embeddings(filepath, embeddings)
            return embeddings
        return cached
    except Exception as e:
        logger.warning(f"Failed to read embedding cache: {e}, regenerating.")
        embeddings = [
            Embedding(
				content=text,
				embedding=embed_func(text),
				hash_val=compute_hash(text),
				model=model
			)
            for text in raw_items
        ]
        write_npz_embeddings(filepath, embeddings)
        return embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors using pure Python.

    Args:
        vec1: First vector (list of floats).
        vec2: Second vector (list of floats).

    Returns:
        Cosine similarity as a float between -1.0 and 1.0.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of same length")

    dot_prod = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0  # Define similarity with zero magnitude vector as 0

    return dot_prod / (norm1 * norm2)


def filter_by_vector_similarity(
    embeddings: List[Embedding],
    text_passage_embedding: List[float],
    similarity_threshold: float = 0.60
) -> List[str]:
    """
    Return content where embedding cosine similarity to text_passage_embedding
    is greater than or equal to similarity_threshold.
    """
    matching_content: Embedding = []
    for embedding in embeddings:
        similarity = cosine_similarity(embedding.embedding, text_passage_embedding)
        if similarity >= similarity_threshold:
            matching_content.append(embedding.content)
    return matching_content
