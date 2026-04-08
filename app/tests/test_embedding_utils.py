import numpy as np

from roop.utils.io import compute_cosine_distance


def test_compute_cosine_distance_flattens_batched_embeddings():
    emb1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    emb2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    distance = compute_cosine_distance(emb1, emb2)

    assert distance == 0.0


def test_compute_cosine_distance_returns_mismatch_distance_for_invalid_shapes():
    emb1 = np.array([[1.0, 0.0]], dtype=np.float32)
    emb2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    distance = compute_cosine_distance(emb1, emb2)

    assert distance == 1.0
