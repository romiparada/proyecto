"""Utilidades de similitud coseno sobre tensores pre-computados."""

import torch
from sentence_transformers import util
from typing import List, Dict


def compute_similarity_matrix(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Matriz de similitud coseno (|emb_a| × |emb_b|)."""
    return util.cos_sim(emb_a, emb_b)


def top_k_matches(
    sim_matrix: torch.Tensor,
    row_idx: int,
    texts_b: List[str],
    k: int = 3,
) -> List[Dict]:
    """Top-k items más similares de texts_b para la fila row_idx."""
    sims = sim_matrix[row_idx]
    actual_k = min(k, len(texts_b))
    top_vals, top_idxs = torch.topk(sims, k=actual_k)

    return [
        {
            "rank": rank + 1,
            "index": int(idx) + 1,
            "text": texts_b[int(idx)],
            "similarity": round(float(score), 4),
        }
        for rank, (idx, score) in enumerate(
            zip(top_idxs.cpu().numpy(), top_vals.cpu().numpy())
        )
    ]
