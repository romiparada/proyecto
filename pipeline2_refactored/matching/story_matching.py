"""Dimensión 2 — Similitud de Solución (Story Matching)."""

from typing import List, Dict, Optional

from ..similarity.metrics import compute_similarity_matrix, top_k_matches
from ..config.settings import PipelineConfig


def evaluate_story_matching(
    generated: List[str],
    expected: List[str],
    encoder,
    config: PipelineConfig,
    ca_generated: Optional[List[List[str]]] = None,
    ca_expected: Optional[List[List[str]]] = None,
) -> Dict:
    """Empareja cada historia generada con las top-k esperadas más similares."""
    emb_gen = encoder.encode(generated, convert_to_tensor=True)
    emb_exp = encoder.encode(expected, convert_to_tensor=True)
    sim_matrix = compute_similarity_matrix(emb_gen, emb_exp)

    results = []
    for i, gen_text in enumerate(generated):
        matches = top_k_matches(sim_matrix, i, expected, k=config.top_k)
        best = matches[0] if matches else {"index": 0, "text": "", "similarity": 0.0}
        results.append({
            "generated_index": i + 1,
            "generated_story": gen_text,
            "best_expected_index": best["index"],
            "best_expected_story": best["text"],
            "best_similarity": best["similarity"],
            "top_matches": matches,
        })

    similarities = [r["best_similarity"] for r in results]
    avg_sim = round(sum(similarities) / len(similarities), 4) if similarities else 0.0

    return {
        "results": results,
        "summary": {
            "total_generated_stories": len(results),
            "average_best_similarity": avg_sim,
        },
    }
