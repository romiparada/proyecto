
from typing import List, Dict, Tuple

from ..similarity.metrics import compute_similarity_matrix, top_k_matches
from ..config.settings import PipelineConfig


def classify_coverage(score: float, config: PipelineConfig) -> str:
    if score >= config.coverage_threshold_covered:
        return "covered"
    elif score >= config.coverage_threshold_partial:
        return "possibly_covered"
    else:
        return "not_covered"


def evaluate_functional_coverage(
    aspects: List[Tuple[str, str]],
    generated: List[str],
    encoder,
    config: PipelineConfig,
) -> Dict:
    aspect_texts = [text for _, text in aspects]

    emb_aspects = encoder.encode(aspect_texts, convert_to_tensor=True)
    emb_gen = encoder.encode(generated, convert_to_tensor=True)

    sim_matrix = compute_similarity_matrix(emb_aspects, emb_gen)

    results = []
    for i, (category, aspect_text) in enumerate(aspects):
        matches = top_k_matches(sim_matrix, i, generated, k=config.top_k)
        best_score = matches[0]["similarity"] if matches else 0.0
        status = classify_coverage(best_score, config)

        results.append({
            "functionality_index": i,
            "category": category,
            "functionality": aspect_text,
            "coverage_status": status,
            "best_match": {
                "story_index": matches[0]["index"] if matches else -1,
                "story_text": matches[0]["text"] if matches else "",
                "similarity": best_score,
            },
            "top_matches": matches,
        })

    counts = {"covered": 0, "possibly_covered": 0, "not_covered": 0}
    for r in results:
        counts[r["coverage_status"]] += 1
    total = len(results)

    summary = {
        "total_functionalities": total,
        **counts,
        "coverage_rate": round(counts["covered"] / total, 4) if total else 0.0,
        "partial_rate": round(counts["possibly_covered"] / total, 4) if total else 0.0,
        "uncovered_rate": round(counts["not_covered"] / total, 4) if total else 0.0,
    }

    return {"results": results, "summary": summary}
