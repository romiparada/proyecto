"""Dimensión 3 — Detección de Alucinaciones.

Clasifica las historias generadas según su similitud con las esperadas:
  - aligned:               similitud >= umbral alto
  - uncertain:             similitud >= umbral bajo
  - possible_hallucination: similitud < umbral bajo
"""

from typing import List, Dict

from ..config.settings import PipelineConfig


def classify_hallucination(score: float, config: PipelineConfig) -> str:
    if score >= config.hallucination_threshold_aligned:
        return "aligned"
    elif score >= config.hallucination_threshold_uncertain:
        return "uncertain"
    else:
        return "possible_hallucination"


def detect_hallucinations(
    matching_results: List[Dict],
    config: PipelineConfig,
) -> Dict:
    #Clasifica cada historia generada según su mejor similitud con las esperadas
    results = []
    for r in matching_results:
        best_score = r["best_similarity"]
        flag = classify_hallucination(best_score, config)
        results.append({
            "generated_index": r["generated_index"],
            "generated_story": r["generated_story"],
            "hallucination_flag": flag,
            "best_similarity": best_score,
            "best_expected_index": r["best_expected_index"],
            "best_expected_story": r["best_expected_story"],
        })

    # Conteos por categoría
    counts = {"aligned": 0, "uncertain": 0, "possible_hallucination": 0}
    for r in results:
        counts[r["hallucination_flag"]] += 1
    total = len(results)

    summary = {
        "total_stories": total,
        **counts,
        "alignment_rate": round(counts["aligned"] / total, 4) if total else 0.0,
        "uncertain_rate": round(counts["uncertain"] / total, 4) if total else 0.0,
        "hallucination_rate": round(counts["possible_hallucination"] / total, 4) if total else 0.0,
    }

    return {"results": results, "summary": summary}
