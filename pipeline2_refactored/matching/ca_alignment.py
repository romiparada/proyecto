"""Dimensión 2b — Alineación a nivel de Criterios de Aceptación (CA).

Para cada historia con similitud >= umbral, compara cada CA generado
contra todos los CA de la historia esperada mejor rankeada (rank 1).
Se preserva la matriz completa (gen_CA × exp_CA) sin reducción.
"""

from typing import List, Dict
from sentence_transformers import util

from ..config.settings import PipelineConfig


def _classify(score: float, config: PipelineConfig) -> str:
    if score >= config.criteria_threshold_matching:
        return "matching"
    elif score >= config.criteria_threshold_low:
        return "low_similarity"
    else:
        return "missing"


def _compare_ca_matrix(
    gen_ca: List[str],
    exp_ca: List[str],
    encoder,
    config: PipelineConfig,
) -> List[Dict]:
    #Compara cada CA generado contra todos los CA esperados. Retorna lista plana de pares.
    if not gen_ca or not exp_ca:
        return []

    emb_gen = encoder.encode(gen_ca, convert_to_tensor=True)
    emb_exp = encoder.encode(exp_ca, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb_gen, emb_exp)  # (|gen_ca| × |exp_ca|)

    comparisons = []
    for gi, g_criterion in enumerate(gen_ca):
        for ei, e_criterion in enumerate(exp_ca):
            score = float(sim_matrix[gi][ei].item())
            comparisons.append({
                "generated_criterion": g_criterion,
                "generated_index": gi + 1,
                "expected_criterion": e_criterion,
                "expected_index": ei + 1,
                "similarity": round(score, 4),
                "status": _classify(score, config),
            })

    return comparisons


def evaluate_ca_alignment(
    matching_results: List[Dict],
    ca_generated: List[List[str]],
    ca_expected: List[List[str]],
    encoder,
    config: PipelineConfig,
) -> Dict:
    #Alineación CA para historias por encima del umbral de similitud.
    results = []
    skipped = []

    for story_result in matching_results:
        gen_idx = story_result["generated_index"]
        best_sim = story_result["best_similarity"]

        # Omitir historias por debajo del umbral
        if best_sim < config.ca_alignment_story_threshold:
            skipped.append({
                "generated_index": gen_idx,
                "generated_story": story_result["generated_story"],
                "best_story_similarity": best_sim,
                "reason": "below_story_threshold",
            })
            continue

        top_matches = story_result.get("top_matches", [])
        if not top_matches:
            skipped.append({
                "generated_index": gen_idx,
                "generated_story": story_result["generated_story"],
                "best_story_similarity": best_sim,
                "reason": "no_top_matches",
            })
            continue

        # Historia esperada rank-1
        best_match = top_matches[0]
        exp_idx = best_match["index"]

        gen_ca: List[str] = ca_generated[gen_idx - 1] if gen_idx - 1 < len(ca_generated) else []
        exp_ca: List[str] = ca_expected[exp_idx - 1] if 1 <= exp_idx <= len(ca_expected) else []

        # Comparación cruzada completa gen_CA × exp_CA
        ca_comparisons = _compare_ca_matrix(gen_ca, exp_ca, encoder, config)

        results.append({
            "generated_index": gen_idx,
            "generated_story": story_result["generated_story"],
            "best_story_similarity": best_sim,
            "best_match_used": {
                "rank": best_match["rank"],
                "expected_index": exp_idx,
                "expected_story": best_match["text"],
                "story_similarity": best_match["similarity"],
            },
            "generated_ca": gen_ca,
            "expected_ca": exp_ca,
            "ca_comparisons": ca_comparisons,
        })

    #Estadísticas
    total_pairs = 0
    pairs_matching = pairs_low = pairs_missing = 0
    for r in results:
        for c in r["ca_comparisons"]:
            total_pairs += 1
            if c["status"] == "matching":
                pairs_matching += 1
            elif c["status"] == "low_similarity":
                pairs_low += 1
            else:
                pairs_missing += 1

    summary = {
        "stories_evaluated": len(results),
        "stories_skipped": len(skipped),
        "ca_alignment_story_threshold": config.ca_alignment_story_threshold,
        "total_ca_pairs_compared": total_pairs,
        "pairs_matching": pairs_matching,
        "pairs_low_similarity": pairs_low,
        "pairs_missing": pairs_missing,
        "pair_match_rate": round(pairs_matching / total_pairs, 4) if total_pairs else 0.0,
    }

    return {"results": results, "skipped": skipped, "summary": summary}
