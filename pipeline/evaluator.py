"""
Dimensión 1 — Cobertura Funcional.
"""

from typing import List, Dict, Tuple

from .embeddings import compute_similarity_matrix, top_k_matches
from .config import PipelineConfig
from .llm_judge import evaluate_coverage_with_llm


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
    use_llm_judge: bool = False,
    api_key: str = "",
    llm_model: str = "",
) -> Dict:
    """Calcula cobertura de cada funcionalidad respecto a las historias generadas."""
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

    # Resumen agregado
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

    coverage_data = {"results": results, "summary": summary}
    if use_llm_judge and api_key and llm_model:
        return evaluate_coverage_with_llm(coverage_data, api_key, llm_model)
    
    return coverage_data


"""Dimensión 2 — Similitud de Solución (Story Matching)."""

from typing import List, Dict, Optional



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
        best = matches[0] if matches else {"index": -1, "text": "", "similarity": 0.0}
        results.append({
            "generated_index": i,
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


"""Dimensión 2b — Alineación a nivel de Criterios de Aceptación (CA).

Para cada historia con similitud >= umbral, compara cada CA generado
contra todos los CA de la historia esperada mejor rankeada (rank 1).
Se preserva la matriz completa (gen_CA × exp_CA) sin reducción.
"""

from typing import List, Dict
from sentence_transformers import util



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
                "generated_index": gi,
                "expected_criterion": e_criterion,
                "expected_index": ei,
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

        gen_ca: List[str] = ca_generated[gen_idx] if gen_idx < len(ca_generated) else []
        exp_ca: List[str] = ca_expected[exp_idx] if 0 <= exp_idx < len(ca_expected) else []

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


"""Dimensión 3 — Detección de Alucinaciones.

Clasifica las historias generadas según su similitud con las esperadas:
  - aligned:               similitud >= umbral alto
  - uncertain:             similitud >= umbral bajo
  - possible_hallucination: similitud < umbral bajo
"""




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


