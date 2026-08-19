"""Parsers de entradas del pipeline (historias, criterios de aceptación, aspectos)."""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Union


def load_stories(path: Union[str, Path]) -> List[str]:
    """Carga historias de usuario desde un .txt (una por línea no vacía)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        stories = [line.strip() for line in f if line.strip()]
    if not stories:
        raise ValueError(f"{path.name}: no stories found (file is empty or all blank lines).")
    return stories


def load_criteria(path: Union[str, Path]) -> List[List[str]]:
    """Carga criterios de aceptación desde JSON (lista de listas de strings).

    Las entradas con texto placeholder se normalizan a listas vacías.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a JSON array at the top level.")

    criteria: List[List[str]] = []
    placeholder_phrases = {"no acceptance criteria defined", "n/a", "none", ""}

    for i, entry in enumerate(raw):
        if not isinstance(entry, list):
            raise ValueError(f"{path.name}: la entrada en índice {i} debe ser una lista de strings.")
        # Filtrar entradas placeholder
        cleaned = [
            c.strip() for c in entry
            if c.strip().lower() not in placeholder_phrases
        ]
        criteria.append(cleaned)

    return criteria


def load_aspects(path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Carga funcionalidades del dominio desde JSON.

    Soporta dos formatos:
      - dict: {"categoria": ["aspecto1", ...], ...}
      - list: ["aspecto1", "aspecto2", ...]

    Retorna lista plana de tuplas (categoria, texto).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        result = []
        for category, items in raw.items():
            if isinstance(items, list):
                for item in items:
                    result.append((category, str(item).strip()))
            else:
                result.append((category, str(items).strip()))
        return result
    elif isinstance(raw, list):
        return [("general", str(item).strip()) for item in raw]
    else:
        raise ValueError(f"{path.name}: expected a JSON object or array.")


def align_criteria_to_stories(
    criteria: List[List[str]],
    stories: List[str],
    label: str = "criteria",
) -> List[List[str]]:
    """Ajusta el largo de la lista de criterios para que coincida con el de historias."""
    n_criteria = len(criteria)
    n_stories = len(stories)
    if n_criteria < n_stories:
        print(
            f"  ⚠ {label} has {n_criteria} entries but {n_stories} stories "
            f"— padding with empty lists."
        )
        criteria.extend([] for _ in range(n_stories - n_criteria))
    elif n_criteria > n_stories:
        print(
            f"  ⚠ {label} has {n_criteria} entries but only {n_stories} stories "
            f"— truncating to match."
        )
        criteria = criteria[:n_stories]
    return criteria


"""Reporters de salida — escritores JSON y CSV para todos los reportes."""

import json
import csv
from pathlib import Path
from typing import Dict, List

SIMILARITY_THRESHOLD = 0.60

def _save_json(data: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─── Dimensión 1: Cobertura Funcional ─────────────────────────

def save_coverage_report(coverage_data: Dict, output_dir: Path) -> Dict[str, str]:
    """Guarda reporte de cobertura en JSON y CSV."""
    json_path = output_dir / "coverage_report.json"
    csv_path = output_dir / "coverage_report.csv"

    _save_json(coverage_data, json_path)

    # Aplanar para CSV
    csv_rows = []
    for r in coverage_data["results"]:
        row = {
            "functionality_index": r["functionality_index"],
            "category": r["category"],
            "functionality": r["functionality"],
            "coverage_status": r["coverage_status"],
            "sbert_status": r.get("sbert_status", ""),
            "llm_reasoning": r.get("llm_reasoning", ""),
            "best_story_index": r["best_match"]["story_index"],
            "best_story_text": r["best_match"]["story_text"],
            "best_similarity": r["best_match"]["similarity"],
        }
        # Columnas top-k
        for m in r["top_matches"]:
            rank = m["rank"]
            row[f"top_{rank}_index"] = m["index"]
            row[f"top_{rank}_story"] = m["text"]
            row[f"top_{rank}_similarity"] = m["similarity"]
        csv_rows.append(row)

    _save_csv(csv_rows, csv_path)
    return {"json": str(json_path), "csv": str(csv_path)}


# ─── Dimensión 2: Story Matching ───────────────────────────────

def save_matching_report(matching_data: Dict, output_dir: Path) -> Dict[str, str]:
    """Guarda reporte de matching en JSON y CSV."""
    json_path = output_dir / "matching_report.json"
    csv_path = output_dir / "matching_report.csv"

    _save_json(matching_data, json_path)

    # Aplanar para CSV
    csv_rows = []
    for r in matching_data["results"]:
        row = {
            "generated_index": r["generated_index"],
            "generated_story": r["generated_story"],
            "best_expected_index": r["best_expected_index"],
            "best_expected_story": r["best_expected_story"],
            "best_similarity": r["best_similarity"],
        }
        # Matches top-k
        for m in r["top_matches"]:
            rank = m["rank"]
            row[f"top_{rank}_expected_index"] = m["index"]
            row[f"top_{rank}_expected_story"] = m["text"]
            row[f"top_{rank}_similarity"] = m["similarity"]

        csv_rows.append(row)

    _save_csv(csv_rows, csv_path)
    return {"json": str(json_path), "csv": str(csv_path)}


# ─── Grafo de Similitud Semántica ──────────────────────────────

def save_semantic_graph(
    aspects: list,
    coverage_data: dict,
    matching_data: dict,
    output_dir: "Path",
) -> Dict[str, str]:
    """Construye grafo semántico tripartito. Compatible con Gephi."""
    nodes: Dict[str, Dict] = {}
    edges: list = []

    # ── Nodos de aspectos ───────────────────────────────────────────
    for idx, asp in enumerate(aspects):
        node_id = f"asp_{idx}"
        category = asp.get("category", "") if isinstance(asp, dict) else ""
        label = asp.get("functionality", str(asp)) if isinstance(asp, dict) else str(asp)
        nodes[node_id] = {
            "key": node_id,
            "id": node_id,
            "type": "aspect",
            "category": category,
            "label": label,
        }

    # ── Nodos de historias generadas ───────────────────────────────
    gen_seen: Dict[int, str] = {}
    for r in coverage_data.get("results", []):
        for m in r.get("top_matches", []):
            s_idx = m["index"]
            if s_idx not in gen_seen:
                node_id = f"gen_{s_idx}"
                gen_seen[s_idx] = node_id
                nodes[node_id] = {
                    "key": node_id,
                    "id": node_id,
                    "type": "generated_story",
                    "label": m["text"],
                }

    # ── Aristas: Aspecto → Historia Generada (cobertura) ────────────────
    for r in coverage_data.get("results", []):
        asp_idx = r["functionality_index"]
        asp_id = f"asp_{asp_idx}"
        for m in r.get("top_matches", []):
            if m["similarity"] < SIMILARITY_THRESHOLD:
                continue

            s_idx = m["index"]
            gen_id = gen_seen.get(s_idx, f"gen_{s_idx}")
            edges.append({
                "source": asp_id,
                "target": gen_id,
                "weight": round(float(m["similarity"]), 4),
                "type": "coverage",
                "rank": m["rank"],
            })

    # ── Nodos esperados + aristas: Generada → Esperada (matching) ────────
    exp_seen: Dict[int, str] = {}
    for r in matching_data.get("results", []):
        gen_idx = r["generated_index"]
        gen_id = f"gen_{gen_idx}"
        if gen_id not in nodes:
            nodes[gen_id] = {
                "key": gen_id,
                "id": gen_id,
                "type": "generated_story",
                "label": r["generated_story"],
            }

        for m in r.get("top_matches", []):
            if m["similarity"] < SIMILARITY_THRESHOLD:
                continue
            e_idx = m["index"]
            if e_idx not in exp_seen:
                exp_node_id = f"exp_{e_idx}"
                exp_seen[e_idx] = exp_node_id
                nodes[exp_node_id] = {
                    "key": exp_node_id,
                    "id": exp_node_id,
                    "type": "expected_story",
                    "label": m["text"],
                }
            exp_node_id = exp_seen[e_idx]
            edges.append({
                "source": gen_id,
                "target": exp_node_id,
                "weight": round(float(m["similarity"]), 4),
                "type": "matching",
                "rank": m["rank"],
            })

    graph = {
        "meta": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": ["aspect", "generated_story", "expected_story"],
            "edge_types": ["coverage", "matching"],
        },
        "nodes": list(nodes.values()),
        "edges": edges,
    }

    json_path = output_dir / "semantic_graph.json"
    _save_json(graph, json_path)
    
    # ── Exportar a CSVs de Gephi ────────────────────────────────────────
    nodes_csv_path = output_dir / "semantic_graph_nodes.csv"
    edges_csv_path = output_dir / "semantic_graph_edges.csv"
    
    # CSV de nodos
    node_rows = []
    for n in nodes.values():
        node_rows.append({
            "Id": n["id"],
            "Label": n["label"],
            "Type": n.get("type", ""),
            "Category": n.get("category", "")
        })
    _save_csv(node_rows, nodes_csv_path)
    
    # CSV de aristas
    edge_rows = []
    for e in edges:
        edge_rows.append({
            "Source": e["source"],
            "Target": e["target"],
            "Weight": float(e["weight"]),
            "Type": "Undirected",
            "relation_type": e.get("type", "")
        })
    _save_csv(edge_rows, edges_csv_path)

    return {
        "semantic_graph": str(json_path),
        "gephi_nodes": str(nodes_csv_path),
        "gephi_edges": str(edges_csv_path)
    }


# ─── Dimensión 2b: Alineación de CA ───────────────────────────────

def save_ca_alignment_report(ca_alignment_data: Dict, output_dir: Path) -> Dict[str, str]:
    """Guarda reporte de alineación CA en JSON y dos CSVs."""
    json_path = output_dir / "ca_alignment_report.json"
    # CSV 1: Matriz de comparación CA generados vs CA esperados
    matrix_csv_path = output_dir / "ca_alignment_matrix.csv"
    matrix_rows = []
    for r in ca_alignment_data.get("results", []):
        for c in r["ca_comparisons"]:
            matrix_rows.append({
                "generated_index": r["generated_index"],
                "generated_story": r["generated_story"],
                "best_story_similarity": r["best_story_similarity"],
                "expected_story_index": r["best_match_used"]["expected_index"],
                "generated_criterion": c["generated_criterion"],
                "expected_criterion": c["expected_criterion"],
                "similarity": c["similarity"],
                "status": c["status"],
            })
    _save_csv(matrix_rows, matrix_csv_path)

    # CSV 2: Resumen de CA por historia
    summary_csv_path = output_dir / "ca_alignment_story_summary.csv"
    summary_rows = []
    for r in ca_alignment_data.get("results", []):
        story_comps = r.get("ca_comparisons", [])
        total_pairs = len(story_comps)
        matching = sum(1 for c in story_comps if c["status"] == "matching")
        low = sum(1 for c in story_comps if c["status"] == "low_similarity")
        missing = sum(1 for c in story_comps if c["status"] == "missing")

        summary_rows.append({
            "generated_index": r["generated_index"],
            "generated_story": r["generated_story"],
            "best_story_similarity": r["best_story_similarity"],
            "expected_story_index": r["best_match_used"]["expected_index"],
            "expected_story": r["best_match_used"]["expected_story"],
            "generated_ca_count": len(r.get("generated_ca", [])),
            "expected_ca_count": len(r.get("expected_ca", [])),
            "total_ca_pairs": total_pairs,
            "pairs_matching": matching,
            "pairs_low_similarity": low,
            "pairs_missing": missing,
            "pair_match_rate": round(matching / total_pairs, 4) if total_pairs else 0.0,
        })
    _save_csv(summary_rows, summary_csv_path)

    return {
        "ca_alignment_json": str(json_path),
        "ca_alignment_matrix_csv": str(matrix_csv_path),
        "ca_alignment_summary_csv": str(summary_csv_path),
    }


# ─── Dimensión 3: Detección de Alucinaciones ────────────────────────

def save_hallucination_report(hallucination_data: Dict, output_dir: Path) -> Dict[str, str]:
    """Guarda reporte de alucinaciones en JSON y CSV."""
    json_path = output_dir / "hallucination_report.json"
    csv_path = output_dir / "hallucination_report.csv"

    _save_json(hallucination_data, json_path)

    csv_rows = []
    for r in hallucination_data["results"]:
        csv_rows.append({
            "generated_index": r["generated_index"],
            "generated_story": r["generated_story"],
            "hallucination_flag": r["hallucination_flag"],
            "best_similarity": r["best_similarity"],
            "best_expected_index": r["best_expected_index"],
            "best_expected_story": r["best_expected_story"],
        })

    _save_csv(csv_rows, csv_path)
    return {"json": str(json_path), "csv": str(csv_path)}


# ─── Resumen en consola ───────────────────────────────────────────

def print_summary(
    coverage_data: Dict,
    matching_data: Dict,
    hallucination_data: Dict,
    output_dir: Path = None,
) -> None:
    """Imprime resumen legible en consola y lo guarda en summary_report.md si se provee output_dir."""
    lines = []
    
    def out(text: str = ""):
        print(text)
        lines.append(text)

    sep = "=" * 65
    out(f"\n{sep}")
    out("PIPELINE2 REFACTORED — RESUMEN DE EVALUACIÓN")
    out(sep)

    # Dimensión 1
    cs = coverage_data["summary"]
    out("\n[DIM 1] Cobertura Funcional")
    out(f"  Total funcionalidades   : {cs['total_functionalities']}")
    
    if "covered_by_llm" in cs:
        out(f"  [LLM JUDGE] [OK] Cubiertas             : {cs['covered_by_llm']}")
        out(f"  [LLM JUDGE] [FAIL] No cubiertas        : {cs['not_covered_by_llm']}")
        out(f"  [LLM JUDGE] Cobertura Total         : {cs['coverage_percentage']:.1f}%")
        
        uncovered = [r for r in coverage_data["results"] if r["coverage_status"] == "not_covered_by_llm"]
        if uncovered:
            out("\n  -> Funcionalidades no cubiertas (según LLM):")
            for r in uncovered:
                out(f"     • [{r['category']}] {r['functionality'][:65]}\n       Razón: {r.get('llm_reasoning', '')[:80]}...")
    else:
        out(f"  [OK] Cubiertas             : {cs['covered']}  ({cs['coverage_rate']:.1%})")
        out(f"  [WARN] Posiblemente cubiertas: {cs['possibly_covered']}  ({cs['partial_rate']:.1%})")
        out(f"  [FAIL] No cubiertas          : {cs['not_covered']}  ({cs['uncovered_rate']:.1%})")

        uncovered = [r for r in coverage_data["results"] if r["coverage_status"] == "not_covered"]
        if uncovered:
            out("\n  -> Funcionalidades no cubiertas:")
            for r in uncovered:
                out(f"     • [{r['category']}] {r['functionality'][:65]}  (score={r['best_match']['similarity']})")

    # Dimensión 2
    ms = matching_data["summary"]
    out(f"\n[DIM 2] Similitud de Solución (Story Matching)")
    out(f"  Total historias generadas : {ms['total_generated_stories']}")
    out(f"  Similitud promedio        : {ms['average_best_similarity']}")

    # Dimensión 3
    hs = hallucination_data["summary"]
    out(f"\n[DIM 3] Detección de Alucinaciones")
    out(f"  Total historias              : {hs['total_stories']}")
    out(f"  [OK] Alineadas               : {hs['aligned']}  ({hs['alignment_rate']:.1%})")
    out(f"  [WARN] Inciertas             : {hs['uncertain']}  ({hs['uncertain_rate']:.1%})")
    out(f"  [FAIL] Posible alucinación   : {hs['possible_hallucination']}  ({hs['hallucination_rate']:.1%})")

    hallucinated = [r for r in hallucination_data["results"] if r["hallucination_flag"] == "possible_hallucination"]
    if hallucinated:
        out("\n  -> Posibles alucinaciones:")
        for r in hallucinated:
            preview = r["generated_story"][:70]
            out(f"     • [idx {r['generated_index']}] {preview}  (score={r['best_similarity']})")

    uncertain = [r for r in hallucination_data["results"] if r["hallucination_flag"] == "uncertain"]
    if uncertain:
        out("\n  -> Inciertas (revisar manualmente):")
        for r in uncertain:
            preview = r["generated_story"][:70]
            out(f"     • [idx {r['generated_index']}] {preview}  (score={r['best_similarity']})")

    # Alineación CA (opcional)
    ca_align = matching_data.get("ca_alignment")
    if ca_align:
        cas = ca_align["summary"]
        out(f"\n[DIM 2b] Alineación CA (historias >= {cas['ca_alignment_story_threshold']} similitud)")
        out(f"  Historias evaluadas     : {cas['stories_evaluated']}  (omitidas: {cas['stories_skipped']})")
        out(f"  — Matriz de comparación CA:")
        out(f"    Total pares comparados  : {cas['total_ca_pairs_compared']}")
        out(f"    [OK] Pares coincidentes : {cas['pairs_matching']}  ({cas['pair_match_rate']:.1%})")
        out(f"    [WARN] Baja similitud   : {cas['pairs_low_similarity']}")
        out(f"    [FAIL] Sin coincidencia : {cas['pairs_missing']}")

    out(f"\n{sep}\n")
    
    # Guardar a archivo
    if output_dir:
        report_path = output_dir / "summary_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  [+] Resumen guardado en {report_path}")
