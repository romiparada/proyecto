"""Pipeline2 Refactored — punto de entrada CLI.

Ejecuta las tres dimensiones de evaluación end-to-end.
"""

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .embeddings import create_encoder
from .data_io import load_stories, load_criteria, load_aspects, align_criteria_to_stories
from .data_io import (
    save_coverage_report,
    save_matching_report,
    save_hallucination_report,
    save_ca_alignment_report,
    save_semantic_graph,
    print_summary,
)
from .evaluator import evaluate_ca_alignment
from .evaluator import evaluate_functional_coverage
from .evaluator import evaluate_story_matching
from .evaluator import detect_hallucinations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de evaluación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Entradas requeridas
    parser.add_argument("--generadas", required=True,
                        help="Path a las historias generadas (.txt, una por línea)")
    parser.add_argument("--esperadas", required=True,
                        help="Path a las historias esperadas (.txt, una por línea)")
    parser.add_argument("--aspectos", required=True,
                        help="Path a las funcionalidades del dominio (.json)")

    # Entradas opcionales
    parser.add_argument("--ca-gen", default=None, dest="ca_gen",
                        help="Path a los criterios de aceptación generados (.json)")
    parser.add_argument("--ca-esp", default=None, dest="ca_esp",
                        help="Path a los criterios de aceptación esperados (.json)")

    # Salida
    parser.add_argument("--output", default="resultados_pipeline2_refactored",
                        help="Directorio de salida (default: resultados_pipeline2_refactored/)")

    # Modelo y overrides de config
    parser.add_argument("--model", default=None,
                        help="Nombre del modelo SBERT (default: all-mpnet-base-v2)")
    parser.add_argument("--device", default=None,
                        help="Device: 'cpu' or 'cuda' (auto-detect if omitted)")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k",
                        help="Número de top matches a recuperar (default: 3)")

    # Overrides de umbrales
    parser.add_argument("--coverage-threshold", type=float, default=None,
                        dest="coverage_threshold",
                        help="Similarity threshold for 'covered' status (default: 0.75)")
    parser.add_argument("--hallucination-threshold", type=float, default=None,
                        dest="hallucination_threshold",
                        help="Similarity threshold for 'aligned' status (default: 0.75)")
    parser.add_argument("--ca-alignment-threshold", type=float, default=None,
                        dest="ca_alignment_threshold",
                        help="Min story similarity to qualify for CA-level alignment (default: 0.60)")

    # Evaluación de Cobertura con LLM
    parser.add_argument("--llm-judge", action="store_true",
                        help="Run LLM-as-a-judge for functional coverage evaluation")

    # Módulo de refinamiento LLM
    parser.add_argument("--refinement", action="store_true",
                        help="Run LLM-assisted post-processing to refine generated stories")
    parser.add_argument("--prd", type=str, default=None,
                        help="Path to PRD text file (required if --refinement is set)")
    parser.add_argument("--api-key", type=str, default=None, dest="api_key",
                        help="API key for the external LLM provider")
    parser.add_argument("--llm-model", type=str, default="gpt-4o",
                        dest="llm_model",
                        help="External LLM model name (default: gpt-4o)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validar rutas requeridas
    for flag, path_str in [("--generadas", args.generadas),
                           ("--esperadas", args.esperadas),
                           ("--aspectos", args.aspectos)]:
        if not Path(path_str).exists():
            print(f"Error: archivo no encontrado para {flag}: {path_str}", file=sys.stderr)
            return 1

    for flag, path_str in [("--ca-gen", args.ca_gen), ("--ca-esp", args.ca_esp)]:
        if path_str and not Path(path_str).exists():
            print(f"Error: archivo no encontrado para {flag}: {path_str}", file=sys.stderr)
            return 1
            
    # Validar LLM Judge
    if args.llm_judge and not args.api_key:
        print("Error: --api-key requerido si se usa --llm-judge", file=sys.stderr)
        return 1

    # Validar requisitos de refinamiento
    if args.refinement:
        if not args.prd or not Path(args.prd).exists():
            print("Error: --prd requerido y debe existir si se usa --refinement", file=sys.stderr)
            return 1
        if not args.api_key:
            print("Error: --api-key requerido si se usa --refinement", file=sys.stderr)
            return 1

    # Construir config con overrides de CLI
    config = PipelineConfig()
    if args.model:
        config.model_name = args.model
    if args.device:
        config.device = args.device
    if args.top_k:
        config.top_k = args.top_k
    if args.coverage_threshold:
        config.coverage_threshold_covered = args.coverage_threshold
    if args.hallucination_threshold:
        config.hallucination_threshold_aligned = args.hallucination_threshold
    if args.ca_alignment_threshold:
        config.ca_alignment_story_threshold = args.ca_alignment_threshold

    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Cargar entradas ---
    print("\n--- Cargando entradas ---")
    generated = load_stories(args.generadas)
    expected = load_stories(args.esperadas)
    aspects = load_aspects(args.aspectos)

    ca_generated = None
    ca_expected = None
    if args.ca_gen and args.ca_esp:
        ca_generated = load_criteria(args.ca_gen)
        ca_expected = load_criteria(args.ca_esp)
        ca_generated = align_criteria_to_stories(ca_generated, generated, "CA generated")
        ca_expected = align_criteria_to_stories(ca_expected, expected, "CA expected")

    print(f"  Historias generadas : {len(generated)}")
    print(f"  Historias esperadas : {len(expected)}")
    print(f"  Funcionalidades     : {len(aspects)}")
    if ca_generated:
        print(f"  CA generados      : {len(ca_generated)} entradas")
    if ca_expected:
        print(f"  CA esperados      : {len(ca_expected)} entradas")

    # --- Inicializar encoder ---
    encoder = create_encoder(config.model_name, config.device)

    # --- Dimensión 1: Cobertura Funcional ---
    print("\n--- Dimensión 1: Cobertura Funcional ---")
    coverage_data = evaluate_functional_coverage(
        aspects, generated, encoder, config,
        use_llm_judge=args.llm_judge,
        api_key=args.api_key,
        llm_model=args.llm_model
    )
    print(f"  Cobertura calculada para {len(aspects)} funcionalidades")

    # --- Dimensión 2: Similitud de solución ---
    print("\n--- Dimensión 2: Similitud de Solución (Story Matching) ---")
    matching_data = evaluate_story_matching(
        generated, expected, encoder, config,
        ca_generated=ca_generated,
        ca_expected=ca_expected,
    )
    print(f"  {len(generated)} historias generadas comparadas contra {len(expected)} esperadas")

    # --- Dimensión 2b: Alineación a nivel de CA ---
    ca_alignment_data = None
    if ca_generated is not None and ca_expected is not None:
        print("\n--- Dimensión 2b: Alineación de CA ---")
        ca_alignment_data = evaluate_ca_alignment(
            matching_results=matching_data["results"],
            ca_generated=ca_generated,
            ca_expected=ca_expected,
            encoder=encoder,
            config=config,
        )
        matching_data["ca_alignment"] = ca_alignment_data
        s = ca_alignment_data["summary"]
        print(f"  Historias evaluadas: {s['stories_evaluated']}, omitidas: {s['stories_skipped']}")
        print(f"  Tasa CA: {s['pair_match_rate']:.1%} ({s['pairs_matching']}/{s['total_ca_pairs_compared']} pares)")

    # --- Dimensión 3: Detección de alucinaciones ---
    print("\n--- Dimensión 3: Detección de Alucinaciones ---")
    hallucination_data = detect_hallucinations(matching_data["results"], config)
    print(f"  {len(generated)} historias clasificadas")

    # --- Guardar reportes ---
    print("\n--- Guardando reportes ---")
    coverage_files = save_coverage_report(coverage_data, output_dir)
    matching_files = save_matching_report(matching_data, output_dir)
    hallucination_files = save_hallucination_report(hallucination_data, output_dir)
    graph_files = save_semantic_graph(aspects, coverage_data, matching_data, output_dir)

    ca_alignment_files = {}
    if ca_alignment_data is not None:
        ca_alignment_files = save_ca_alignment_report(ca_alignment_data, output_dir)

    all_files = {**coverage_files, **matching_files, **hallucination_files, **graph_files, **ca_alignment_files}
    for label, path in all_files.items():
        print(f"  [+] {path}")

    # --- Refinamiento LLM ---
    if args.refinement:
        print("\n--- Refinamiento Asistido por LLM ---")
        try:
            from .refinement import run_refinement
            import json
            
            prd_text = Path(args.prd).read_text(encoding="utf-8")
            print(f"  Llamando a {args.llm_model} para sugerencias...")
            refinement_report = run_refinement(
                generated_stories=generated,
                functionalities=aspects,
                coverage_data=coverage_data,
                matching_data=matching_data,
                hallucination_data=hallucination_data,
                prd_text=prd_text,
                api_key=args.api_key,
                model=args.llm_model
            )
            
            ref_path = output_dir / "refinement_report.json"
            with open(ref_path, "w", encoding="utf-8") as f:
                json.dump(refinement_report, f, indent=2, ensure_ascii=False)
            print(f"  [+] {ref_path}")
        except Exception as e:
            print(f"  [!] Error en refinamiento: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # --- Resumen en consola y archivo ---
    print_summary(coverage_data, matching_data, hallucination_data, output_dir=output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
