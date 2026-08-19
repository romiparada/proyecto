"""
Dimensión 3 — Detección de Alucinaciones
=========================================
Clasifica cada historia generada según qué tan respaldada está
por las historias esperadas, calculando los umbrales dinámicamente
en base a la revisión manual.
"""

import sys
import subprocess
from pathlib import Path
import csv
import json
import requests
original_request = requests.Session.request
def patched_request(self, method, url, *args, **kwargs):
    if url.startswith('/api/'):
        url = 'https://huggingface.co' + url
    return original_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_request

for pkg in ("sentence-transformers",):
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


from pipeline2_refactored.io.loaders import load_stories
from pipeline2_refactored.embeddings.encoder import create_encoder
from pipeline2_refactored.matching.story_matching import evaluate_story_matching
from pipeline2_refactored.hallucination.detector import detect_hallucinations
from pipeline2_refactored.config.settings import PipelineConfig

GENERADAS   = "datos/generadas.txt"
ESPERADAS   = "datos/esperadas.txt"
REVISION    = "revision_manual_alineacion.csv"
OUTPUT_DIR  = "resultados/alucinaciones"
MODEL_NAME  = "all-mpnet-base-v2"
DEVICE      = None
TOP_K       = 3

def main():
    generadas_path = Path(GENERADAS)
    esperadas_path = Path(ESPERADAS)
    revision_path  = Path(REVISION)
    output_dir     = Path(OUTPUT_DIR)

    for label, p in [("GENERADAS", generadas_path), ("ESPERADAS", esperadas_path)]:
        if not p.exists():
            print(f"Error: no se encontró el archivo {label}: {p.resolve()}")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar y parsear archivo de revisión manual para umbrales dinámicos
    aligned_scores = []
    absent_scores = []
    if revision_path.exists():
        with open(revision_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Usar keys seguras o fallback
                hist_id = row.get("ID Hist Generada", "").strip()
                if not hist_id:
                    hist_id = list(row.values())[0].strip()
                
                score_str = row.get("Similitud Maxima")
                matches = row.get("Correspondencias Reales (IDs)")
                if score_str is None:
                    score_str = list(row.values())[12]
                    matches = list(row.values())[14]

                if score_str:
                    try:
                        score = float(score_str)
                        if matches and matches.strip():
                            aligned_scores.append(score)
                        else:
                            absent_scores.append(score)
                    except ValueError:
                        pass
    
    threshold_aligned = min(aligned_scores) if aligned_scores else 0.75
    threshold_uncertain = max(absent_scores) if absent_scores else 0.60

    # Calcular umbral unico intermedio
    dynamic_threshold = (threshold_aligned + threshold_uncertain) / 2.0 if aligned_scores and absent_scores else 0.60

    print(f"\nUmbrales dinámicos calculados basados en {revision_path}:")
    print(f"  - Historias CON correspondencia  : {len(aligned_scores)} (min sim = {min(aligned_scores):.4f})" if aligned_scores else "  - Sin datos de correspondencia")
    print(f"  - Historias SIN correspondencia  : {len(absent_scores)} (max sim = {max(absent_scores):.4f})" if absent_scores else "  - Sin datos de alucinaciones")
    print(f"  - THRESHOLD_ALIGNED (piso)       : {threshold_aligned:.4f}")
    print(f"  - THRESHOLD_UNCERTAIN (techo)    : {threshold_uncertain:.4f}")
    print(f"  - UMBRAL DINAMICO CALCULADO      : {dynamic_threshold:.4f}")

    print("\nCargando historias...")
    generated = load_stories(generadas_path)
    expected  = load_stories(esperadas_path)

    config = PipelineConfig()
    config.model_name                      = MODEL_NAME
    config.device                          = DEVICE
    config.top_k                           = TOP_K
    config.hallucination_threshold_aligned  = threshold_aligned
    config.hallucination_threshold_uncertain = dynamic_threshold

    encoder = create_encoder(config.model_name, config.device)
    print("\nCalculando similitudes...")
    matching_data = evaluate_story_matching(generated, expected, encoder, config)

    print("Clasificando historias...")
    hallucination_data = detect_hallucinations(matching_data["results"], config)

    json_path = output_dir / "hallucination_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hallucination_data, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "hallucination_report.csv"
    csv_rows = []
    for r in hallucination_data["results"]:
        score = float(r['best_similarity'])
        is_hallucination = "Alucinación" if score < dynamic_threshold else "No Alucinación"
        csv_rows.append({
            "ID Hist Generada":        str(r["generated_index"] + 1),
            "Historia Generada":       r["generated_story"],
            "Clasificacion Alucinacion": is_hallucination,
            "Similitud Maxima":        f"{score:.4f}",
            "Umbral Dinamico":         f"{dynamic_threshold:.4f}",
            "Historia Esperada Mejor": r["best_expected_story"],
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(csv_rows)

    s = hallucination_data["summary"]
    print(f"\n{'='*50}")
    print(f"  DETECCIÓN DE ALUCINACIONES")
    print(f"{'='*50}")
    print(f"  Total historias          : {s['total_stories']}")
    print(f"  [+] Alineadas              : {s['aligned']}  ({s['alignment_rate']:.1%})")
    print(f"  [?] Inciertas              : {s['uncertain']}  ({s['uncertain_rate']:.1%})")
    print(f"  [-] Posible alucinación    : {s['possible_hallucination']}  ({s['hallucination_rate']:.1%})")
    print(f"\n  Archivos generados:")
    print(f"    [+] {csv_path}")

if __name__ == "__main__":
    main()
