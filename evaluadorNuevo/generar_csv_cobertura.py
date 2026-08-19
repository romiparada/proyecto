"""
Dimensión 1 — Cobertura Funcional
==================================
Compara cada funcionalidad del dominio contra las historias generadas
y determina si está cubierta, posiblemente cubierta o no cubierta.

Salidas:
  - coverage_report.csv   → una fila por funcionalidad con estado y top-k matches
  - coverage_report.json  → versión completa del reporte
"""

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
# Editá estas rutas antes de correr el script

GENERADAS   = "datos/generadas.txt"
ASPECTOS    = "datos/aspectosHospital.json"
OUTPUT_DIR  = "resultados/cobertura"

# Opciones del modelo (podés dejarlo como está)
MODEL_NAME  = "all-mpnet-base-v2"
DEVICE      = None          # None = auto (usa GPU si hay, sino CPU)
TOP_K       = 3             # cuántos matches reportar por funcionalidad

# Umbrales de clasificación
THRESHOLD_COVERED  = 0.75   # similitud mínima para "cubierta"
THRESHOLD_PARTIAL  = 0.60   # similitud mínima para "posiblemente cubierta"
# ──────────────────────────────────────────────────────────────────────────────

import sys
import subprocess
from pathlib import Path

# Auto-instalar dependencias si faltan
for pkg in ("sentence-transformers",):
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

import csv
import json
import requests
original_request = requests.Session.request
def patched_request(self, method, url, *args, **kwargs):
    if url.startswith('/api/'):
        url = 'https://huggingface.co' + url
    return original_request(self, method, url, *args, **kwargs)
requests.Session.request = patched_request


from pipeline2_refactored.io.loaders import load_stories, load_aspects
from pipeline2_refactored.embeddings.encoder import create_encoder
from pipeline2_refactored.coverage.functional_coverage import evaluate_functional_coverage
from pipeline2_refactored.config.settings import PipelineConfig


def main():
    # ── Validar rutas ──────────────────────────────────────────────────────
    generadas_path = Path(GENERADAS)
    aspectos_path  = Path(ASPECTOS)
    output_dir     = Path(OUTPUT_DIR)

    for label, p in [("GENERADAS", generadas_path), ("ASPECTOS", aspectos_path)]:
        if not p.exists():
            print(f"Error: no se encontró el archivo {label}: {p.resolve()}")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ───────────────────────────────────────────────────────
    print("\nCargando datos...")
    generated = load_stories(generadas_path)
    aspects   = load_aspects(aspectos_path)
    print(f"  Historias generadas : {len(generated)}")
    print(f"  Funcionalidades     : {len(aspects)}")

    # ── Preparar config ────────────────────────────────────────────────────
    config = PipelineConfig()
    config.model_name                  = MODEL_NAME
    config.device                      = DEVICE
    config.top_k                       = TOP_K
    config.coverage_threshold_covered  = THRESHOLD_COVERED
    config.coverage_threshold_partial  = THRESHOLD_PARTIAL

    # ── Cargar modelo ──────────────────────────────────────────────────────
    encoder = create_encoder(config.model_name, config.device)

    # ── Calcular cobertura ─────────────────────────────────────────────────
    print("\nCalculando cobertura funcional...")
    coverage_data = evaluate_functional_coverage(aspects, generated, encoder, config)

    # ── Guardar JSON ───────────────────────────────────────────────────────
    json_path = output_dir / "coverage_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coverage_data, f, indent=2, ensure_ascii=False)

    # ── Guardar CSV ────────────────────────────────────────────────────────
    csv_path = output_dir / "coverage_report.csv"
    csv_rows = []
    for r in coverage_data["results"]:
        row = {
            "indice_funcionalidad": r["functionality_index"] + 1,
            "categoria":            r["category"],
            "funcionalidad":        r["functionality"],
            "estado_cobertura":     r["coverage_status"],
            "mejor_historia_idx":   r["best_match"]["story_index"] + 1,
            "mejor_historia_texto": r["best_match"]["story_text"],
            "mejor_similitud":      f"{r['best_match']['similarity']:.4f}",
        }
        for m in r["top_matches"]:
            k = m["rank"]
            row[f"top{k}_idx"]        = m["index"] + 1
            row[f"top{k}_historia"]   = m["text"]
            row[f"top{k}_similitud"]  = f"{m['similarity']:.4f}"
        # Columnas vacías para revisión manual
        row["Clasificación"] = ""
        row["Correspondencias Reales (IDs)"] = ""
        row["Notas"] = ""
        csv_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── Resumen ────────────────────────────────────────────────────────────
    s = coverage_data["summary"]
    print(f"\n{'='*50}")
    print(f"  COBERTURA FUNCIONAL")
    print(f"{'='*50}")
    print(f"  Total funcionalidades    : {s['total_functionalities']}")
    print(f"  ✔ Cubiertas              : {s['covered']}  ({s['coverage_rate']:.1%})")
    print(f"  ⚠ Posiblemente cubiertas : {s['possibly_covered']}  ({s['partial_rate']:.1%})")
    print(f"  ✖ No cubiertas           : {s['not_covered']}  ({s['uncovered_rate']:.1%})")

    no_cubiertas = [r for r in coverage_data["results"] if r["coverage_status"] == "not_covered"]
    if no_cubiertas:
        print("\n  Funcionalidades no cubiertas:")
        for r in no_cubiertas:
            print(f"    • [{r['category']}] {r['functionality'][:65]}  (score={r['best_match']['similarity']:.4f})")

    print(f"\n  Archivos generados:")
    print(f"    [+] {csv_path}")
    print(f"    [+] {json_path}\n")


if __name__ == "__main__":
    main()
