"""
Dimensión 2 — Similitud de Solución (Story Matching)
=====================================================
Compara cada historia generada contra todas las esperadas usando
similitud coseno sobre embeddings SBERT y reporta el top-k más similares.

Reemplaza al script original 'generar_csv_similitud.py' con rutas
configurables para Linux.

Salidas:
  - matching_report.csv   → una fila por historia generada con top-k matches
  - matching_report.json  → versión completa del reporte
"""

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
# Editá estas rutas antes de correr el script

GENERADAS   = "datos/generadas.txt"
ESPERADAS   = "datos/esperadas.txt"
OUTPUT_DIR  = "resultados/similitud"

# Opciones del modelo (podés dejarlo como está)
MODEL_NAME  = "all-mpnet-base-v2"
DEVICE      = None      # None = auto (usa GPU si hay, sino CPU)
TOP_K       = 10        # cuántas historias esperadas reportar por generada
# ──────────────────────────────────────────────────────────────────────────────

import sys
import subprocess
from pathlib import Path

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


from pipeline2_refactored.io.loaders import load_stories
from pipeline2_refactored.embeddings.encoder import create_encoder
from pipeline2_refactored.matching.story_matching import evaluate_story_matching
from pipeline2_refactored.config.settings import PipelineConfig


def main():
    # ── Validar rutas ──────────────────────────────────────────────────────
    generadas_path = Path(GENERADAS)
    esperadas_path = Path(ESPERADAS)
    output_dir     = Path(OUTPUT_DIR)

    for label, p in [("GENERADAS", generadas_path), ("ESPERADAS", esperadas_path)]:
        if not p.exists():
            print(f"Error: no se encontró el archivo {label}: {p.resolve()}")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ───────────────────────────────────────────────────────
    print("\nCargando historias...")
    generated = load_stories(generadas_path)
    expected  = load_stories(esperadas_path)
    print(f"  Historias generadas : {len(generated)}")
    print(f"  Historias esperadas : {len(expected)}")

    # ── Preparar config ────────────────────────────────────────────────────
    config            = PipelineConfig()
    config.model_name = MODEL_NAME
    config.device     = DEVICE
    config.top_k      = TOP_K

    # ── Cargar modelo ──────────────────────────────────────────────────────
    encoder = create_encoder(config.model_name, config.device)

    # ── Calcular matching ──────────────────────────────────────────────────
    print("\nCalculando similitudes...")
    matching_data = evaluate_story_matching(generated, expected, encoder, config)

    # ── Guardar JSON ───────────────────────────────────────────────────────
    json_path = output_dir / "matching_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matching_data, f, indent=2, ensure_ascii=False)

    # ── Guardar CSV ────────────────────────────────────────────────────────
    # Formato compatible con el script original (usando ; como separador)
    csv_path = output_dir / "matching_report.csv"
    csv_rows = []
    for r in matching_data["results"]:
        row = {
            "ID Hist Generada":  str(r["generated_index"] + 1),
            "Historia Generada": r["generated_story"],
        }
        for m in r["top_matches"]:
            k = m["rank"]
            idx_esperada = m.get("expected_index", m.get("index", 0)) + 1
            row[f"Historia Esperada {k} (Similitud)"] = f"[{idx_esperada}] [{m['similarity']:.4f}] {m['text']}"
        row["Similitud Maxima"] = f"{r['best_similarity']:.4f}"
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
    s = matching_data["summary"]
    print(f"\n{'='*50}")
    print(f"  SIMILITUD DE SOLUCIÓN (STORY MATCHING)")
    print(f"{'='*50}")
    print(f"  Total historias generadas : {s['total_generated_stories']}")
    print(f"  Similitud promedio        : {s['average_best_similarity']:.4f}")
    print(f"\n  Archivos generados:")
    print(f"    [+] {csv_path}")
    print(f"    [+] {json_path}\n")


if __name__ == "__main__":
    main()
