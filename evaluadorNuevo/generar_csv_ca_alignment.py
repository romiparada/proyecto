"""
Dimensión 2b — Alineación de Criterios de Aceptación (CA)
===========================================================
Para cada historia generada, lee sus correspondencias manuales
(desde el archivo CSV revisado) y extrae todos los CA esperados
de esas historias correspondientes. Compara los CA generados
contra ese subconjunto de CA esperados y genera el CSV final.
"""

import sys
import subprocess
from pathlib import Path
import csv
import json

for pkg in ("sentence-transformers", "torch"):
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Instalando {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

from sentence_transformers import util


from pipeline2_refactored.io.loaders import load_stories, load_criteria, align_criteria_to_stories
from pipeline2_refactored.embeddings.encoder import create_encoder
from pipeline2_refactored.config.settings import PipelineConfig

GENERADAS   = "datos/generadas.txt"
ESPERADAS   = "datos/esperadas.txt"
CA_GEN      = "datos/ca_generados.json"
CA_ESP      = "datos/ca_esperados.json"
REVISION_ALINEACION = "revision_manual_alineacion.csv"
OUTPUT_DIR  = "resultados/ca_alignment"

MODEL_NAME  = "all-mpnet-base-v2"
DEVICE      = None
TOP_K       = 3

def main():
    generadas_path = Path(GENERADAS)
    esperadas_path = Path(ESPERADAS)
    ca_gen_path    = Path(CA_GEN)
    ca_esp_path    = Path(CA_ESP)
    revision_path  = Path(REVISION_ALINEACION)
    output_dir     = Path(OUTPUT_DIR)

    rutas = [
        ("GENERADAS", generadas_path),
        ("ESPERADAS", esperadas_path),
        ("CA_GEN",    ca_gen_path),
        ("CA_ESP",    ca_esp_path),
        ("REVISION_ALINEACION", revision_path)
    ]
    for label, p in rutas:
        if not p.exists():
            print(f"Error: no se encontró el archivo {label}: {p.resolve()}")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\\nCargando datos...")
    generated    = load_stories(generadas_path)
    expected     = load_stories(esperadas_path)
    ca_generated = load_criteria(ca_gen_path)
    ca_expected  = load_criteria(ca_esp_path)

    ca_generated = align_criteria_to_stories(ca_generated, generated, "CA generated")
    ca_expected  = align_criteria_to_stories(ca_expected,  expected,  "CA expected")

    # Leer correspondencias
    correspondencias = {}
    with open(revision_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        # Manejar posible BOM en la primera columna
        reader.fieldnames = [name.lstrip('\ufeff') for name in reader.fieldnames] if reader.fieldnames else []
        for idx, row in enumerate(reader):
            # Parseamos de 1-indexed (e.g. 1, 2, 3)
            gen_idx_str = row.get("ID Hist Generada")
            if not gen_idx_str:
                gen_idx_str = str(idx + 1)
            gen_idx = int(gen_idx_str)
            # Parsing nuevo ID guardado (comma separated)
            sel_str = row.get("Correspondencias Reales (IDs)", "").strip()
            sel = []
            if sel_str:
                try:
                    sel = [x.strip() for x in sel_str.split(',') if x.strip()]
                except:
                    pass
            
            correspondencias[str(gen_idx)] = sel

    config = PipelineConfig()
    config.model_name = MODEL_NAME
    config.device = DEVICE
    
    encoder = create_encoder(config.model_name, config.device)
    
    matrix_rows = []
    
    print("Calculando alineación de criterios de aceptación...")
    for gen_idx, story in enumerate(generated):
        gen_cas = ca_generated[gen_idx] if gen_idx < len(ca_generated) else []
        if not gen_cas: continue
        
        # matched_exp_idxs should look up using the 1-indexed generated ID which is saved in the CSV
        matched_exp_idxs = correspondencias.get(str(gen_idx + 1), [])
        if not matched_exp_idxs: 
            for gca in gen_cas:
                matrix_rows.append({
                    "ID_Hist_Generada": gen_idx + 1,
                    "CA_Generado": gca,
                    "CA_Esp_1": "", "Sim_1": "", "Hist_Esp_1": "",
                    "CA_Esp_2": "", "Sim_2": "", "Hist_Esp_2": "",
                    "CA_Esp_3": "", "Sim_3": "", "Hist_Esp_3": "",
                    "Correspondencias Reales (IDs)": "", "Notas": ""
                })
            continue

        exp_candidates = []
        for eidx in matched_exp_idxs:
            try:
                # El ID extraído ahora representa la posición 1-indexed
                idx_int = int(eidx) - 1
                if 0 <= idx_int < len(ca_expected):
                    for crit in ca_expected[idx_int]:
                        exp_candidates.append({
                            "text": crit,
                            "source_story": eidx
                        })
            except ValueError:
                pass

        
        if not exp_candidates:
            for gca in gen_cas:
                matrix_rows.append({
                    "ID_Hist_Generada": gen_idx + 1,
                    "CA_Generado": gca,
                    "CA_Esp_1": "", "Sim_1": "", "Hist_Esp_1": "",
                    "CA_Esp_2": "", "Sim_2": "", "Hist_Esp_2": "",
                    "CA_Esp_3": "", "Sim_3": "", "Hist_Esp_3": "",
                    "Correspondencias Reales (IDs)": "", "Notas": ""
                })
            continue

        ca_emb_gen = encoder.encode(gen_cas)
        ca_emb_exp = encoder.encode([c["text"] for c in exp_candidates])
        cosine_scores = util.cos_sim(ca_emb_gen, ca_emb_exp).tolist()
        
        for i, gca in enumerate(gen_cas):
            scores = cosine_scores[i]
            scored_candidates = []
            for j, score in enumerate(scores):
                scored_candidates.append({
                    "score": score,
                    "text": exp_candidates[j]["text"],
                    "source": exp_candidates[j]["source_story"]
                })
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            
            top3 = scored_candidates[:3]
            
            row = {
                "ID_Hist_Generada": gen_idx + 1,
                "CA_Generado": gca
            }
            for rank in range(3):
                col_c = f"CA_Esp_{rank+1}"
                col_s = f"Sim_{rank+1}"
                col_h = f"Hist_Esp_{rank+1}"
                if rank < len(top3):
                    row[col_c] = top3[rank]["text"]
                    row[col_s] = f"{top3[rank]['score']:.4f}"
                    row[col_h] = top3[rank]["source"]
                else:
                    row[col_c] = ""
                    row[col_s] = ""
                    row[col_h] = ""
            
            # Columnas vacías para revisión manual
            row["Correspondencias Reales (IDs)"] = ""
            row["Notas"] = ""
            
            matrix_rows.append(row)

    matrix_csv_path = output_dir / "ca_alignment_report.csv"
    if matrix_rows:
        with open(matrix_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(matrix_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(matrix_rows)
        print(f"\\n  Archivo generado: [+] {matrix_csv_path}")

if __name__ == "__main__":
    main()
