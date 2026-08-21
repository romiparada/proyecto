from sentence_transformers import SentenceTransformer
import torch
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top", type=int)

args = parser.parse_args()

top = args.top
prefix = f"top_{top}.json" if top is not None else "all.json"

with open("esperados/solucion.json", "r", encoding="utf-8") as file:
    solucion_esperada = json.load(file)

with open("esperados/funcionalidades.json", "r", encoding="utf-8") as file:
    funcionalidades = json.load(file)

with open("generados/solucion.json", "r", encoding="utf-8") as file:
    solucion_generada = json.load(file)

historias_esperadas = [historia["desc"] for historia in solucion_esperada]
criterios_esperados = [historia["ac"] for historia in solucion_esperada]
historias_generadas = [historia["desc"] for historia in solucion_generada] 
criterios_generados = [historia["ac"] for historia in solucion_generada]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

embeddings_esperadas = model.encode(historias_esperadas)
embeddings_generadas = model.encode(historias_generadas)
embeddings_criterios_esperados = [model.encode(criterio_esperado) for criterio_esperado in criterios_esperados]
embeddings_criterios_generados = [model.encode(criterio_generado) for criterio_generado in criterios_generados]
embeddings_funcionalidades = model.encode(funcionalidades)

res = []

similarity_matrix = model.similarity(embeddings_generadas, embeddings_esperadas)

for i in range(len(embeddings_generadas)):
    res_h = {}
    res_h["id"] = f"HU{i + 1}"
    res_h["desc"] = historias_generadas[i]
    res_h["h_sim"] = []

    k = min(top, len(embeddings_esperadas)) if top is not None else len(embeddings_esperadas)
    scores, indices = torch.topk(similarity_matrix[i], k=k)

    for score, idx in zip(scores, indices):
        res_h_s = {}
        res_h_s["id"] = f"HUE{int(idx) + 1}"
        res_h_s["desc"] = historias_esperadas[idx]
        res_h_s["sim"] = float(score)
        res_h_s["ac"] = []

        if len(criterios_generados[i]) > 0 and len(criterios_esperados[idx]) > 0:
            similarity_matrix_criterios = model.similarity(embeddings_criterios_generados[i], embeddings_criterios_esperados[idx])
            
            for i_c in range(len(embeddings_criterios_generados[i])):
                res_c = {}
                res_c["id"] = f'AC{i + 1}.{i_c + 1}'
                res_c["desc"] = criterios_generados[i][i_c]
                res_c["ac_sim"] = []

                k = min(top, len(embeddings_criterios_esperados[idx])) if top is not None else len(embeddings_criterios_esperados[idx])
                scores_c, indices_c = torch.topk(similarity_matrix_criterios[i_c], k=k)

                for score_c, idx_c in zip(scores_c, indices_c):
                    res_c_s = {}
                    res_c_s["id"] = f'ACE{idx + 1}.{int(idx_c) + 1}'
                    res_c_s["desc"] = criterios_esperados[idx][idx_c]
                    res_c_s["sim"] = float(score_c)
                    res_c["ac_sim"].append(res_c_s) 
                res_h_s["ac"].append(res_c)
            
        res_h["h_sim"].append(res_h_s)
    res.append(res_h)

with open(f"resultados/similitud_historias_criterios_{prefix}", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)

res = [{"id": res_h["id"], "desc": res_h["desc"], "h_sim": [{"id": h_sim["id"], "desc": h_sim["desc"], "sim": h_sim["sim"]} for h_sim in res_h["h_sim"]]} for res_h in res]
with open(f"resultados/similitud_historias_{prefix}", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)


res = []

similarity_matrix_funcionalidades = model.similarity(embeddings_funcionalidades, embeddings_generadas)

for i_f in range(len(embeddings_funcionalidades)):
    res_f = {}
    res_f["id"] = f"F{i_f+1}"
    res_f["desc"] = funcionalidades[i_f]
    res_f["h_sim"] = []

    k = min(top, len(embeddings_generadas)) if top is not None else len(embeddings_generadas)
    scores, indices = torch.topk(similarity_matrix_funcionalidades[i_f], k=k)

    for score, idx in zip(scores, indices):
        res_f_s = {}
        res_f_s["id"] = f"HU{int(idx) + 1}"
        res_f_s["desc"] = historias_generadas[idx]
        res_f_s["ac"] = criterios_generados[idx]
        res_f_s["sim"] = float(score)
        res_f["h_sim"].append(res_f_s)
    
    res.append(res_f)

with open(f"resultados/similitud_funcionalidades_{prefix}", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)