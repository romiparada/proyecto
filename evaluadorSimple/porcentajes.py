
import json

with open("resultados/evaluador_historias.json", "r", encoding="utf-8") as file:
    evaluador_historias = json.load(file)

with open("resultados/evaluador_criterios.json", "r", encoding="utf-8") as file:
    evaluador_criterios = json.load(file)

with open("resultados/evaluador_funcionalidades.json", "r", encoding="utf-8") as file:
    evaluador_funcionalidades = json.load(file)

clasificacion = ["Alineada", "No Alineada"]

def porcentajes_evaluador(evaluadas):
    res = {}
    res["total_evaluadas"] = len(evaluadas) 
    for c in clasificacion:
        res[c] = {}
        res[c]["total"] = 0

    for evaluada in evaluadas:
        res[evaluada["eval"]]["total"] = res[evaluada["eval"]]["total"] + 1

    res[clasificacion[0]]["porcentaje"] = res[clasificacion[0]]["total"] / res["total_evaluadas"] * 100
    res[clasificacion[1]]["porcentaje"] = 100 - res[clasificacion[0]]["porcentaje"]
    return res

res = porcentajes_evaluador(evaluador_historias) 
with open(f"resultados/porcentaje_historias.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)

res = porcentajes_evaluador(evaluador_criterios) 
with open(f"resultados/porcentaje_criterios.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)

res = porcentajes_evaluador(evaluador_funcionalidades) 
with open(f"resultados/porcentaje_funcionalidades.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)