
import json

clasificacion = ["Alineada", "No Alineada"]

def porcentajes_evaluador(tipo):
    try:
        with open(f"resultados/evaluador_{tipo}.json", "r", encoding="utf-8") as file:
            evaluadas = json.load(file)
    except:
        return
        
    if not evaluadas:
        return

    res = {}
    res["total_evaluadas"] = len(evaluadas) 
    for c in clasificacion:
        res[c] = {}
        res[c]["total"] = 0

    for evaluada in evaluadas:
        res[evaluada["eval"]]["total"] = res[evaluada["eval"]]["total"] + 1

    res[clasificacion[0]]["porcentaje"] = res[clasificacion[0]]["total"] / res["total_evaluadas"] * 100
    res[clasificacion[1]]["porcentaje"] = 100 - res[clasificacion[0]]["porcentaje"]
    
    with open(f"resultados/porcentaje_{tipo}.json", "w", encoding="utf-8") as file:
        json.dump(res, file, indent=2, ensure_ascii=False)

for tipo in ["historias", "funcionalidaes", "criterios"]:
    porcentajes_evaluador(tipo)