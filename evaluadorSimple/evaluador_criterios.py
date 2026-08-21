
import json
from evaluador import evaluar_similares

with open("resultados/similitud_historias_alineadas_criterios_all.json", "r", encoding="utf-8") as file:
    criterios_similares = json.load(file)

res = []
evaluar_similares(res, criterios_similares, "ac_sim") 

with open(f"resultados/evaluador_criterios.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)