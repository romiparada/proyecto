
import json
from evaluador import evaluar_smilares

with open("resultados/similitud_funcionalidades_all.json", "r", encoding="utf-8") as file:
    similitud_funcionalidad = json.load(file)

res = []
evaluador(res, similitud_funcionalidad, "h_sim")

with open(f"resultados/evaluador_funcionalidades.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)