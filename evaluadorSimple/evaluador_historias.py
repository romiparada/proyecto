
import json
from evaluador.py import evaluador

with open("resultados/similitud_historias_all.json", "r", encoding="utf-8") as file:
    similitud_historias = json.load(file)

res = []
evaluador(res, similitud_historias, "h_sim")

with open(f"resultados/evaluador_historias.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)