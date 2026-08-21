import json
from evaluador import evaluar_similares

with open("resultados/similitud_historias_criterios_all.json", "r", encoding="utf-8") as file:
    similitud_historias = json.load(file)

with open("resultados/evaluador_historias.json", "r", encoding="utf-8") as file:
    historias_evaluadas = json.load(file)

def similitud_criterios(criterios_similares, historias_similares_alineadas):
    for i in range(len(historias_similares_alineadas)):
        for criterio_i in historias_similares_alineadas[i]["ac"]:
            encontrado = False
            for c_sim in criterios_similares:
                if c_sim["id"] == criterio_i["id"]:
                    encontrado = True
                    c_sim["ac_sim"] = sorted(
                        c_sim["ac_sim"] + criterio_i["ac_sim"],
                        key=lambda x: x["sim"],
                        reverse=True
                    )
            if not encontrado:
                criterios_similares.append(criterio_i)

clasificacion = ["Alineada", "No Alineada"]

historias_similares_alineadas_simple = []
for i in range(len(historias_evaluadas)):
    if historias_evaluadas[i]["eval"] == clasificacion[0]:
        similitud_historia_evaluada = similitud_historias[i]["h_sim"]
        for historia_evaluada_alineacion in historias_evaluadas[i]["alineacion"]:
            if historia_evaluada_alineacion["eval"] == clasificacion[0]:
                for similitud_historia in similitud_historia_evaluada:
                    if historia_evaluada_alineacion["id"] == similitud_historia["id"]:
                        historias_similares_alineadas_simple.append(similitud_historia)

criterios_similares_simple = []
similitud_criterios(criterios_similares_simple, historias_similares_alineadas_simple)

with open("resultados/similitud_historias_alineadas_criterios_simple_all.json", "w", encoding="utf-8") as file:
    json.dump(criterios_similares_simple, file, indent=2, ensure_ascii=False)

historias_similares_alineadas = []
for i in range(len(historias_evaluadas)):
    if historias_evaluadas[i]["eval"] == clasificacion[0]:
        similitud_historia_evaluada = similitud_historias[i]["h_sim"]
        for historia_evaluada_alineacion in historias_evaluadas[i]["alineacion"]:
            if historia_evaluada_alineacion["eval"] == clasificacion[0]:
                for similitud_historia in similitud_historia_evaluada:
                    if historia_evaluada_alineacion["id"] == similitud_historia["id"]:
                        for j in range(len(similitud_historia["ac"])):
                            criterio_generado = similitud_historia["ac"][j]
                            for k in range(len(criterio_generado["ac_sim"])):
                                criterio_esperado = criterio_generado["ac_sim"][k]
                                criterio_generado["ac_sim"][k] = {**criterio_esperado, "h_id": historia_evaluada_alineacion["id"],"h_desc": historia_evaluada_alineacion["desc"]}
                            similitud_historia["ac"][j] = {"h_id": historias_evaluadas[i]["id"],"h_desc": historias_evaluadas[i]["desc"], **criterio_generado}
                        historias_similares_alineadas.append(similitud_historia)

criterios_similares = []
similitud_criterios(criterios_similares, historias_similares_alineadas)

with open("resultados/similitud_historias_alineadas_criterios_all.json", "w", encoding="utf-8") as file:
    json.dump(criterios_similares, file, indent=2, ensure_ascii=False)