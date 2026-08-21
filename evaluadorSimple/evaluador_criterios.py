
import json
from evaluador import evaluar_similares

with open("resultados/similitud_historias_criterios_all.json", "r", encoding="utf-8") as file:
    similitud_historias = json.load(file)

with open("resultados/evaluador_historias.json", "r", encoding="utf-8") as file:
    historias_evaluadas = json.load(file)

clasificacion = ["Alineada", "No Alineada"]
historias_similares_alineadas = []
for i in range(len(historias_evaluadas)):
    if historias_evaluadas[i]["eval"] == clasificacion[0]:
        similitud_historia_evaluada = similitud_historias[i]["h_sim"]
        for historia_evaluada_alineacion in historias_evaluadas[i]["alineacion"]:
            if historia_evaluada_alineacion["eval"] == clasificacion[0]:
                for similitud_historia in similitud_historia_evaluada:
                    if historia_evaluada_alineacion["id"] == similitud_historia["id"]:
                        historias_similares_alineadas.append(similitud_historia)

criterios_similares = []
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

with open("resultados/evaluador_historias_criterios.json", "w", encoding="utf-8") as file:
    json.dump(criterios_similares, file, indent=2, ensure_ascii=False)

res = []
evaluar_similares(res, criterios_similares, "ac_sim")

# res = []
# for historia_similar_alineada in historias_similares_alineadas:
#     criterios = historia_similar_alineada["ac"]
#     evaluar_similares(res, criterios, "ac_sim", 3)    
#     print("1- Siguiente historia")
#     print("2- Salir")
#     while True:
#         try:
#             opcion = int(input("Ingrese opcion: "))

#             if opcion in [1, 2]:
#                 salir = True
#                 break



#         except ValueError:
#             pass
#     if opcion != 1:
#         break    

with open(f"resultados/evaluador_criterios.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)