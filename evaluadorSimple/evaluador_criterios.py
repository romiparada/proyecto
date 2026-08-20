
import json

with open("resultados/similitud_historias_criterios_all.json", "r", encoding="utf-8") as file:
    similitud_historias = json.load(file)

with open("resultados/evaluador_historias.json", "r", encoding="utf-8") as file:
    historias_evaluadas = json.load(file)

clasificacion = ["Alineada", "Posible", "Ausente"]
historias_similares_alineadas = []
for i in range(len(historias_evaluadas)):
    if historias_evaluadas[i]["eval"] == clasificacion[0]:
        similitud_historia_evaluada = similitud_historias[i]["h_sim"]
        for historia_evaluada_alineacion in historias_evaluadas[i]["alineacion"]:
            if historia_evaluada_alineacion["eval"] == clasificacion[0]:
                for similitud_historia in similitud_historia_evaluada:
                    if historia_evaluada_alineacion["id"] == similitud_historia["id"]:
                        historias_similares_alineadas.append(similitud_historia)

res = []
for historia_similar_alineada in historias_similares_alineadas:
    criterios = historia_similar_alineada["ac"]
    for criterio in criterios:
        res_i = {"id": criterio["id"], "desc": criterio["desc"], "alineacion": [], "eval": "Ausente"}
        top = 0
        similares = criterio["ac_sim"]
        for similar in similares:

            print("\n")
            print(criterio["id"])
            print(criterio["desc"])
            print("\n")

            top = top + 1
            print(similar["id"])
            print(similar["desc"])

            print("\nClasificación")
            print("1 - Alineada")
            print("2 - Posible")
            print("3 - Ausente")
            print("4 - Salir")

            while True:
                try:
                    opcion = int(input("Ingrese la clasificación: "))

                    if opcion in [1, 2, 3, 4]:
                        break
                    borrar_lineas(1)


                except ValueError:
                    pass


            if opcion == 4:
                break

            res_alineacion = {"id": similar["id"], "desc": similar["desc"], "sim": similar["sim"], "eval": clasificacion[opcion-1]}
            res_i["alineacion"].append(res_alineacion)

            if opcion == 1:              
                if (top <= 3):
                    res_i["eval"] = clasificacion[0]
                else:
                    res_i["eval"] = clasificacion[1]
            else:     
                if opcion == 2:
                    if res_i["eval"] == clasificacion[2]:
                        res_i["eval"] = clasificacion[1]
            
            if (top >= 3):
                if res_i["eval"] != clasificacion[2]:
                    break

        res.append(res_i)
        print("Evaluacion: ", res_i["eval"])

        print("1- Siguiente")
        print("2- Salir")
        while True:
            try:
                opcion = int(input("Ingrese opcion: "))

                if opcion in [1, 2]:
                    salir = True
                    break



            except ValueError:
                pass
        if opcion != 1:
            break
    print("1- Siguiente historia")
    print("2- Salir")
    while True:
        try:
            opcion = int(input("Ingrese opcion: "))

            if opcion in [1, 2]:
                salir = True
                break



        except ValueError:
            pass
    if opcion != 1:
        break
    

with open(f"resultados/evaluador_criterios.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)