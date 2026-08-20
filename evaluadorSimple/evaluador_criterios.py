
import json

with open("resultados/similitud_historias_criterios_all.json", "r", encoding="utf-8") as file:
    similitud_historias = json.load(file)

with open("resultados/evaluador_historias.json", "r", encoding="utf-8") as file:
    historias_evaluadas = json.load(file)

def borrar_lineas(n):
    for _ in range(n):
        print("\033[1A\033[2K", end="")

clasificacion = ["Alineada", "Posible", "Ausente"]
historias_similares_alineadas = []
for i in range(len(historias_evaluadas)):
    if historias_evaluadas[i]["eval"] != clasificacion[2]:
        similitud_historia_evaluada = similitud_historias[i]["h_sim"]
        for historia_evaluada_alineacion in historias_evaluadas[i]["alineacion"]:
            for similitud_historia in similitud_historia_evaluada:
                if historia_evaluada_alineacion["id"] == similitud_historia["id"]:
                    historias_similares_alineadas.append(similitud_historia)

res = []
for historia_similar_alineada in historias_similares_alineadas:
    criterios = historia_similar_alineada["ac"]
    for criterio in criterios:
        print(criterio["id"])
        print(criterio["desc"])
        print("\n")
        res_c = {"id": criterio["id"], "desc": criterio["desc"], "alineacion": [], "eval": "Ausente"}
        top = 0
        similares = criterio["ac_sim"]
        for similar in similares:
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

            borrar_lineas(9)

            if opcion == 4:
                break

            if opcion == 1:              
                if res_c["eval"] != clasificacion[0]:
                    res_c["alineacion"] = []
                if (top <= 3):
                    res_c["eval"] = clasificacion[0]
                else:
                    res_c["eval"] = clasificacion[1]
                res_c["alineacion"].append({"sim": similar["sim"], "id": similar["id"]})
            else:     
                if res_c["eval"] != clasificacion[0]:
                    res_c["eval"] = clasificacion[opcion-1]
                    res_c["alineacion"].append({"sim": similar["sim"], "id": similar["id"]})
            
            if (top >= 3):
                if res_c["eval"] != clasificacion[2]:
                    break

        res.append(res_c)
        print("Evaluacion: ", res_c["eval"])

        print("1- Siguiente")
        print("2- Salir")
        while True:
            try:
                opcion = int(input("Ingrese opcion: "))

                if opcion in [1, 2]:
                    salir = True
                    break

                borrar_lineas(1)


            except ValueError:
                pass
        if opcion == 1:
            borrar_lineas(8)
        else:
            break
    print("1- Siguiente historia")
    print("2- Salir")
    while True:
        try:
            opcion = int(input("Ingrese opcion: "))

            if opcion in [1, 2]:
                salir = True
                break

            borrar_lineas(1)


        except ValueError:
            pass
    if opcion == 1:
        borrar_lineas(8)
    else:
            break
    

with open(f"resultados/evaluador_criterios.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)