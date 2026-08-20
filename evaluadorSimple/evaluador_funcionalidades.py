
import json

with open("resultados/similitud_funcionalidades_all.json", "r", encoding="utf-8") as file:
    similitud_funcionalidad = json.load(file)


clasificacion = ["Alineada", "Posible", "Ausente"]

res = []
for funcionalidad in similitud_funcionalidad:
    similares = funcionalidad["h_sim"]
    res_i = {"id": funcionalidad["id"], "desc": funcionalidad["desc"], "alineacion": [], "eval": "Ausente"}
    top = 0
    for similar in similares:
        top = top + 1

        print("\n")
        print(funcionalidad["id"])
        print(funcionalidad["desc"])
        print("\n")

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
                break



        except ValueError:
            pass
    if opcion != 1:
        break

with open(f"resultados/evaluador_funcionalidades.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)