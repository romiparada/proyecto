
import json

with open("resultados/similitud_funcionalidades_all.json", "r", encoding="utf-8") as file:
    similitud_funcionalidad = json.load(file)

def borrar_lineas(n):
    for _ in range(n):
        print("\033[1A\033[2K", end="")

clasificacion = ["Alineada", "Posible", "Ausente"]

res = []
for funcionalidad in similitud_funcionalidad:
    print(funcionalidad["id"])
    print(funcionalidad["desc"])
    print("\n")

    similares = funcionalidad["h_sim"]
    res_f = {"id": funcionalidad["id"], "desc": funcionalidad["desc"], "alineacion": [], "eval": "Ausente"}
    top = 0
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
            if res_f["eval"] != clasificacion[0]:
                res_f["alineacion"] = []
            if (top <= 3):
                res_f["eval"] = clasificacion[0]
            else:
                res_f["eval"] = clasificacion[1]
            res_f["alineacion"].append({"sim": similar["sim"], "id": similar["id"]})
        else:     
            if res_f["eval"] != clasificacion[0]:
                res_f["eval"] = clasificacion[opcion-1]
                res_f["alineacion"].append({"sim": similar["sim"], "id": similar["id"]})
        
        if (top >= 3):
            if res_f["eval"] != clasificacion[2]:
                break

    res.append(res_f)
    print("Evaluacion: ", res_f["eval"])

    print("1- Siguiente")
    print("2- Salir")
    while True:
        try:
            opcion = int(input("Ingrese opcion: "))

            if opcion in [1, 2]:
                break

            borrar_lineas(1)


        except ValueError:
            pass
    if opcion == 1:
        borrar_lineas(8)
    else:
        break

with open(f"resultados/evaluador_funcionalidades.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)