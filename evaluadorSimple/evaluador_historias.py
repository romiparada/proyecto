
import json

with open("resultados/similitud_historias_all.json", "r", encoding="utf-8") as file:
    similitud_historias = json.load(file)

def borrar_lineas(n):
    for _ in range(n):
        print("\033[1A\033[2K", end="")

clasificacion = ["Alineada", "Posible", "Ausente"]

for historia in similitud_historias:
    print(historia["id"])
    print(historia["desc"])
    print("\n")

    similares = historia["h_sim"]
    res = []
    res_h = {"id": historia["id"], "desc": historia["desc"], "alineacion": [], "eval": "Ausente"}
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
            if res_h["eval"] != clasificacion[0]:
                res_h["alineacion"] = []
            if (top <= 3):
                res_h["eval"] = clasificacion[0]
            else:
                res_h["eval"] = clasificacion[1]
            res_h["alineacion"].append({"sim": similar["sim"], "id": similar["id"]})
        else:     
            if res_h["eval"] != clasificacion[0]:
                res_h["eval"] = clasificacion[opcion-1]
                res_h["alineacion"].append({"sim": similar["sim"], "id": similar["id"]})
        
        if (top >= 3):
            if res_h["eval"] != clasificacion[2]:
                break

    res.append(res_h)
    print("Evaluacion: ", res_h["eval"])

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

with open(f"resultados/evaluador_historias.json", "w", encoding="utf-8") as file:
    json.dump(res, file, indent=2, ensure_ascii=False)