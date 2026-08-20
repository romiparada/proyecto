def evaluar_similares(res, elementos, sim_key):
    clasificacion = ["Alineada", "No Alineada"]
    current_top = 5    
    for elemento in elementos:
        res_i = {"id": elemento["id"], "desc": elemento["desc"], "alineacion": [], "eval": clasificacion[1]}
        top = 0
        similares = elemento[sim_key]
        for similar in similares:

            print("\n")
            print(elemento["id"])
            print(elemento["desc"])
            print("\n")

            top = top + 1
            print(similar["id"])
            print(similar["desc"])

            print("\nClasificación")
            print("1 - Alineada")
            print("2 - No Alineada")
            print("3 - Salir")

            while True:
                try:
                    opcion = int(input("Ingrese la clasificación: "))

                    if opcion in [1, 2, 3]:
                        break


                except ValueError:
                    pass


            if opcion == 3:
                break

            res_alineacion = {"id": similar["id"], "desc": similar["desc"], "sim": similar["sim"], "eval": clasificacion[opcion-1]}
            res_i["alineacion"].append(res_alineacion)

            if opcion == 1:
                res_i["eval"] = clasificacion[0]

            if (top >= current_top):
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