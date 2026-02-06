import os
import sys
import argparse
from datetime import datetime
from io import StringIO
import json
import numpy as np
from evaluador_metricas import EvaluadorMetricas
from evaluador_criterios_aceptacion import EvaluadorCriteriosAceptacion

from evaluador_metricas import EvaluadorMetricas
from cargador_datos import (
    cargar_historias,
    cargar_aspectos,
    cargar_metadata,
    cargar_conjuntos_diversidad,
    descubrir_casos,
    guardar_json,
    guardar_txt
)

from evaluador_criterios_aceptacion import EvaluadorCriteriosAceptacion
from cargador_datos import cargar_criterios


def timestamp():
    return datetime.now().isoformat()


def ejecutar_caso(nombre_caso, dir_casos, archivo_aspectos, evaluador, dir_salida):
    dir_caso = os.path.join(dir_casos, nombre_caso)


    ruta_ca_gen = os.path.join(dir_caso, "ca_generados.json")
    ruta_ca_exp = os.path.join(dir_caso, "ca_esperados.json")

    criterios_generados = cargar_criterios(ruta_ca_gen)
    criterios_esperados = cargar_criterios(ruta_ca_exp)
    
    print(f"\ncaso: {nombre_caso}")
    
    metadata = cargar_metadata(dir_caso)
    print(f"Descripcion: {metadata['descripcion']}")
    print(f"Dominio: {metadata['dominio']}")
    
    ruta_generadas = os.path.join(dir_caso, "generadas.txt")
    historias_generadas = cargar_historias(ruta_generadas)
    print(f"Historias generadas: {len(historias_generadas)}")
    
    ruta_esperadas = os.path.join(dir_caso, "esperadas.txt")
    tiene_esperadas = os.path.exists(ruta_esperadas)
    
    if tiene_esperadas:
        historias_esperadas = cargar_historias(ruta_esperadas)
        print(f"Historias esperadas: {len(historias_esperadas)}")
    else:
        print("sin historias esperadas")
    
    aspectos = cargar_aspectos(archivo_aspectos)
    print(f"Aspectos: {len(aspectos)}")
    
    aspect_embeddings = {}
    for nombre, descripciones in aspectos.items():
        aspect_embeddings[nombre] = evaluador.model_sbert.encode(
            descripciones, convert_to_tensor=True
        )
    
    ts = timestamp()
    resultados = {
        "timestamp": ts,
        "caso": nombre_caso,
        "metadata": metadata,
        "config": evaluador.obtener_config(),
        "num_generadas": len(historias_generadas),
        "num_esperadas": len(historias_esperadas) if tiene_esperadas else 0,
        "resultados": {}
    }
    


    
    salida_txt = StringIO()
    salida_txt.write(f"Evaluacion experimental \n")
    salida_txt.write(f"Caso: {nombre_caso}\n")
    salida_txt.write(f"Descripcion: {metadata['descripcion']}\n")
    salida_txt.write(f"Timestamp: {ts}\n")
    salida_txt.write(f"Generadas: {len(historias_generadas)}\n")
    if tiene_esperadas:
        salida_txt.write(f"Esperadas: {len(historias_esperadas)}\n")
    
    resultados_alineacion = evaluador.evaluar_alineacion(
        historias_generadas,
        historias_esperadas
    )
    alineaciones = resultados_alineacion["por_historia"]

    scores = np.array([a["sbert"]["similitud"] for a in alineaciones])
    p50 = np.percentile(scores, 50)
    p75 = np.percentile(scores, 75)

    for a in alineaciones:
        if a["sbert"]["similitud"] >= p75:
            a["nivel"] = "alineacion_fuerte"
        elif a["sbert"]["similitud"] >= p50:
            a["nivel"] = "alineacion_media"
        else:
            a["nivel"] = "no_alineada"

    resultados["resultados"]["alineacion"] = alineaciones


    historias_validas = [
        a for a in alineaciones
        if a["nivel"] != "no_alineada"
    ]




    #solo se evaluan los criterios de aceptacion a las historias que tienen una alineacion media o fuerte correspondiente con una historia esparada, es decir se descartan los no alineados
    evaluador_ca = EvaluadorCriteriosAceptacion(evaluador.model_sbert)

    resultados_ca = []

    for a in historias_validas:
        idx_gen = a["indice"]
        idx_ref = a["sbert"]["indice_match"]

        ca_gen = criterios_generados[idx_gen]
        ca_ref = criterios_esperados[idx_ref]

        eval_ca = evaluador_ca.evaluar(ca_gen, ca_ref)

        resultados_ca.append({
            "historia_generada_idx": idx_gen,
            "historia_referencia_idx": idx_ref,
            "evaluacion": eval_ca
        })

    resultados["resultados"]["criterios_aceptacion"] = resultados_ca

    
    ts_safe = ts.replace(":", "-").replace(".", "-")
    ruta_json = os.path.join(dir_salida, f"{nombre_caso}_{ts_safe}.json")
    ruta_txt = os.path.join(dir_salida, f"{nombre_caso}_{ts_safe}.txt")
    
    guardar_json(resultados, ruta_json)
    guardar_txt(salida_txt.getvalue(), ruta_txt)
    
    print(f"Guardado: {ruta_json}")
    print(f"Guardado: {ruta_txt}")
    
    return resultados


def main():
    parser = argparse.ArgumentParser(description="Evaluacion experimental de historias de usuario")
    
    parser.add_argument("--caso", required=True)
    parser.add_argument("--salida", default="resultados")
    parser.add_argument("--casos-dir", default="casos_prueba")
    parser.add_argument("--aspectos", default="casos_prueba/aspectos_hospital.json")
    parser.add_argument("--diversidad-json", default=None)
    
    args = parser.parse_args()

    
    os.makedirs(args.salida, exist_ok=True)
    
    evaluador = EvaluadorMetricas()
    
    if args.caso == "todos":
        casos = descubrir_casos(args.casos_dir)
        print(f"numero de casos a probar: {len(casos)}")
    else:
        casos = [args.caso]
    
    resultados_todos = []
    for nombre_caso in casos:
        resultado = ejecutar_caso(nombre_caso, args.casos_dir, args.aspectos, evaluador, args.salida)
        if resultado is not None:
            resultados_todos.append(resultado)



    if args.diversidad_json:

        print("\nevaluando diversidad de las generaciones de HU entre las distintas herramientas")

        nombres_modelos, conjuntos = cargar_conjuntos_diversidad(args.diversidad_json)

        resultado_div = evaluador.evaluar_diversidad(conjuntos)

        print(f"Diversidad media: {resultado_div['diversidad_media']:.2f}%")
        print(f"Diversidad std: {resultado_div['diversidad_std']:.2f}%")

        resultado_div["modelos"] = nombres_modelos

        ruta_div_json = os.path.join(args.salida, "diversidad_modelos.json")
        guardar_json(resultado_div, ruta_div_json)

        print(f"Guardado diversidad en: {ruta_div_json}")


    print(f"Ejecutados: {len(resultados_todos)}/{len(casos)}")
    print(f"Resultados en: {args.salida}/")
    print(f"**********************\n")


if __name__ == "__main__":
    main()
