import os
import sys
import argparse
from datetime import datetime
from io import StringIO
import json
import numpy as np

from evaluador_metricas import EvaluadorMetricas
from cargador_datos import (
    cargar_historias,
    cargar_aspectos,
    cargar_metadata,
    descubrir_casos,
    guardar_json,
    guardar_txt
)

from evaluador_criterios_aceptacion import EvaluadorCriteriosAceptacion
from cargador_datos import cargar_criterios


def timestamp():
    return datetime.now().isoformat()


def alinear_historias(resultados, evaluador):
    output = StringIO()
    
    output.write(f"\nUmbral SBERT: {evaluador.sbert_umbral:.2f}\n\n")
    
    for res in resultados["por_historia"]:
        i = res["indice"]
        gen = res["generada"]
        
        output.write(f"[{i}] {gen}\n\n")
        
        sbert = res["sbert"]
        output.write(f"  SBERT [{sbert['indice_match']}]: {sbert['texto_match']}\n")
        output.write(f"  Similitud: {sbert['similitud']:.4f} ({sbert['nivel']})\n\n")
        
        bert = res["bertscore"]
        output.write(f"  BERTScore [{bert['indice_match']}]: {bert['texto_match']}\n")
        output.write(f"  F1: {bert['f1']:.4f}\n\n")
        
        bleu = res["bleu"]
        #output.write(f"  BLEU [{bleu['indice_match']}]: {bleu['texto_match']}\n")
        #output.write(f"  Score: {bleu['score']:.4f}\n\n")
        
        #rouge = res["rouge_l"]
        #output.write(f"  ROUGE-L [{rouge['indice_match']}]: {rouge['texto_match']}\n")
        #output.write(f"  F1: {rouge['score']:.4f}\n\n")
        
        output.write("---------------------------------")
    
    agg = resultados["agregado"]
    output.write(f"SBERT media: {agg['sbert']['media']:.4f} (±{agg['sbert']['std']:.4f})\n")
    output.write(f"Alineacion fuerte: {agg['sbert']['alineacion_fuerte_pct']:.1f}%\n\n")
    output.write(f"BERTScore media: {agg['bertscore']['media']:.4f} (±{agg['bertscore']['std']:.4f})\n")
    #output.write(f"BLEU media: {agg['bleu']['media']:.4f} (±{agg['bleu']['std']:.4f})\n")
    #output.write(f"ROUGE-L media: {agg['rouge_l']['media']:.4f} (±{agg['rouge_l']['std']:.4f})\n\n")
    
    return output.getvalue()


def calcular_completitud(score, detalle):
    output = StringIO()
    output.write("\nEvaluacion completitud\n")
    output.write(f"Score: {score:.2f}\n")
    for aspecto, cubierto in detalle.items():
        estado = "CUBIERTA" if cubierto else "FALTA"
        output.write(f"  {aspecto}: {estado}\n")
    output.write("\n")
    return output.getvalue()



#se puede setear un umbral de consistencia, por predeterminado esta en CONSISTENCIA_UMBRAL = 0.85,
#es decir, si la similitud semantica de dos historias es mayor que este valor, se consideran redundantes (solapadas)
def calcular_consistencia(score, solapamientos):
    output = StringIO()
    output.write("\nEvaluacion consistencia\n\n")
    output.write(f"Score: {score:.2f}\n")
    if solapamientos:
        #esto es si el par indica alta similitud semántica, 
        #   ejemplo 
        #       texto1: El sistema debe permitir registrar pacientes
        #       texto2:El sistema debe permitir dar de alta pacientes
        #
        output.write("historias (pares) redundantes:\n")
        for i, j, sim in solapamientos:
            output.write(f"  [{i}] - [{j}] ({sim:.2f})\n")
    else:
        output.write("Sin redundantes\n")
    output.write("\n")
    return output.getvalue()


def ejecutar_caso(nombre_caso, dir_casos, archivo_aspectos, evaluador, dir_salida):
    dir_caso = os.path.join(dir_casos, nombre_caso)


    ruta_ca_gen = os.path.join(dir_caso, "ca_generados.json")
    ruta_ca_exp = os.path.join(dir_caso, "ca_esperados.json")

    criterios_generados = cargar_criterios(ruta_ca_gen)
    criterios_esperados = cargar_criterios(ruta_ca_exp)
    
    print(f"\ncaso: {nombre_caso}")
    print("--------------------------")
    
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
    salida_txt.write(f"*********************************")
    salida_txt.write(f"Caso: {nombre_caso}\n")
    salida_txt.write(f"Descripcion: {metadata['descripcion']}\n")
    salida_txt.write(f"Timestamp: {ts}\n")
    salida_txt.write(f"Generadas: {len(historias_generadas)}\n")
    if tiene_esperadas:
        salida_txt.write(f"Esperadas: {len(historias_esperadas)}\n")
    
    alineaciones = evaluador.evaluar_alineacion_sbert(
        historias_generadas,
        historias_esperadas
    )

    scores = np.array([a["sbert_sim"] for a in alineaciones])
    p50 = np.percentile(scores, 50)
    p75 = np.percentile(scores, 75)

    for a in alineaciones:
        if a["sbert_sim"] >= p75:
            a["nivel"] = "alineacion_fuerte"
        elif a["sbert_sim"] >= p50:
            a["nivel"] = "alineacion_media"
        else:
            a["nivel"] = "no_alineada"

    resultados["resultados"]["alineacion"] = alineaciones


    historias_validas = [
        a for a in alineaciones
        if a["nivel"] != "no_alineada"
    ]


    evaluador_ca = EvaluadorCriteriosAceptacion(evaluador.model_sbert)

    resultados_ca = []

    for a in historias_validas:   # ← SOLO alineadas
        idx_gen = a["indice"]
        idx_ref = a["match_idx"]

        ca_gen = criterios_generados[idx_gen]
        ca_ref = criterios_esperados[idx_ref]

        eval_ca = evaluador_ca.evaluar(ca_gen, ca_ref)

        resultados_ca.append({
            "historia_generada_idx": idx_gen,
            "historia_referencia_idx": idx_ref,
            "evaluacion": eval_ca
        })

    resultados["resultados"]["criterios_aceptacion"] = resultados_ca
    
    if metadata.get("evaluar_completitud", True):
        print("eval completitud")

        historiasCompletitud = [a["generada"] for a in historias_validas]
        comp_score, comp_detalle = evaluador.calcular_completitud(historiasCompletitud, aspect_embeddings)
        resultados["resultados"]["completitud"] = {
            "score": comp_score,
            "cobertura": comp_detalle
        }
        salida_txt.write(calcular_completitud(comp_score, comp_detalle))
    
    if metadata.get("evaluar_consistencia", True):
        print("eval consistencia")

        historiasConsistencia = [a["generada"] for a in historias_validas]
        cons_score, solapamientos = evaluador.calcular_consistencia(historiasConsistencia)
        resultados["resultados"]["consistencia"] = {
            "score": cons_score,
            "pares_solapados": solapamientos
        }
        salida_txt.write(calcular_consistencia(cons_score, solapamientos))
    
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

    print(f"Ejecutados: {len(resultados_todos)}/{len(casos)}")
    print(f"Resultados en: {args.salida}/")
    print(f"**********************\n")


if __name__ == "__main__":
    main()
