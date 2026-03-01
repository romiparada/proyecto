"""
Ejecutor de casos usando el pipeline refactorizado.

Uso:
    python ejecutar_pipeline.py --caso caso_A_alineadas
    python ejecutar_pipeline.py --caso todos
    python ejecutar_pipeline.py --caso caso_A_alineadas --diversidad-json modelos.json
"""

import os
import argparse
from datetime import datetime

from pipeline import EvaluationPipeline, EvaluationInput
from cargador_datos import (
    cargar_historias,
    cargar_aspectos,
    cargar_metadata,
    cargar_conjuntos_diversidad,
    cargar_criterios,
    descubrir_casos,
    guardar_json
)


def timestamp():
    return datetime.now().isoformat()


def ejecutar_caso(
    nombre_caso: str,
    dir_casos: str,
    archivo_aspectos: str,
    pipeline: EvaluationPipeline,
    dir_salida: str
) -> dict:
    """
    Ejecuta evaluación completa para un caso de prueba.

    Sigue el pipeline LEVEL 0 → LEVEL 5.
    """
    dir_caso = os.path.join(dir_casos, nombre_caso)
    
    print(f"\n{'='*50}")
    print(f"Caso: {nombre_caso}")
    print(f"{'='*50}")
    
    # =====================================================
    # Cargar datos
    # =====================================================
    metadata = cargar_metadata(dir_caso)
    print(f"Descripción: {metadata['descripcion']}")
    print(f"Dominio: {metadata['dominio']}")
    
    # Historias
    ruta_generadas = os.path.join(dir_caso, "generadas.txt")
    historias_generadas = cargar_historias(ruta_generadas)
    print(f"Historias generadas: {len(historias_generadas)}")
    
    ruta_esperadas = os.path.join(dir_caso, "esperadas.txt")
    tiene_esperadas = os.path.exists(ruta_esperadas)
    
    if tiene_esperadas:
        historias_esperadas = cargar_historias(ruta_esperadas)
        print(f"Historias esperadas: {len(historias_esperadas)}")
    else:
        print("Sin historias esperadas - abortando caso")
        return None
    
    # Criterios de aceptación
    ruta_ca_gen = os.path.join(dir_caso, "ca_generados.json")
    ruta_ca_exp = os.path.join(dir_caso, "ca_esperados.json")
    
    criterios_generados = []
    criterios_esperados = []
    
    if os.path.exists(ruta_ca_gen):
        criterios_generados = cargar_criterios(ruta_ca_gen)
        print(f"CA generados: {len(criterios_generados)} historias")
    
    if os.path.exists(ruta_ca_exp):
        criterios_esperados = cargar_criterios(ruta_ca_exp)
        print(f"CA esperados: {len(criterios_esperados)} historias")
    
    # Aspectos del dominio
    aspectos = cargar_aspectos(archivo_aspectos)
    print(f"Aspectos: {len(aspectos)}")
    
    # =====================================================
    # Preparar input para el pipeline
    # =====================================================
    input_data = EvaluationInput(
        stories_generated=historias_generadas,
        stories_expected=historias_esperadas,
        ca_generated=criterios_generados,
        ca_expected=criterios_esperados,
        domain_concepts=aspectos,
        metadata=metadata
    )
    
    # =====================================================
    # Ejecutar pipeline LEVEL 0 → LEVEL 5
    # =====================================================
    print("\nEjecutando pipeline de evaluación...")
    t_inicio = datetime.now()
    resultado = pipeline.run(input_data)
    
    # =====================================================
    # Mostrar resumen
    # =====================================================
    print("\n--- RESUMEN ---")
    summary = resultado.summary
    
    print(f"\nLEVEL 1 - Alineación:")
    print(f"  Historias alineadas: {summary['input_stats']['stories_aligned']}/{summary['input_stats']['stories_generated']}")
    print(f"  Tasa de alineación: {summary['input_stats']['alignment_rate']:.2%}")
    print(f"  SBERT media: {summary['level_1_alignment']['sbert_mean']:.3f}")
    print(f"  Alineadas (Strong+Conservative): {summary['level_1_alignment']['aligned_count']}")
    print(f"  Strong: {summary['level_1_alignment']['strong_count']}, Conservative: {summary['level_1_alignment']['conservative_count']}, Weak: {summary['level_1_alignment']['weak_count']}")
    
    print(f"\nLEVEL 2 - Coverage:")
    print(f"  Story coverage: {summary['level_2_coverage']['story_coverage']:.2%}")
    print(f"  Cubiertas: {summary['level_2_coverage']['stories_covered']}/{summary['level_2_coverage']['stories_total']}")
    
    if "level_3_criteria" in summary:
        l3 = summary["level_3_criteria"]
        print(f"\nLEVEL 3 - Criterios de Aceptación:")
        print(f"  Pares evaluados: {l3['pairs_evaluated']}")
        print(f"  Cobertura funcional promedio: {l3['avg_functional_coverage']:.2%}")
        print(f"  Verificabilidad global: {l3['global_verifiability']:.2%}")
        print(f"  No-ambigüedad global: {l3['global_no_ambiguity']:.2%}")
        print(f"  Ambigüedad global: {l3['global_ambiguity']:.2%}")
    
    if "level_4_concepts" in summary:
        l4 = summary["level_4_concepts"]
        print(f"\nLEVEL 4 - Coverage Conceptual:")
        print(f"  Cobertura: {l4['concept_coverage']:.2%}")
        print(f"  Conceptos cubiertos: {l4['concepts_covered']}/{l4['concepts_total']}")

    if "level_5_invest" in summary:
        l5 = summary["level_5_invest"]
        print(f"\nLEVEL 5 - Evaluación INVEST ({l5.get('model_used', '?')}):")
        print(f"  Puntaje global: {l5['overall_mean']:.2f}/5.0")
        print(f"  Historias evaluadas: {l5['valid_stories']} (errores: {l5['error_stories']})")
        scores = l5.get("scores_per_criterion", {})
        for c, mean in scores.items():
            bar = "█" * int(round(mean)) + "░" * (5 - int(round(mean)))
            print(f"  {c}: {bar} {mean:.2f}")

    duracion = (datetime.now() - t_inicio).total_seconds()
    print(f"\nTiempo total del caso: {duracion:.1f}s")
    
    # =====================================================
    # Guardar resultados
    # =====================================================
    ts = timestamp()
    ts_safe = ts.replace(":", "-").replace(".", "-")
    
    resultado_dict = pipeline.to_dict(resultado)
    resultado_dict["caso"] = nombre_caso
    resultado_dict["num_generadas"] = len(historias_generadas)
    resultado_dict["num_esperadas"] = len(historias_esperadas)
    
    ruta_json = os.path.join(dir_salida, f"{nombre_caso}_{ts_safe}.json")
    guardar_json(resultado_dict, ruta_json)
    print(f"\nGuardado: {ruta_json}")
    
    return resultado_dict


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación experimental de HU - Pipeline LEVEL 0-5"
    )
    
    parser.add_argument("--caso", required=True, help="Nombre del caso o 'todos'")
    parser.add_argument("--salida", default="resultados", help="Directorio de salida")
    parser.add_argument("--casos-dir", default="casos_prueba", help="Directorio de casos")
    parser.add_argument("--aspectos", default="casos_prueba/aspectos_hospital.json", help="Archivo de aspectos")
    parser.add_argument("--diversidad-json", default=None, help="JSON para comparar modelos")
    
    args = parser.parse_args()
    
    os.makedirs(args.salida, exist_ok=True)
    
    # Inicializar pipeline
    print("Inicializando pipeline de evaluación...")
    pipeline = EvaluationPipeline()
    
    # Determinar casos a ejecutar
    if args.caso == "todos":
        casos = descubrir_casos(args.casos_dir)
        print(f"Casos encontrados: {len(casos)}")
    else:
        casos = [args.caso]
    
    # Ejecutar casos
    resultados_todos = []
    for nombre_caso in casos:
        try:
            resultado = ejecutar_caso(
                nombre_caso,
                args.casos_dir,
                args.aspectos,
                pipeline,
                args.salida
            )
            if resultado:
                resultados_todos.append(resultado)
        except FileNotFoundError as e:
            print(f"Error en {nombre_caso}: {e}")
        except Exception as e:
            print(f"Error en {nombre_caso}: {e}")
            raise
    
    # Evaluar diversidad si se especificó
    if args.diversidad_json:
        print("\n" + "="*50)
        print("Evaluando diversidad entre modelos")
        print("="*50)
        
        nombres_modelos, conjuntos = cargar_conjuntos_diversidad(args.diversidad_json)
        
        resultado_div = pipeline.evaluate_diversity(
            model_story_sets=conjuntos,
            model_names=nombres_modelos
        )
        
        print(f"Diversidad media: {resultado_div['diversity_mean']:.2f}%")
        print(f"Diversidad std: {resultado_div['diversity_std']:.2f}%")
        
        ruta_div = os.path.join(args.salida, "diversidad_modelos.json")
        guardar_json(resultado_div, ruta_div)
        print(f"Guardado: {ruta_div}")
    
    print(f"\n{'='*50}")
    print(f"Ejecutados: {len(resultados_todos)}/{len(casos)}")
    print(f"Resultados en: {args.salida}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
