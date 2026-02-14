from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
import json
import argparse
from pathlib import Path

from .semantic_encoder import SemanticEncoder
from .evaluar_alineacion_historias import EvaluadorAlineacionHistorias
from .calcular_cobertura_historias import CalculadorCoberturaHistorias
from .evaluador_ca import EvaluadorCriteriosAceptacion
from .evaluar_cobertura_conceptos import EvaluadorCoberturaConceptos

@dataclass
class EvaluationInput:
    historias_generadas: List[str]
    historias_esperadas: List[str]
    ca_generados: List[List[str]] = field(default_factory=list)
    ca_esperados: List[List[str]] = field(default_factory=list)
    conceptos_dominio: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)



@dataclass 
class EvaluationOutput:
    timestamp: str
    metadata: Dict
    config: Dict
    alineacion_nivel1: Dict
    cobertura_nivel2: Dict
    criterios_nivel3: Optional[Dict]
    conceptos_nivel4: Optional[Dict]
    resumen: Dict


class EvaluationPipeline:
    def __init__(
        self,
        sbert_model: str = None,
        device: str = None,
        umbral_fuerte: float = 0.80,
        umbral_conservador: float = 0.85,
        umbral_cobertura: float = 0.75,
        umbral_ca_cobertura: float = 0.75,
        umbral_concepto: float = 0.70
    ):

        #encoder para todos los evaluadores
        self.encoder = SemanticEncoder(
            model_name=sbert_model,
            device=device
        ) if sbert_model else SemanticEncoder(device=device)
        
        #Nivel 1: Alineación de historias
        self.alignment_evaluator = EvaluadorAlineacionHistorias(
            encoder=self.encoder,
            umbral_fuerte=umbral_fuerte,
            umbral_conservador=umbral_conservador
        )
        
        #Nivel 2: Coverage de historias
        self.coverage_calculator = CalculadorCoberturaHistorias(
            encoder=self.encoder,
            threshold=umbral_cobertura
        )
        
        #Nivel 3: Evaluación de CA
        self.criteria_evaluator = EvaluadorCriteriosAceptacion(
            encoder=self.encoder,
            coverage_threshold=umbral_ca_cobertura
        )
        
        #Nivel 4: Coverage conceptual
        self.concept_evaluator = EvaluadorCoberturaConceptos(
            encoder=self.encoder,
            threshold=umbral_concepto
        )
        
        # Configuración
        self._config = {
            "umbral_fuerte": umbral_fuerte,
            "umbral_conservador": umbral_conservador,
            "umbral_cobertura": umbral_cobertura,
            "umbral_ca_cobertura": umbral_ca_cobertura,
            "umbral_concepto": umbral_concepto,
            "encoder": self.encoder.get_config()
        }
    
    def run(self, input_data: EvaluationInput) -> EvaluationOutput:
        timestamp = datetime.now().isoformat()
        
        #nivel 1: alineacion de hu
        level_1_result = self.alignment_evaluator.evaluate(
            stories_generated=input_data.historias_generadas,
            stories_expected=input_data.historias_esperadas
        )
        
        #saco los pares alineados para usarlos en el nivel 3 y 4
        aligned_pairs = self.alignment_evaluator.get_aligned_pairs(level_1_result)
        aligned_stories_text = [
            a.text_generated for a in level_1_result["aligned_stories"]
        ]
        
        #Nivel 2: Coverage de historias
        coverage_score, coverage_detail = self.coverage_calculator.calculate_coverage(
            stories_generated=input_data.historias_generadas,
            stories_expected=input_data.historias_esperadas
        )
        
        level_2_result = {
            "story_coverage_score": coverage_score,
            "diversity_complement": 100.0 * (1.0 - coverage_score),
            **coverage_detail
        }
        
        #Nivel 3 : Eval criterios aceptacion de las historias alineadas
        level_3_result = None
        if input_data.ca_generados and input_data.ca_esperados and aligned_pairs:
            level_3_result = self.criteria_evaluator.evaluate_aligned_stories(
                aligned_pairs=aligned_pairs,
                all_ca_generated=input_data.ca_generados,
                all_ca_expected=input_data.ca_esperados
            )
        
        #Nivel 4: Coverage de conceptos en las historias alineadas
        level_4_result = None
        if input_data.conceptos_dominio and aligned_stories_text:
            level_4_result = self.concept_evaluator.evaluate_aspect_coverage(
                aligned_stories=aligned_stories_text,
                aspects=input_data.conceptos_dominio
            )
        
        summary = self._build_summary(
            input_data, level_1_result, level_2_result,
            level_3_result, level_4_result
        )
        
        level_1_serializable = self._serialize_level_1(level_1_result)
        
        return EvaluationOutput(
            timestamp=timestamp,
            metadata=input_data.metadata,
            config=self._config,
            alineacion_nivel1=level_1_serializable,
            cobertura_nivel2=level_2_result,
            criterios_nivel3=level_3_result,
            conceptos_nivel4=level_4_result,
            resumen=summary
        )
    
    def _serialize_level_1(self, level_1_result: Dict) -> Dict:
        alignments_list = []
        for a in level_1_result["alignments"]:
            alignments_list.append({
                "index_generated": a.index_generated,
                "text_generated": a.text_generated,
                "index_matched": a.index_matched,
                "text_matched": a.text_matched,
                "sbert_similarity": a.sbert_similarity,
                "bertscore_f1": a.bertscore_f1,
                "alignment_level": a.alignment_level,
                "is_aligned": a.is_aligned
            })
        
        aligned_list = []
        for a in level_1_result["aligned_stories"]:
            aligned_list.append({
                "index_generated": a.index_generated,
                "index_matched": a.index_matched,
                "sbert_similarity": a.sbert_similarity,
                "alignment_level": a.alignment_level
            })
        
        return {
            "alignments": alignments_list,
            "aligned_stories": aligned_list,
            "aggregate": level_1_result["aggregate"],
            "config": level_1_result["config"]
        }
    
    def _build_summary(
        self,
        input_data: EvaluationInput,
        level_1: Dict,
        level_2: Dict,
        level_3: Optional[Dict],
        level_4: Optional[Dict]
    ) -> Dict:
        summary = {
            "input_stats": {
                "stories_generated": len(input_data.historias_generadas),
                "stories_expected": len(input_data.historias_esperadas),
                "stories_aligned": level_1["aggregate"]["total_aligned"],
                "alignment_rate": level_1["aggregate"]["alignment_rate"],
            },
            "level_1_alignment": {
                "sbert_mean_aligned": level_1["aggregate"]["sbert"]["mean_aligned"],
                "sbert_mean_all": level_1["aggregate"]["sbert"]["mean_all"],
                "aligned_count": level_1["aggregate"]["sbert"]["aligned_count"],
                "strong_count": level_1["aggregate"]["sbert"]["strong_count"],
                "conservative_count": level_1["aggregate"]["sbert"]["conservative_count"],
                "weak_count": level_1["aggregate"]["sbert"]["weak_count"],
            },
            "level_2_coverage": {
                "story_coverage": level_2["story_coverage_score"],
                "stories_covered": level_2["covered"],
                "stories_total": level_2["total"],
            }
        }
        
        if level_3:
            no_ambiguity = level_3["global_metrics"]["ambiguity"]["score"]
            summary["level_3_criteria"] = {
                "pairs_evaluated": level_3["aggregate"]["num_aligned_pairs"],
                "avg_functional_coverage": level_3["aggregate"]["avg_functional_coverage"],
                "avg_verifiability": level_3["aggregate"]["avg_verifiability_score"],
                "avg_ambiguity_score": level_3["aggregate"]["avg_ambiguity_score"],
                "global_verifiability": level_3["global_metrics"]["verifiability"]["score"],
                "global_no_ambiguity": no_ambiguity,
                "global_ambiguity": 1.0 - no_ambiguity,
            }
        
        if level_4:
            summary["level_4_concepts"] = {
                "concept_coverage": level_4["aspect_coverage_score"],
                "concepts_covered": level_4["aspects_covered"],
                "concepts_total": level_4["aspects_total"],
            }
        
        return summary
    

    # Esta funcion es para evaluar la diversidad entre conj de historias generadas por disintos LLM's
    def evaluate_diversity(
        self,
        model_story_sets: List[List[str]],
        model_names: List[str] = None
    ) -> Dict:
        return self.coverage_calculator.evaluate_multi_model_diversity(
            model_story_sets=model_story_sets,
            model_names=model_names
        )
    
    def get_config(self) -> Dict:
        return self._config
    
    def to_dict(self, output: EvaluationOutput) -> Dict:
        return {
            "timestamp": output.timestamp,
            "metadata": output.metadata,
            "config": output.config,
            "level_1_alignment": output.alineacion_nivel1,
            "level_2_coverage": output.cobertura_nivel2,
            "level_3_criteria": output.criterios_nivel3,
            "level_4_concepts": output.conceptos_nivel4,
            "summary": output.resumen
        }


def cargar_historias(archivo: Path) -> List[str]:
    with open(archivo, 'r', encoding='utf-8') as f:
        return [linea.strip() for linea in f if linea.strip()]


def cargar_json(archivo: Path):
    with open(archivo, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Ejecutar pipeline de evaluación de historias de usuario',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--caso', required=True,
                        help='Nombre del directorio del caso (ej: caso_A_alineadas)')
    parser.add_argument('--aspectos', default=None,
                        help='Ruta al archivo JSON de aspectos/conceptos del dominio')
    parser.add_argument('--output', default=None,
                        help='Archivo de salida para los resultados (default: resultados/<caso>_resultado.json)')
    parser.add_argument('--umbral-fuerte', type=float, default=0.80,
                        help='Umbral para alineación fuerte (default: 0.80)')
    parser.add_argument('--umbral-conservador', type=float, default=0.85,
                        help='Umbral para alineación conservadora (default: 0.85)')
    parser.add_argument('--umbral-cobertura', type=float, default=0.75,
                        help='Umbral para cobertura de historias (default: 0.75)')
    parser.add_argument('--umbral-ca', type=float, default=0.75,
                        help='Umbral para cobertura de CA (default: 0.75)')
    parser.add_argument('--umbral-concepto', type=float, default=0.70,
                        help='Umbral para cobertura de conceptos (default: 0.70)')
    
    args = parser.parse_args()
    
    caso_dir = Path('casos_prueba') / args.caso
    
    if not caso_dir.exists():
        return 1
    
    
    esperadas_file = caso_dir / 'esperadas.txt'
    generadas_file = caso_dir / 'generadas.txt'
    
    if not esperadas_file.exists():
        return 1
    if not generadas_file.exists():
        return 1
    
    historias_esperadas = cargar_historias(esperadas_file)
    historias_generadas = cargar_historias(generadas_file)
    
    print(f"Historias esperadas: {len(historias_esperadas)}")
    print(f"Historias generadas: {len(historias_generadas)}")
    
    ca_esperados = []
    ca_generados = []
    
    ca_esperados_file = caso_dir / 'ca_esperados.json'
    ca_generados_file = caso_dir / 'ca_generados.json'
    
    if ca_esperados_file.exists() and ca_generados_file.exists():
        ca_esperados = cargar_json(ca_esperados_file)
        ca_generados = cargar_json(ca_generados_file)
        print(f"CA esperados: {len(ca_esperados)} historias")
        print(f"CA generados: {len(ca_generados)} historias")
    else:
        print("No se evaluarán criterios de aceptación (archivos CA no encontrados)")
    
    conceptos_dominio = {}
    
    if args.aspectos:
        aspectos_file = Path(args.aspectos)
        if aspectos_file.exists():
            conceptos_dominio = cargar_json(aspectos_file)
            print(f"Conceptos del dominio: {len(conceptos_dominio)}")
        else:
            print(f"Advertencia: No se encontró el archivo de aspectos: {aspectos_file}")
    else:
        print("No se evaluará cobertura de conceptos (--aspectos no especificado)")
    
    # Cargar metadata si existe
    metadata = {}
    metadata_file = caso_dir / 'metadata.json'
    if metadata_file.exists():
        metadata = cargar_json(metadata_file)
        print(f"Metadata cargada")
    
    # Crear input para el pipeline
    input_data = EvaluationInput(
        historias_generadas=historias_generadas,
        historias_esperadas=historias_esperadas,
        ca_generados=ca_generados,
        ca_esperados=ca_esperados,
        conceptos_dominio=conceptos_dominio,
        metadata=metadata
    )
    
    # Inicializar pipeline
    print("Inicializando pipeline...")
    pipeline = EvaluationPipeline(
        umbral_fuerte=args.umbral_fuerte,
        umbral_conservador=args.umbral_conservador,
        umbral_cobertura=args.umbral_cobertura,
        umbral_ca_cobertura=args.umbral_ca,
        umbral_concepto=args.umbral_concepto
    )
    
    # Ejecutar evaluación
    print("\nEjecutando evaluación...")
    
    resultado = pipeline.run(input_data)
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    resumen = resultado.resumen
    
    print("\nNIVEL 1 - Alineación de Historias:")
    l1 = resumen['level_1_alignment']
    input_stats = resumen['input_stats']
    print(f"  • Historias alineadas: {input_stats['stories_aligned']}/{input_stats['stories_generated']}")
    print(f"  • Tasa de alineación: {input_stats['alignment_rate']:.1%}")
    print(f"  • Similitud promedio (alineadas): {l1['sbert_mean_aligned']:.3f}")
    print(f"  • Alineaciones fuertes: {l1['strong_count']}")
    print(f"  • Alineaciones conservadoras: {l1['conservative_count']}")
    print(f"  • Alineaciones débiles: {l1['weak_count']}")
    
    print("\nNIVEL 2 - Cobertura de Historias:")
    l2 = resumen['level_2_coverage']
    print(f"  • Cobertura: {l2['story_coverage']:.1%}")
    print(f"  • Historias cubiertas: {l2['stories_covered']}/{l2['stories_total']}")
    
    if 'level_3_criteria' in resumen:
        print("\nNIVEL 3 - Criterios de Aceptación:")
        l3 = resumen['level_3_criteria']
        print(f"  • Pares evaluados: {l3['pairs_evaluated']}")
        print(f"  • Cobertura funcional promedio: {l3['avg_functional_coverage']:.1%}")
        print(f"  • Verificabilidad promedio: {l3['avg_verifiability']:.3f}")
        print(f"  • Sin ambigüedad (global): {l3['global_no_ambiguity']:.1%}")
    
    if 'level_4_concepts' in resumen:
        print("\nNIVEL 4 - Cobertura de Conceptos:")
        l4 = resumen['level_4_concepts']
        print(f"  • Cobertura de conceptos: {l4['concept_coverage']:.1%}")
        print(f"  • Conceptos cubiertos: {l4['concepts_covered']}/{l4['concepts_total']}")
    
    # Guardar resultado completo
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path('resultados') / f"{args.caso}_resultado.json"
    
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline.to_dict(resultado), f, indent=2, ensure_ascii=False)
    
    print(f"\nResultado completo guardado en: {output_file}")
    print("\nEvaluación completada exitosamente!\n")
    
    return 0


if __name__ == "__main__":
    exit(main())