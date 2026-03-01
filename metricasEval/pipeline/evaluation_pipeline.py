"""
EvaluationPipeline — Orquestador LEVEL 0 → LEVEL 5

Ejecuta el pipeline completo de evaluación metodológica
siguiendo la secuencia definida:

LEVEL 0: Representación Semántica (encoding)
LEVEL 1: Alineación de HU (SBERT + BERTScore)
LEVEL 2: Coverage de HU (expected con match)
LEVEL 3: Evaluación de CA (solo alineadas)
LEVEL 4: Coverage Conceptual (aspectos/dominio)
LEVEL 5: Evaluación INVEST con LLM (gpt-4o)
"""

import time
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

from .semantic_encoder import SemanticEncoder
from .story_alignment_evaluator import StoryAlignmentEvaluator
from .story_coverage_calculator import StoryCoverageCalculator
from .acceptance_criteria_evaluator import AcceptanceCriteriaEvaluator
from .concept_coverage_evaluator import ConceptCoverageEvaluator
from .invest_evaluator import InvestEvaluator, InvestEvaluationResult


@dataclass
class EvaluationInput:
    """Datos de entrada para el pipeline de evaluación."""
    stories_generated: List[str]
    stories_expected: List[str]
    ca_generated: List[List[str]] = field(default_factory=list)
    ca_expected: List[List[str]] = field(default_factory=list)
    domain_concepts: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EvaluationOutput:
    """Resultado completo del pipeline de evaluación."""
    timestamp: str
    metadata: Dict
    config: Dict
    level_1_alignment: Dict
    level_2_coverage: Dict
    level_3_criteria: Optional[Dict]
    level_4_concepts: Optional[Dict]
    level_5_invest: Optional[Dict]
    summary: Dict


class EvaluationPipeline:
    """
    Orquestador del pipeline de evaluación metodológica.
    
    Ejecuta los 5 niveles en secuencia correcta,
    asegurando que cada nivel reciba solo los datos
    que le corresponden según la metodología.
    """
    
    def __init__(
        self,
        sbert_model: str = None,
        device: str = None,
        umbral_strong: float = 0.80,
        umbral_conservative: float = 0.85,
        coverage_threshold: float = 0.75,
        ca_coverage_threshold: float = 0.75,
        concept_threshold: float = 0.70,
        openai_api_key: str = None,
        invest_model: str = "gpt-4o",
        enable_invest: bool = True
    ):
        """
        Inicializa el pipeline con todos los evaluadores.

        Args:
            sbert_model: Modelo SBERT a usar.
            device: Dispositivo (cuda/cpu).
            umbral_strong: Umbral para alineación strong.
            umbral_conservative: Umbral para alineación conservative.
            coverage_threshold: Umbral para story coverage.
            ca_coverage_threshold: Umbral para CA coverage.
            concept_threshold: Umbral para concept coverage.
            openai_api_key: API key de OpenAI. Si es None, usa OPENAI_API_KEY del entorno.
            invest_model: Modelo de OpenAI para LEVEL 5 (por defecto "gpt-4o").
            enable_invest: Si False, omite el LEVEL 5 (útil si no hay API key).
        """
        # LEVEL 0: Encoder semántico
        self.encoder = SemanticEncoder(
            model_name=sbert_model,
            device=device
        ) if sbert_model else SemanticEncoder(device=device)

        # LEVEL 1: Alineación de historias
        self.alignment_evaluator = StoryAlignmentEvaluator(
            encoder=self.encoder,
            umbral_strong=umbral_strong,
            umbral_conservative=umbral_conservative
        )

        # LEVEL 2: Coverage de historias
        self.coverage_calculator = StoryCoverageCalculator(
            encoder=self.encoder,
            threshold=coverage_threshold
        )

        # LEVEL 3: Evaluación de CA
        self.criteria_evaluator = AcceptanceCriteriaEvaluator(
            encoder=self.encoder,
            coverage_threshold=ca_coverage_threshold
        )

        # LEVEL 4: Coverage conceptual
        self.concept_evaluator = ConceptCoverageEvaluator(
            encoder=self.encoder,
            threshold=concept_threshold
        )

        # LEVEL 5: Evaluador INVEST con LLM
        self.invest_evaluator = None
        if enable_invest:
            self.invest_evaluator = InvestEvaluator(
                api_key=openai_api_key,
                model=invest_model
            )

        # Configuración
        self._config = {
            "umbral_strong": umbral_strong,
            "umbral_conservative": umbral_conservative,
            "coverage_threshold": coverage_threshold,
            "ca_coverage_threshold": ca_coverage_threshold,
            "concept_threshold": concept_threshold,
            "encoder": self.encoder.get_config(),
            "invest_model": invest_model,
            "enable_invest": enable_invest,
        }
    
    def run(self, input_data: EvaluationInput) -> EvaluationOutput:
        """
        Ejecuta el pipeline completo de evaluación.

        LEVEL 0: Encoding (implícito en cada evaluador)
        LEVEL 1: Alineación de HU
        LEVEL 2: Coverage de HU
        LEVEL 3: Evaluación de CA (solo historias alineadas)
        LEVEL 4: Coverage conceptual (solo historias alineadas)
        LEVEL 5: Evaluación INVEST con LLM (historias generadas)

        Args:
            input_data: Datos de entrada encapsulados.

        Returns:
            EvaluationOutput con resultados de todos los niveles.
        """
        timestamp = datetime.now().isoformat()
        t_total = time.time()

        # =====================================================
        # LEVEL 1: Alineación de HU
        # =====================================================
        print("[LEVEL 1] Alineación de historias (SBERT + BERTScore)...", flush=True)
        t0 = time.time()
        level_1_result = self.alignment_evaluator.evaluate(
            stories_generated=input_data.stories_generated,
            stories_expected=input_data.stories_expected
        )
        print(f"[LEVEL 1] OK ({time.time() - t0:.1f}s) — "
              f"alineadas: {level_1_result['aggregate']['total_aligned']}/{len(input_data.stories_generated)}")

        # Extraer pares alineados para LEVEL 3 y 4
        aligned_pairs = self.alignment_evaluator.get_aligned_pairs(level_1_result)
        aligned_stories_text = [
            a.text_generated for a in level_1_result["aligned_stories"]
        ]

        # =====================================================
        # LEVEL 2: Coverage de HU
        # =====================================================
        print("[LEVEL 2] Calculando story coverage...", flush=True)
        t0 = time.time()
        coverage_score, coverage_detail = self.coverage_calculator.calculate_coverage(
            stories_generated=input_data.stories_generated,
            stories_expected=input_data.stories_expected
        )
        print(f"[LEVEL 2] OK ({time.time() - t0:.1f}s) — coverage: {coverage_score:.2%}")

        level_2_result = {
            "story_coverage_score": coverage_score,
            "diversity_complement": 100.0 * (1.0 - coverage_score),
            **coverage_detail
        }

        # =====================================================
        # LEVEL 3: Evaluación de CA (SOLO historias alineadas)
        # =====================================================
        level_3_result = None
        if input_data.ca_generated and input_data.ca_expected and aligned_pairs:
            print("[LEVEL 3] Evaluando criterios de aceptación...", flush=True)
            t0 = time.time()
            level_3_result = self.criteria_evaluator.evaluate_aligned_stories(
                aligned_pairs=aligned_pairs,
                all_ca_generated=input_data.ca_generated,
                all_ca_expected=input_data.ca_expected
            )
            print(f"[LEVEL 3] OK ({time.time() - t0:.1f}s)")
        else:
            print("[LEVEL 3] Omitido (sin CA o sin pares alineados)")

        # =====================================================
        # LEVEL 4: Coverage conceptual (SOLO historias alineadas)
        # =====================================================
        level_4_result = None
        if input_data.domain_concepts and aligned_stories_text:
            print("[LEVEL 4] Evaluando cobertura conceptual...", flush=True)
            t0 = time.time()
            level_4_result = self.concept_evaluator.evaluate_aspect_coverage(
                aligned_stories=aligned_stories_text,
                aspects=input_data.domain_concepts
            )
            print(f"[LEVEL 4] OK ({time.time() - t0:.1f}s) — "
                  f"cobertura: {level_4_result.get('aspect_coverage_score', 0):.2%}")
        else:
            print("[LEVEL 4] Omitido (sin conceptos o sin historias alineadas)")

        # =====================================================
        # LEVEL 5: Evaluación INVEST con LLM
        # =====================================================
        level_5_result = None
        if self.invest_evaluator:
            print(f"[LEVEL 5] Evaluando INVEST con {self._config['invest_model']}...", flush=True)
            t0 = time.time()
            invest_output = self.invest_evaluator.evaluate(
                stories=input_data.stories_generated,
                metadata=input_data.metadata
            )
            level_5_result = self._serialize_invest(invest_output)
            overall = level_5_result["aggregate"].get("overall_mean", 0)
            print(f"[LEVEL 5] OK ({time.time() - t0:.1f}s) — INVEST mean: {overall:.2f}/5.0")
        else:
            print("[LEVEL 5] Omitido (INVEST deshabilitado)")

        print(f"\n[Pipeline] Completado en {time.time() - t_total:.1f}s totales")

        # =====================================================
        # Resumen ejecutivo
        # =====================================================
        summary = self._build_summary(
            input_data, level_1_result, level_2_result,
            level_3_result, level_4_result, level_5_result
        )

        # Serializar alignments para JSON
        level_1_serializable = self._serialize_level_1(level_1_result)

        return EvaluationOutput(
            timestamp=timestamp,
            metadata=input_data.metadata,
            config=self._config,
            level_1_alignment=level_1_serializable,
            level_2_coverage=level_2_result,
            level_3_criteria=level_3_result,
            level_4_concepts=level_4_result,
            level_5_invest=level_5_result,
            summary=summary
        )
    
    def _serialize_invest(self, result: InvestEvaluationResult) -> Dict:
        """Convierte InvestEvaluationResult a diccionario serializable."""
        return {
            "story_results": [asdict(r) for r in result.story_results],
            "aggregate": result.aggregate,
            "language_detected": result.language_detected,
            "model_used": result.model_used,
            "total_stories": result.total_stories,
        }

    def _serialize_level_1(self, level_1_result: Dict) -> Dict:
        """Convierte objetos AlignmentResult a diccionarios."""
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
        level_4: Optional[Dict],
        level_5: Optional[Dict] = None
    ) -> Dict:
        """Construye resumen ejecutivo de la evaluación."""
        summary = {
            "input_stats": {
                "stories_generated": len(input_data.stories_generated),
                "stories_expected": len(input_data.stories_expected),
                "stories_aligned": level_1["aggregate"]["total_aligned"],
                "alignment_rate": level_1["aggregate"]["alignment_rate"],
            },
            "level_1_alignment": {
                "sbert_mean": level_1["aggregate"]["sbert"]["mean"],
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

        if level_5:
            agg = level_5.get("aggregate", {})
            summary["level_5_invest"] = {
                "overall_mean": agg.get("overall_mean", 0.0),
                "valid_stories": agg.get("valid_stories", 0),
                "error_stories": agg.get("error_stories", 0),
                "scores_per_criterion": {
                    c: agg.get("scores_per_criterion", {}).get(c, {}).get("mean", 0.0)
                    for c in ["I", "N", "V", "E", "S", "T"]
                },
                "model_used": level_5.get("model_used"),
                "language_detected": level_5.get("language_detected"),
            }

        return summary
    
    def evaluate_diversity(
        self,
        model_story_sets: List[List[str]],
        model_names: List[str] = None
    ) -> Dict:
        """
        Evalúa diversidad entre múltiples modelos.
        
        Método separado porque diversidad compara outputs
        de diferentes herramientas, no contra referencia.
        
        Args:
            model_story_sets: Lista de conjuntos de HU por modelo.
            model_names: Nombres de los modelos.
            
        Returns:
            Resultado de diversidad.
        """
        return self.coverage_calculator.evaluate_multi_model_diversity(
            model_story_sets=model_story_sets,
            model_names=model_names
        )
    
    def get_config(self) -> Dict:
        """Retorna configuración completa del pipeline."""
        return self._config
    
    def to_dict(self, output: EvaluationOutput) -> Dict:
        """Convierte EvaluationOutput a diccionario serializable."""
        return {
            "timestamp": output.timestamp,
            "metadata": output.metadata,
            "config": output.config,
            "level_1_alignment": output.level_1_alignment,
            "level_2_coverage": output.level_2_coverage,
            "level_3_criteria": output.level_3_criteria,
            "level_4_concepts": output.level_4_concepts,
            "level_5_invest": output.level_5_invest,
            "summary": output.summary
        }
