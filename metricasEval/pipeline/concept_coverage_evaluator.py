"""
LEVEL 4 — Coverage Conceptual

Responsabilidades:
- Usar lista de conceptos/aspectos del dominio
- Evaluar SOLO sobre historias alineadas
- Calcular qué proporción de conceptos están cubiertos

Conceptos pueden ser:
- Aspectos funcionales del sistema
- Entidades del dominio
- Stakeholders
- Requisitos no funcionales
"""

from sentence_transformers import util
from typing import List, Dict, Tuple
import numpy as np

# Umbral para considerar un concepto cubierto
CONCEPT_COVERAGE_THRESHOLD = 0.70


class ConceptCoverageEvaluator:
    """
    LEVEL 4: Evaluador de Cobertura Conceptual.
    
    Evalúa qué proporción de conceptos del dominio
    están representados en las historias alineadas.
    
    IMPORTANTE: Solo evalúa sobre historias que pasaron
    el filtro de alineación en LEVEL 1.
    """
    
    def __init__(
        self,
        encoder,
        threshold: float = CONCEPT_COVERAGE_THRESHOLD
    ):
        self.encoder = encoder
        self.threshold = threshold
    
    def evaluate_concept_coverage(
        self,
        aligned_stories: List[str],
        concepts: Dict[str, List[str]],
        threshold: float = None
    ) -> Tuple[float, Dict]:
        """
        Evalúa cobertura de conceptos del dominio.
        
        Para cada concepto (conjunto de descripciones),
        verifica si alguna historia alineada lo cubre.
        
        Args:
            aligned_stories: Textos de historias alineadas.
            concepts: Dict {nombre_concepto: [descripciones]}.
            threshold: Umbral de similitud.
            
        Returns:
            (score, detalle)
        """
        threshold = threshold or self.threshold
        
        if not concepts:
            return 1.0, {"covered": 0, "total": 0, "details": []}
        
        if not aligned_stories:
            return 0.0, {
                "covered": 0,
                "total": len(concepts),
                "details": [
                    {"concept": name, "covered": False, "max_similarity": 0.0}
                    for name in concepts.keys()
                ]
            }
        
        # Codificar historias alineadas
        emb_stories = self.encoder.encode_stories(aligned_stories)
        if emb_stories.dim() == 1:
            emb_stories = emb_stories.unsqueeze(0)
        
        covered = 0
        details = []
        
        for concept_name, descriptions in concepts.items():
            # Codificar descripciones del concepto
            emb_concept = self.encoder.encode(descriptions, convert_to_tensor=True)

            if emb_concept.dim() == 1:
                emb_concept = emb_concept.unsqueeze(0)
            
            # Buscar máxima similitud entre cualquier descripción
            # del concepto y cualquier historia alineada
            max_sim = 0.0
            best_story_idx = -1
            best_description = None
            
            for desc_idx, emb_desc in enumerate(emb_concept):
                emb_desc_2d = emb_desc if emb_desc.dim() == 2 else emb_desc.unsqueeze(0)
                sims = util.cos_sim(emb_desc_2d, emb_stories)[0]
                sim_max = float(sims.max().item())
                
                if sim_max > max_sim:
                    max_sim = sim_max
                    best_story_idx = int(sims.argmax().item())
                    best_description = descriptions[desc_idx]
            
            is_covered = max_sim >= threshold
            if is_covered:
                covered += 1
            
            details.append({
                "concept": concept_name,
                "covered": is_covered,
                "max_similarity": max_sim,
                "best_description": best_description,
                "best_story_idx": best_story_idx,
                "best_story": aligned_stories[best_story_idx] if best_story_idx >= 0 else None
            })
        
        score = covered / len(concepts)
        
        return score, {
            "covered": covered,
            "total": len(concepts),
            "threshold": threshold,
            "details": details
        }
    
    def evaluate_aspect_coverage(
        self,
        aligned_stories: List[str],
        aspects: Dict[str, List[str]],
        threshold: float = None
    ) -> Dict:
        """
        Evalúa cobertura de aspectos funcionales.
        
        Wrapper de evaluate_concept_coverage para aspectos.
        
        Args:
            aligned_stories: Historias alineadas.
            aspects: Dict {nombre_aspecto: [descripciones]}.
            threshold: Umbral.
            
        Returns:
            Resultado de evaluación.
        """
        score, detail = self.evaluate_concept_coverage(
            aligned_stories, aspects, threshold
        )
        
        return {
            "aspect_coverage_score": score,
            "aspects_covered": detail["covered"],
            "aspects_total": detail["total"],
            "threshold": detail["threshold"],
            "details": detail["details"]
        }
    
    def evaluate_stakeholder_coverage(
        self,
        aligned_stories: List[str],
        stakeholders: List[str],
        threshold: float = None
    ) -> Dict:
        """
        Evalúa cobertura de stakeholders/actores.
        
        Args:
            aligned_stories: Historias alineadas.
            stakeholders: Lista de stakeholders esperados.
            threshold: Umbral.
            
        Returns:
            Resultado de evaluación.
        """
        threshold = threshold or self.threshold
        
        if not stakeholders:
            return {
                "stakeholder_coverage_score": 1.0,
                "covered": 0,
                "total": 0,
                "details": []
            }
        
        if not aligned_stories:
            return {
                "stakeholder_coverage_score": 0.0,
                "covered": 0,
                "total": len(stakeholders),
                "details": [
                    {"stakeholder": s, "covered": False, "max_similarity": 0.0}
                    for s in stakeholders
                ]
            }
        
        # Convertir a formato de conceptos
        stakeholder_concepts = {s: [s] for s in stakeholders}
        
        score, detail = self.evaluate_concept_coverage(
            aligned_stories, stakeholder_concepts, threshold
        )
        
        return {
            "stakeholder_coverage_score": score,
            "covered": detail["covered"],
            "total": detail["total"],
            "threshold": threshold,
            "details": detail["details"]
        }
    
    def get_config(self) -> dict:
        return {
            "threshold": self.threshold
        }
