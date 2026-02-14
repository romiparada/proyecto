from sentence_transformers import util
from typing import List, Dict, Tuple
import numpy as np

# Umbral para considerar un concepto cubierto
CONCEPT_COVERAGE_THRESHOLD = 0.70

#esta clase deberia evaluar la proporcion de conceptos del dominiio que estan representados en las historias generadas
class EvaluadorCoberturaConceptos:
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
        #para cada concepto, se verifica si alguna historia alineada lo cubre
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
        
        #codifico historias alineadas
        emb_stories = self.encoder.encode_stories(aligned_stories)
        if emb_stories.dim() == 1:
            emb_stories = emb_stories.unsqueeze(0)
        
        covered = 0
        details = []
        
        for concept_name, descriptions in concepts.items():
            #codifico conceptos del dominio
            emb_concept = self.encoder.encode(descriptions, convert_to_tensor=True)

            if emb_concept.dim() == 1:
                emb_concept = emb_concept.unsqueeze(0)
            
            #busco maxima similitud entre descripciones del concepto y las historias alineadas
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
