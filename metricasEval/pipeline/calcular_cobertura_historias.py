# Coverage(X,Y) = expected_stories_with_match / total_expected_stories (paper)
# Diversity(X,Y) = 100 - Coverage(X,Y) (paper)

import numpy as np
import itertools
from sentence_transformers import util
from typing import List, Dict, Tuple

# Umbral para considerar que una historia esta cubierta
COVERAGE_THRESHOLD = 0.75


class CalculadorCoberturaHistorias:
    def __init__(self, encoder, threshold: float = COVERAGE_THRESHOLD):
        self.encoder = encoder
        self.threshold = threshold
    
    def calculate_coverage(
        self,
        stories_generated: List[str],
        stories_expected: List[str],
        threshold: float = None
    ) -> Tuple[float, Dict]:
        threshold = threshold or self.threshold
        
        if not stories_expected:
            return 1.0, {"covered": 0, "total": 0, "details": []}
        
        if not stories_generated:
            return 0.0, {"covered": 0, "total": len(stories_expected), "details": []}
        
        # Codificar las listas
        emb_gen = self.encoder.encode_stories(stories_generated)
        emb_exp = self.encoder.encode_stories(stories_expected)
        
        covered = 0
        details = []
        
        for i, emb_e in enumerate(emb_exp):
            # Máxima similitud con cualquier historia generada
            sims = util.cos_sim(emb_e, emb_gen)[0]
            max_sim = float(sims.max().item())
            is_covered = max_sim >= threshold
            
            if is_covered:
                covered += 1
                best_idx = int(sims.argmax().item())
            else:
                best_idx = -1
            
            details.append({
                "expected_idx": i,
                "expected_text": stories_expected[i],
                "max_similarity": max_sim,
                "covered": is_covered,
                "best_match_idx": best_idx,
                "best_match_text": stories_generated[best_idx] if is_covered else None
            })
        
        score = covered / len(stories_expected)
        
        return score, {
            "covered": covered,
            "total": len(stories_expected),
            "threshold": threshold,
            "details": details
        }
    
    def calculate_diversity(
        self,
        stories_set_1: List[str],
        stories_set_2: List[str],
        threshold: float = None
    ) -> float:
        coverage, _ = self.calculate_coverage(
            stories_set_1,
            stories_set_2,
            threshold=threshold
        )
        return 100.0 * (1.0 - coverage)
    
    def evaluate_multi_model_diversity(
        self,
        model_story_sets: List[List[str]],
        model_names: List[str] = None,
        threshold: float = None
    ) -> Dict:
        n = len(model_story_sets)
        
        if n < 2:
            return {
                "diversity_mean": None,
                "diversity_std": None,
                "num_comparisons": 0,
                "pairwise": [],
                "models": model_names or []
            }
        
        model_names = model_names or [f"model_{i}" for i in range(n)]
        
        pairwise = []
        for i, j in itertools.combinations(range(n), 2):
            div_ij = self.calculate_diversity(
                model_story_sets[i],
                model_story_sets[j],
                threshold=threshold
            )
            pairwise.append({
                "model_i": model_names[i],
                "model_j": model_names[j],
                "index_i": i,
                "index_j": j,
                "diversity_pct": float(div_ij)
            })
        
        diversity_vals = [p["diversity_pct"] for p in pairwise]
        
        return {
            "diversity_mean": float(np.mean(diversity_vals)),
            "diversity_std": float(np.std(diversity_vals)),
            "num_comparisons": len(pairwise),
            "pairwise": pairwise,
            "models": model_names,
            "threshold": threshold or self.threshold
        }
    
    def get_config(self) -> dict:
        return {
            "threshold": self.threshold
        }
