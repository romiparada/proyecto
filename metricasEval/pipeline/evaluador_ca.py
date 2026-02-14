#solo se evaluan los criterios de aceptación de las historias alineadas en nivel 1
import os
import sys
import re
from sentence_transformers import util
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

if sys.platform == "win32":
    dll_path = os.path.join(sys.prefix, "Lib", "site-packages", "numpy.libs")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
    dll_path2 = os.path.join(sys.prefix, "Lib", "site-packages", "scipy.libs")
    if os.path.exists(dll_path2):
        os.add_dll_directory(dll_path2)


SCENARIO_COVERAGE_THRESHOLD = 0.75

AMBIGUOUS_TERMS = [
    "fast", "quick", "adequate", "appropriate",
    "correct", "proper", "user-friendly",
    "efficient", "intuitive",
    "as soon as possible", "reasonable time",
    "easily", "simple", "good", "bad",
    "optimal", "best", "worst"
]

#CONDITION_PATTERNS = [
#    r"\bif\b",
#    r"\bwhen\b",
#    r"\bgiven\b",
#    r"\bprovided\b",
#    r"\bassuming\b"
#]

# Patrones que indican resultado observable (Then/Shall/Must)
RESULT_PATTERNS = [
    r"\bshall\b",
    r"\bmust\b",
    r"\bshould\b",
    r"\bthen\b",
    r"\bdisplays?\b",
    r"\breturns?\b",
    r"\bshows?\b",
    r"\bcreates?\b",
    r"\bgenerates?\b",
    r"\bupdates?\b",
    r"\bstores?\b",
    r"\bsaves?\b",
    r"\bcalculates?\b",
    r"\bvalidates?\b",
    r"\bnotifies?\b",
    r"\bprevents?\b",
    r"\ballows?\b"
]


@dataclass
class CriteriaEvaluationResult:
    story_generated_idx: int
    story_expected_idx: int
    cobertura_funcional: float
    cobertura_funcional_detalles: List[Dict]
    puntuacion_ambiguedad: float
    detalles_ambiguedad: List[Dict]
    puntuacion_verificabilidad: float
    detalles_verificabilidad: List[Dict]


class EvaluadorCriteriosAceptacion:
    def __init__(
        self,
        encoder,
        coverage_threshold: float = SCENARIO_COVERAGE_THRESHOLD,
        ambiguous_terms: List[str] = None,
        condition_patterns: List[str] = None,
        result_patterns: List[str] = None
    ):
        self.encoder = encoder
        self.coverage_threshold = coverage_threshold
        self.ambiguous_terms = ambiguous_terms or AMBIGUOUS_TERMS
        #self.condition_patterns = condition_patterns or CONDITION_PATTERNS
        self.result_patterns = result_patterns or RESULT_PATTERNS
    
    def evaluate_functional_coverage(
        self,
        ca_generated: List[str],
        ca_expected: List[str]
    ) -> Tuple[float, List[Dict]]:

        if not ca_expected:
            return 1.0, []
        
        if not ca_generated:
            return 0.0, [
                {"expected": ca, "covered": False, "max_similarity": 0.0}
                for ca in ca_expected
            ]
        
        emb_gen = self.encoder.encode(ca_generated, convert_to_tensor=True)
        emb_exp = self.encoder.encode(ca_expected, convert_to_tensor=True)
        
        covered = 0
        details = []
        
        for i, emb_e in enumerate(emb_exp):
            #veo la similitud maxima con cada CA generada
            sims = util.cos_sim(emb_e, emb_gen)[0]
            max_sim = float(sims.max().item())
            is_covered = max_sim >= self.coverage_threshold
            
            if is_covered:
                covered += 1
                best_idx = int(sims.argmax().item())
                best_match = ca_generated[best_idx]
            else:
                best_match = None
            
            details.append({
                "expected_ca": ca_expected[i],
                "max_similarity": max_sim,
                "covered": is_covered,
                "best_match": best_match
            })
        
        score = covered / len(ca_expected)
        return score, details
    
    def evaluate_ambiguity(
        self,
        ca_list: List[str]
    ) -> Tuple[float, List[Dict]]:
        if not ca_list:
            return 1.0, []
        
        ambiguous_count = 0
        details = []
        
        for ca in ca_list:
            text_lower = ca.lower()
            found_terms = [
                term for term in self.ambiguous_terms
                if term in text_lower
            ]
            
            is_ambiguous = len(found_terms) > 0
            if is_ambiguous:
                ambiguous_count += 1
            
            details.append({
                "criteria": ca,
                "is_ambiguous": is_ambiguous,
                "ambiguous_terms_found": found_terms
            })
        
        # Score alto implica menos ambigüedad
        score = 1 - (ambiguous_count / len(ca_list))
        return score, details
    
    def evaluate_verifiability(
        self,
        ca_list: List[str]
    ) -> Tuple[float, List[Dict]]:
        if not ca_list:
            return 0.0, []
        
        verifiable_count = 0
        details = []
        
        for ca in ca_list:
            text_lower = ca.lower()
            
            #has_condition = any(
            #    re.search(pattern, text_lower)
            #    for pattern in self.condition_patterns
            #)
            
            # Uso de result patterns para detectar resultado observable
            has_result = any(
                re.search(pattern, text_lower)
                for pattern in self.result_patterns
            )
            
            # ambiguedad
            ambiguous_found = [
                term for term in self.ambiguous_terms
                if term in text_lower
            ]
            is_ambiguous = len(ambiguous_found) > 0
            
            # Verificable = condición + resultado + no ambiguo
            #is_verifiable = has_condition and has_result and not is_ambiguous
            is_verifiable =  has_result and not is_ambiguous
            
            if is_verifiable:
                verifiable_count += 1
            
            details.append({
                "criteria": ca,
                #"has_condition": has_condition,
                "has_observable_result": has_result,
                "ambiguous_terms": ambiguous_found,
                "is_verifiable": is_verifiable
            })
        
        score = verifiable_count / len(ca_list)
        return score, details
    
    def evaluate_pair(
        self,
        ca_generated: List[str],
        ca_expected: List[str],
        story_gen_idx: int,
        story_exp_idx: int
    ) -> CriteriaEvaluationResult:
        # Cobertura funcional
        cov_score, cov_details = self.evaluate_functional_coverage(
            ca_generated, ca_expected
        )
        
        # Ambigüedad
        amb_score, amb_details = self.evaluate_ambiguity(ca_generated)
        
        ver_score, ver_details = self.evaluate_verifiability(ca_generated)
        
        return CriteriaEvaluationResult(
            story_generated_idx=story_gen_idx,
            story_expected_idx=story_exp_idx,
            cobertura_funcional=cov_score,
            cobertura_funcional_detalles=cov_details,
            puntuacion_ambiguedad=amb_score,
            detalles_ambiguedad=amb_details,
            puntuacion_verificabilidad=ver_score,
            detalles_verificabilidad=ver_details
        )
    
    def evaluate_aligned_stories(
        self,
        aligned_pairs: List[Tuple[int, int]],
        all_ca_generated: List[List[str]],
        all_ca_expected: List[List[str]]
    ) -> Dict:

        results = []
        
        for idx_gen, idx_exp in aligned_pairs:
            ca_gen = all_ca_generated[idx_gen] if idx_gen < len(all_ca_generated) else []
            ca_exp = all_ca_expected[idx_exp] if idx_exp < len(all_ca_expected) else []
            
            pair_result = self.evaluate_pair(
                ca_gen, ca_exp, idx_gen, idx_exp
            )
            results.append(pair_result)
        
        if results:
            avg_coverage = sum(r.cobertura_funcional for r in results) / len(results)
            avg_ambiguity = sum(r.puntuacion_ambiguedad for r in results) / len(results)
            avg_verifiability = sum(r.puntuacion_verificabilidad for r in results) / len(results)
        else:
            avg_coverage = avg_ambiguity = avg_verifiability = 0.0
        
        all_gen_ca_flat = []
        for idx_gen, _ in aligned_pairs:
            if idx_gen < len(all_ca_generated):
                all_gen_ca_flat.extend(all_ca_generated[idx_gen])
        
        global_ver_score, global_ver_details = self.evaluate_verifiability(all_gen_ca_flat)
        global_amb_score, global_amb_details = self.evaluate_ambiguity(all_gen_ca_flat)
        
        return {
            "pair_results": [
                {
                    "story_generated_idx": r.story_generated_idx,
                    "story_expected_idx": r.story_expected_idx,
                    "functional_coverage": r.cobertura_funcional,
                    "functional_coverage_details": r.cobertura_funcional_detalles,
                    "ambiguity_score": r.puntuacion_ambiguedad,
                    "ambiguity_details": r.detalles_ambiguedad,
                    "verifiability_score": r.puntuacion_verificabilidad,
                    "verifiability_details": r.detalles_verificabilidad
                }
                for r in results
            ],
            "aggregate": {
                "num_aligned_pairs": len(results),
                "avg_functional_coverage": avg_coverage,
                "avg_ambiguity_score": avg_ambiguity,
                "avg_verifiability_score": avg_verifiability,
            },
            "global_metrics": {
                "total_criteria_evaluated": len(all_gen_ca_flat),
                "verifiability": {
                    "score": global_ver_score,
                    "details": global_ver_details
                },
                "ambiguity": {
                    "score": global_amb_score,
                    "details": global_amb_details
                }
            },
            "config": {
                "coverage_threshold": self.coverage_threshold
            }
        }
    
    def get_config(self) -> dict:
        return {
            "coverage_threshold": self.coverage_threshold,
            "num_ambiguous_terms": len(self.ambiguous_terms),
            #"num_condition_patterns": len(self.condition_patterns),
            "num_result_patterns": len(self.result_patterns)
        }
