import os
import sys
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

if sys.platform == "win32":
    dll_path = os.path.join(sys.prefix, "Lib", "site-packages", "numpy.libs")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
    dll_path2 = os.path.join(sys.prefix, "Lib", "site-packages", "scipy.libs")
    if os.path.exists(dll_path2):
        os.add_dll_directory(dll_path2)

import numpy as np
from sentence_transformers import util
from bert_score import score as bert_score
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


UMBRAL_STRONG = 0.80      # Alineación fuerte
UMBRAL_CONSERVATIVE = 0.85 # Alineación conservadora

BERT_MODEL = "microsoft/deberta-xlarge-mnli"
LANG = "en"


@dataclass
class AlignmentResult:
    index_generated: int
    text_generated: str
    index_matched: int
    text_matched: str
    sbert_similarity: float
    bertscore_f1: Optional[float]
    alignment_level: str  # "strong", "conservative", "weak"
    
    @property
    def is_aligned(self) -> bool:
        return self.alignment_level in ("strong", "conservative")


class EvaluadorAlineacionHistorias:
    #evaluamos alineacion con sbert
    
    def __init__(
        self,
        encoder,
        umbral_fuerte: float = UMBRAL_STRONG,
        umbral_conservador: float = UMBRAL_CONSERVATIVE,
        bert_model: str = BERT_MODEL,
        lang: str = LANG
    ):
        self.encoder = encoder
        self.umbral_fuerte = umbral_fuerte
        self.umbral_conservador = umbral_conservador
        self.bert_model = bert_model
        self.lang = lang
    

    
    def _calculate_bertscore(
        self,
        text: str,
        matched_ref: str
    ) -> Optional[float]:
        try:
            cands = [text]
            refs = [matched_ref]
            P, R, F1 = bert_score(
                cands,
                refs,
                lang=self.lang,
                model_type=self.bert_model,
                device=self.encoder.device,
                verbose=False,
                rescale_with_baseline=True
            )
            return float(F1.cpu().numpy()[0])
            
        except (OverflowError, RuntimeError):
            return None
    
    def _classify_alignment(self, sbert_similarity: float) -> str:
        # conservative: >= 0.85
        # strong: >= 0.80 y < 0.85
        # weak: < 0.80
        if sbert_similarity >= self.umbral_conservador:
            return "conservative"
        elif sbert_similarity >= self.umbral_fuerte:
            return "strong"
        else:
            return "weak"
    
    def evaluate(
        self,
        stories_generated: List[str],
        stories_expected: List[str]
    ) -> Dict:
        emb_generated = self.encoder.encode_stories(stories_generated)
        emb_expected = self.encoder.encode_stories(stories_expected)
        
        # Calcular matriz de similitud completa de una vez
        similarity_matrix = util.cos_sim(emb_generated, emb_expected)
        
        alignments: List[AlignmentResult] = []
        scores_sbert = []
        scores_bert = []
        
        for i, gen_story in enumerate(stories_generated):
            #Obtener similitudes de la fila i (historia generada i vs todas las esperadas)
            similarities = similarity_matrix[i]
            
            #mejor match SBERT
            sbert_idx = int(similarities.argmax().item())
            sbert_sim = float(similarities[sbert_idx].item())
            sbert_match = stories_expected[sbert_idx]
            
            # BERTScore (informativo)
            bert_f1 = self._calculate_bertscore(gen_story, sbert_match)
            
            #alineacion basada en sbert
            level = self._classify_alignment(sbert_sim)
            
            alignment = AlignmentResult(
                index_generated=i,
                text_generated=gen_story,
                index_matched=sbert_idx,
                text_matched=sbert_match,
                sbert_similarity=sbert_sim,
                bertscore_f1=bert_f1,
                alignment_level=level
            )
            
            alignments.append(alignment)
            scores_sbert.append(sbert_sim)
            if bert_f1 is not None:
                scores_bert.append(bert_f1)
        
        aligned_stories = [a for a in alignments if a.is_aligned]
        
        scores_sbert_aligned = [a.sbert_similarity for a in aligned_stories]
        scores_bert_aligned = [a.bertscore_f1 for a in aligned_stories if a.bertscore_f1 is not None]
        
        return {
            "alignments": alignments,
            "aligned_stories": aligned_stories,
            "aggregate": {
                "total_generated": len(stories_generated),
                "total_expected": len(stories_expected),
                "total_aligned": len(aligned_stories),
                "alignment_rate": len(aligned_stories) / len(stories_generated) if stories_generated else 0,
                "sbert": {
                    "mean_aligned": float(np.mean(scores_sbert_aligned)) if scores_sbert_aligned else 0,
                    "std_aligned": float(np.std(scores_sbert_aligned)) if scores_sbert_aligned else 0,
                    "mean_all": float(np.mean(scores_sbert)) if scores_sbert else 0,
                    "std_all": float(np.std(scores_sbert)) if scores_sbert else 0,
                    "aligned_count": sum(1 for a in alignments if a.alignment_level in ("strong", "conservative")),
                    "strong_count": sum(1 for a in alignments if a.alignment_level == "strong"),
                    "conservative_count": sum(1 for a in alignments if a.alignment_level == "conservative"),
                    "weak_count": sum(1 for a in alignments if a.alignment_level == "weak"),
                },
                "bertscore": {
                    "mean_aligned": float(np.mean(scores_bert_aligned)) if scores_bert_aligned else None,
                    "std_aligned": float(np.std(scores_bert_aligned)) if scores_bert_aligned else None,
                    "mean_all": float(np.mean(scores_bert)) if scores_bert else None,
                    "std_all": float(np.std(scores_bert)) if scores_bert else None,
                    "available_count": len(scores_bert),
                }
            },
            "config": {
                "umbral_strong": self.umbral_fuerte,
                "umbral_conservative": self.umbral_conservador,
            }
        }
    
    def get_aligned_pairs(self, alignment_results: Dict) -> List[Tuple[int, int]]:
        return [
            (a.index_generated, a.index_matched)
            for a in alignment_results["aligned_stories"]
        ]
