"""
LEVEL 1 — Evaluación de Historias de Usuario (HU)

Responsabilidades:
- Matching SBERT (cosine similarity) contra HU_expected
- Selección del mejor match (argmax)
- BERTScore F1 como métrica secundaria (NO decide alineación)
- Clasificación: Strong ≥ 0.80, Conservative ≥ 0.85, Weak < 0.80
- Filtrado: Solo historias alineadas continúan al LEVEL 3
"""

import os
import sys
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# FIX Windows DLL loading
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


# Umbrales según metodología experimental
UMBRAL_STRONG = 0.80      # Alineación fuerte
UMBRAL_CONSERVATIVE = 0.85 # Alineación conservadora (subconjunto de strong)

BERT_MODEL = "microsoft/deberta-xlarge-mnli"
LANG = "en"


@dataclass
class AlignmentResult:
    """Resultado de alineación para una historia generada."""
    index_generated: int
    text_generated: str
    index_matched: int
    text_matched: str
    sbert_similarity: float
    bertscore_f1: Optional[float]
    alignment_level: str  # "strong", "conservative", "weak"
    
    @property
    def is_aligned(self) -> bool:
        """Historia está alineada si es strong o conservative."""
        return self.alignment_level in ("strong", "conservative")


class StoryAlignmentEvaluator:
    """
    LEVEL 1: Evaluador de alineación de HU.
    
    Evalúa qué tan bien las historias generadas se alinean
    con las historias esperadas usando SBERT.
    
    BERTScore se calcula como métrica informativa pero
    NO participa en la decisión de alineación.
    """
    
    def __init__(
        self,
        encoder,
        umbral_strong: float = UMBRAL_STRONG,
        umbral_conservative: float = UMBRAL_CONSERVATIVE,
        bert_model: str = BERT_MODEL,
        lang: str = LANG
    ):
        self.encoder = encoder
        self.umbral_strong = umbral_strong
        self.umbral_conservative = umbral_conservative
        self.bert_model = bert_model
        self.lang = lang
    
    def _calculate_sbert_match(
        self,
        text: str,
        embeddings_ref,
        texts_ref: List[str]
    ) -> Tuple[float, int, str]:
        """
        Calcula similitud SBERT y encuentra mejor match.
        
        Returns:
            (similitud, índice_match, texto_match)
        """
        emb_text = self.encoder.encode(text, convert_to_tensor=True)
        similarities = util.cos_sim(emb_text, embeddings_ref)[0]
        
        idx_best = int(np.argmax(similarities.cpu().numpy()))
        score = float(similarities[idx_best].item())
        
        return score, idx_best, texts_ref[idx_best]
    
    def _calculate_bertscore(
        self,
        text: str,
        matched_ref: str
    ) -> Optional[float]:
        """
        Calcula BERTScore F1 (métrica secundaria/informativa).
        
        Returns:
            f1_score contra el match SBERT.
        """
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
        """
        Clasifica nivel de alineación basado en SBERT.
        
        - conservative: ≥ 0.85 (subconjunto más estricto)
        - strong: ≥ 0.80
        - weak: < 0.80
        """
        if sbert_similarity >= self.umbral_conservative:
            return "conservative"
        elif sbert_similarity >= self.umbral_strong:
            return "strong"
        else:
            return "weak"
    
    def evaluate(
        self,
        stories_generated: List[str],
        stories_expected: List[str]
    ) -> Dict:
        """
        Evalúa alineación de historias generadas contra esperadas.
        
        Args:
            stories_generated: Lista de HU generadas.
            stories_expected: Lista de HU esperadas (referencia).
            
        Returns:
            Diccionario con resultados de alineación.
        """
        # Pre-codificar historias esperadas (LEVEL 0)
        emb_expected = self.encoder.encode_stories(stories_expected)
        
        alignments: List[AlignmentResult] = []
        scores_sbert = []
        scores_bert = []
        
        for i, gen_story in enumerate(stories_generated):
            # SBERT matching (decisión principal)
            sbert_sim, sbert_idx, sbert_match = self._calculate_sbert_match(
                gen_story, emb_expected, stories_expected
            )
            
            # BERTScore (informativo, NO decide)
            bert_f1 = self._calculate_bertscore(gen_story, sbert_match)
            
            # Clasificación basada SOLO en SBERT
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
        
        # Separar historias alineadas (para LEVEL 3)
        aligned_stories = [a for a in alignments if a.is_aligned]
        
        # Estadísticas agregadas
        return {
            "alignments": alignments,
            "aligned_stories": aligned_stories,
            "aggregate": {
                "total_generated": len(stories_generated),
                "total_expected": len(stories_expected),
                "total_aligned": len(aligned_stories),
                "alignment_rate": len(aligned_stories) / len(stories_generated) if stories_generated else 0,
                "sbert": {
                    "mean": float(np.mean(scores_sbert)) if scores_sbert else 0,
                    "std": float(np.std(scores_sbert)) if scores_sbert else 0,
                    "aligned_count": sum(1 for a in alignments if a.alignment_level in ("strong", "conservative")),
                    "strong_count": sum(1 for a in alignments if a.alignment_level == "strong"),
                    "conservative_count": sum(1 for a in alignments if a.alignment_level == "conservative"),
                    "weak_count": sum(1 for a in alignments if a.alignment_level == "weak"),
                },
                "bertscore": {
                    "mean": float(np.mean(scores_bert)) if scores_bert else None,
                    "std": float(np.std(scores_bert)) if scores_bert else None,
                    "available_count": len(scores_bert),
                }
            },
            "config": {
                "umbral_strong": self.umbral_strong,
                "umbral_conservative": self.umbral_conservative,
            }
        }
    
    def get_aligned_pairs(self, alignment_results: Dict) -> List[Tuple[int, int]]:
        """
        Extrae pares (idx_generada, idx_esperada) de historias alineadas.
        
        Útil para pasar al LEVEL 3 (evaluación de CA).
        """
        return [
            (a.index_generated, a.index_matched)
            for a in alignment_results["aligned_stories"]
        ]
