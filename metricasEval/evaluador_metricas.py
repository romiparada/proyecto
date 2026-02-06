import os
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import torch
import numpy as np
import itertools
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

LANG = "en"
BERT_MODEL = "microsoft/deberta-xlarge-mnli"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
SBERT_UMBRAL = 0.80
CONSISTENCIA_UMBRAL = 0.85
COMPLETITUD_UMBRAL = 0.60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(8)
torch.set_num_interop_threads(8)


class EvaluadorMetricas:
    
    def __init__(self, device=None):
        self.device = device if device is not None else DEVICE
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.device = torch.device("cuda:0")
        
        self.lang = LANG
        self.bert_model = BERT_MODEL
        self.sbert_model_name = SBERT_MODEL
        self.sbert_umbral = SBERT_UMBRAL
        self.consistencia_umbral = CONSISTENCIA_UMBRAL
        self.completitud_umbral = COMPLETITUD_UMBRAL
        
        
        self.model_sbert = SentenceTransformer(self.sbert_model_name, device=self.device)

    
    def calcular_sbert(self, texto, emb_refs, textos_refs):
        emb_texto = self.model_sbert.encode(texto, convert_to_tensor=True)
        sims = util.cos_sim(emb_texto, emb_refs)[0].cpu().numpy()
        idx = int(np.argmax(sims))
        return float(sims[idx]), idx, textos_refs[idx]
    
    def calcular_bertscore(self, texto, conjunto):
        cands = [texto]
        refs = conjunto
        
        try:
            P, R, F1 = bert_score(
                cands * len(refs),
                refs,
                lang=self.lang,
                model_type=self.bert_model,
                device=self.device,
                verbose=False,
                rescale_with_baseline=True
            )
            
            f1_scores = F1.cpu().numpy()
            idx = int(np.argmax(f1_scores))
            return float(f1_scores[idx]), idx, conjunto[idx]
        except (OverflowError, RuntimeError) as e:
            return 0.0, -1, ""
        

    def calcular_coverage(self, historias_generadas, historias_esperadas, threshold=0.75):
        emb_gen = self.model_sbert.encode(historias_generadas, convert_to_tensor=True)
        emb_esperadas = self.model_sbert.encode(historias_esperadas, convert_to_tensor=True)

        covered = 0

        for emb in emb_esperadas:
            sims = util.cos_sim(emb, emb_gen).max().item()
            if sims >= threshold:
                covered += 1

        return covered / len(historias_esperadas)
   
    # la idea es comparar n conjuntos de historias generadas por distintos llm
    # objetivo: evaluar la diversidad entre los distintos llm's / modelos usados
    # dsp de tener la diversidad entre cada par de conjuntos, se hace un promedio general (paper 2507.15157)
    def calcular_diversidad(self, set_gen_1, set_gen_2, threshold=0.75):

        # Diversity(X,Y) = 100 - Coverage(X,Y) (percentage)
        coverage = self.calcular_coverage(
            set_gen_1,
            set_gen_2,
            threshold=threshold
        )

        diversidad_pct = 100.0 * (1.0 - coverage)
        return diversidad_pct
    
    def evaluar_diversidad(self, conjuntos_historias, threshold=0.75):
        #conjuntos_historias seria [set_historias_generadas1, set_historias_generadas2, ...]
        #cada set de hist generadas corresponde a un output de cada herramienta / modelo
        n = len(conjuntos_historias)
        if n < 2:
            return {
                "diversidad_media": None,
                "diversidad_std": None,
                "num_comparaciones": 0,
                "pairwise": []
            }

        pairwise = []
        for i, j in itertools.combinations(range(n), 2):
            div_ij = self.calcular_diversidad(
                conjuntos_historias[i],
                conjuntos_historias[j],
                threshold=threshold
            )
            pairwise.append({
                "i": i,
                "j": j,
                "diversidad_pct": float(div_ij)
            })

        diversidad_vals = [p["diversidad_pct"] for p in pairwise]
        diversidad_media = float(np.mean(diversidad_vals))
        diversidad_std = float(np.std(diversidad_vals))

        return {
            "diversidad_media": diversidad_media,
            "diversidad_std": diversidad_std,
            "num_comparaciones": len(pairwise),
            "pairwise": pairwise
        }

    
    def evaluar_alineacion(self, historias_generadas, historias_esperadas):
        emb_esperadas = self.model_sbert.encode(
            historias_esperadas,
            convert_to_tensor=True,
            batch_size=16,
            show_progress_bar=False
        )
        
        resultados = {
            "por_historia": [],
            "scores_sbert": [],
            "scores_bertscore": [],
        }
        
        for i, gen in enumerate(historias_generadas):
            sbert_sim, sbert_idx, sbert_match = self.calcular_sbert(gen, emb_esperadas, historias_esperadas)
            bert_f1, bert_idx, bert_match = self.calcular_bertscore(gen, historias_esperadas)

            nivel = (
                "alta" if sbert_sim >= self.sbert_umbral
                else "media" if sbert_sim >= 0.60
                else "baja"
            )
            
            resultado_historia = {
                "indice": i,
                "generada": gen,
                "sbert": {
                    "similitud": sbert_sim,
                    "indice_match": sbert_idx,
                    "texto_match": sbert_match,
                    "nivel": nivel
                },
                "bertscore": {
                    "f1": bert_f1,
                    "indice_match": bert_idx,
                    "texto_match": bert_match
                }
            }
            
            resultados["por_historia"].append(resultado_historia)
            resultados["scores_sbert"].append(sbert_sim)
            resultados["scores_bertscore"].append(bert_f1)
        
        resultados["agregado"] = {
            "sbert": {
                "media": float(np.mean(resultados["scores_sbert"])),
                "std": float(np.std(resultados["scores_sbert"])),
                "alineacion_fuerte_pct": float(
                    sum(1 for s in resultados["scores_sbert"] if s >= self.sbert_umbral) 
                    / len(resultados["scores_sbert"]) * 100
                )
            },
            "bertscore": {
                "media": float(np.mean(resultados["scores_bertscore"])),
                "std": float(np.std(resultados["scores_bertscore"]))
            }
        }
        
        return resultados
    
    def obtener_config(self):
        return {
            "lang": self.lang,
            "bert_model": self.bert_model,
            "sbert_model": self.sbert_model_name,
            "sbert_umbral": self.sbert_umbral,
            "consistencia_umbral": self.consistencia_umbral,
            "completitud_umbral": self.completitud_umbral,
            "device": str(self.device),
            "cuda_disponible": torch.cuda.is_available()
        }

