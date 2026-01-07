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
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

LANG = "en"
BERT_MODEL = "roberta-large"
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
    
    def calcular_bleu(self, generada, referencias):
        scores = [
            sentence_bleu([ref.split()], generada.split())
            for ref in referencias
        ]
        idx = int(np.argmax(scores))
        return float(scores[idx]), idx, referencias[idx]
    
    def calcular_rouge_l(self, generada, referencias):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = [scorer.score(generada, ref)['rougeL'].fmeasure for ref in referencias]
        idx = int(np.argmax(scores))
        return float(scores[idx]), idx, referencias[idx]
    
    def calcular_completitud(self, historias, aspect_embeddings, umbral=None):
        if umbral is None:
            umbral = self.completitud_umbral
        
        cobertura = {aspect: False for aspect in aspect_embeddings}
        
        for historia in historias:
            emb_historia = self.model_sbert.encode(historia, convert_to_tensor=True)
            for aspect, emb_aspects in aspect_embeddings.items():
                max_sim = util.cos_sim(emb_historia, emb_aspects).max().item()
                if max_sim >= umbral:
                    cobertura[aspect] = True
        
        score = sum(cobertura.values()) / len(cobertura)
        return score, cobertura
    
    def calcular_consistencia(self, historias, umbral=None):
        if umbral is None:
            umbral = self.consistencia_umbral
        
        embeddings = self.model_sbert.encode(historias, convert_to_tensor=True)
        sims = util.cos_sim(embeddings, embeddings)
        
        n = len(historias)
        pares_solapados = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i][j] >= umbral:
                    pares_solapados.append((i, j, float(sims[i][j])))
        
        if not pares_solapados:
            return 1.0, []
        
        consistencia = 1 - (len(pares_solapados) / (n * (n - 1) / 2))
        return consistencia, pares_solapados
    
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
            "scores_bleu": [],
            "scores_rouge": []
        }
        
        for i, gen in enumerate(historias_generadas):
            sbert_sim, sbert_idx, sbert_match = self.calcular_sbert(gen, emb_esperadas, historias_esperadas)
            bert_f1, bert_idx, bert_match = self.calcular_bertscore(gen, historias_esperadas)
            bleu_score, bleu_idx, bleu_match = self.calcular_bleu(gen, historias_esperadas)
            rouge_score, rouge_idx, rouge_match = self.calcular_rouge_l(gen, historias_esperadas)
            
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
                },
                "bleu": {
                    "score": bleu_score,
                    "indice_match": bleu_idx,
                    "texto_match": bleu_match
                },
                "rouge_l": {
                    "score": rouge_score,
                    "indice_match": rouge_idx,
                    "texto_match": rouge_match
                }
            }
            
            resultados["por_historia"].append(resultado_historia)
            resultados["scores_sbert"].append(sbert_sim)
            resultados["scores_bertscore"].append(bert_f1)
            resultados["scores_bleu"].append(bleu_score)
            resultados["scores_rouge"].append(rouge_score)
        
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
            },
            "bleu": {
                "media": float(np.mean(resultados["scores_bleu"])),
                "std": float(np.std(resultados["scores_bleu"]))
            },
            "rouge_l": {
                "media": float(np.mean(resultados["scores_rouge"])),
                "std": float(np.std(resultados["scores_rouge"]))
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
