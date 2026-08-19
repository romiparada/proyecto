"""Wrapper SBERT: carga del modelo y encoding de textos."""

import os
import sys
import warnings
import logging

# Silenciar librerías ruidosas antes de importar torch/transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
for _lib in ("transformers", "huggingface_hub", "torch", "sentence_transformers"):
    logging.getLogger(_lib).setLevel(logging.ERROR)

# Fix de DLLs en Windows con numpy/scipy
if sys.platform == "win32":
    for _pkg in ("numpy.libs", "scipy.libs"):
        _dll = os.path.join(sys.prefix, "Lib", "site-packages", _pkg)
        if os.path.exists(_dll):
            os.add_dll_directory(_dll)

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union

# Modelo por defecto; cambiar aquí para probar otros
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
# SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# SBERT_MODEL = "intfloat/e5-base-v2"


class SemanticEncoder:
    def __init__(self, model_name: str = SBERT_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.device = torch.device("cuda:0")
        torch.set_num_threads(8)
        torch.set_num_interop_threads(8)
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = True,
        batch_size: int = 16,
        show_progress_bar: bool = False,
    ) -> torch.Tensor:
        return self.model.encode(
            texts,
            convert_to_tensor=convert_to_tensor,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )




def create_encoder(model_name: str, device: str = None) -> SemanticEncoder:
    print("Cargando modelo SBERT...")
    encoder = SemanticEncoder(model_name=model_name, device=device)
    print(f"  Dispositivo : {encoder.device}")
    print(f"  Modelo      : {encoder.model_name}")
    return encoder


"""Utilidades de similitud coseno sobre tensores pre-computados."""

import torch
from sentence_transformers import util
from typing import List, Dict


def compute_similarity_matrix(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Matriz de similitud coseno (|emb_a| × |emb_b|)."""
    return util.cos_sim(emb_a, emb_b)


def top_k_matches(
    sim_matrix: torch.Tensor,
    row_idx: int,
    texts_b: List[str],
    k: int = 3,
) -> List[Dict]:
    """Top-k items más similares de texts_b para la fila row_idx."""
    sims = sim_matrix[row_idx]
    actual_k = min(k, len(texts_b))
    top_vals, top_idxs = torch.topk(sims, k=actual_k)

    return [
        {
            "rank": rank + 1,
            "index": int(idx),
            "text": texts_b[int(idx)],
            "similarity": round(float(score), 4),
        }
        for rank, (idx, score) in enumerate(
            zip(top_idxs.cpu().numpy(), top_vals.cpu().numpy())
        )
    ]
