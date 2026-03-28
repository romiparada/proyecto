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
