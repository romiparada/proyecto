"""
LEVEL 0 — Representación Semántica

Responsabilidad única: Codificar textos usando SBERT.
Esta fase NO evalúa nada, solo prepara embeddings.
"""

import os
import sys
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

# FIX Python 3.8+ DLL loading (Windows)
if sys.platform == "win32":
    dll_path = os.path.join(sys.prefix, "Lib", "site-packages", "numpy.libs")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
    dll_path2 = os.path.join(sys.prefix, "Lib", "site-packages", "scipy.libs")
    if os.path.exists(dll_path2):
        os.add_dll_directory(dll_path2)

import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

# Modelo SBERT recomendado para tareas de similaridad semántica
# https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"


class SemanticEncoder:
    """
    LEVEL 0: Codificador semántico basado en SBERT.
    
    Responsabilidad: Generar embeddings para textos.
    NO realiza evaluación ni comparación.
    """

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
        show_progress_bar: bool = False
    ):
        """
        Codifica uno o más textos en embeddings.
        
        Args:
            texts: Texto o lista de textos a codificar.
            convert_to_tensor: Si True, retorna tensor PyTorch.
            batch_size: Tamaño de batch para codificación.
            show_progress_bar: Mostrar barra de progreso.
            
        Returns:
            Tensor o array numpy de embeddings.
        """
        return self.model.encode(
            texts,
            convert_to_tensor=convert_to_tensor,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )
    
    def encode_stories(self, stories: List[str]):
        """Codifica lista de historias de usuario."""
        return self.encode(stories, convert_to_tensor=True)
    
    def encode_acceptance_criteria(self, criteria: List[str]):
        """Codifica lista de criterios de aceptación."""
        return self.encode(criteria, convert_to_tensor=True)
    
    def encode_concepts(self, concepts: List[str]):
        """Codifica lista de conceptos del dominio."""
        return self.encode(concepts, convert_to_tensor=True)
    
    def get_config(self) -> dict:
        """Retorna configuración del encoder."""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available()
        }
