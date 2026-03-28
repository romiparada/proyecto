"""Configuración centralizada del pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # Modelo SBERT
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = None  # auto: cuda si disponible, si no cpu

    # Dimensión 1 - Cobertura Funcional
    coverage_threshold_covered: float = 0.75
    coverage_threshold_partial: float = 0.60

    # Dimensiones 2 & 3 - Umbrales de similitud/alucinación
    hallucination_threshold_aligned: float = 0.75
    hallucination_threshold_uncertain: float = 0.60

    # Umbrales de matching de Criterios de Aceptación
    criteria_threshold_matching: float = 0.75
    criteria_threshold_low: float = 0.50

    # Umbral de similitud de historia para entrar en alineación CA
    ca_alignment_story_threshold: float = 0.60

    # Búsqueda semántica top-k
    top_k: int = 3

    # Umbral mínimo para incluir aristas en el grafo semántico
    graph_edge_threshold: float = 0.60
