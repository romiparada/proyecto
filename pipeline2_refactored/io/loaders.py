"""Parsers de entradas del pipeline (historias, criterios de aceptación, aspectos)."""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Union


def load_stories(path: Union[str, Path]) -> List[str]:
    """Carga historias de usuario desde un .txt (una por línea no vacía)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        stories = [line.strip() for line in f if line.strip()]
    if not stories:
        raise ValueError(f"{path.name}: no stories found (file is empty or all blank lines).")
    return stories


def load_criteria(path: Union[str, Path]) -> List[List[str]]:
    """Carga criterios de aceptación desde JSON (lista de listas de strings).

    Las entradas con texto placeholder se normalizan a listas vacías.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: expected a JSON array at the top level.")

    criteria: List[List[str]] = []
    placeholder_phrases = {"no acceptance criteria defined", "n/a", "none", ""}

    for i, entry in enumerate(raw):
        if not isinstance(entry, list):
            raise ValueError(f"{path.name}: la entrada en índice {i} debe ser una lista de strings.")
        # Filtrar entradas placeholder
        cleaned = [
            c.strip() for c in entry
            if c.strip().lower() not in placeholder_phrases
        ]
        criteria.append(cleaned)

    return criteria


def load_aspects(path: Union[str, Path]) -> List[Tuple[str, str]]:
    """Carga funcionalidades del dominio desde JSON.

    Soporta dos formatos:
      - dict: {"categoria": ["aspecto1", ...], ...}
      - list: ["aspecto1", "aspecto2", ...]

    Retorna lista plana de tuplas (categoria, texto).
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        result = []
        for category, items in raw.items():
            if isinstance(items, list):
                for item in items:
                    result.append((category, str(item).strip()))
            else:
                result.append((category, str(items).strip()))
        return result
    elif isinstance(raw, list):
        return [("general", str(item).strip()) for item in raw]
    else:
        raise ValueError(f"{path.name}: expected a JSON object or array.")


def align_criteria_to_stories(
    criteria: List[List[str]],
    stories: List[str],
    label: str = "criteria",
) -> List[List[str]]:
    """Ajusta el largo de la lista de criterios para que coincida con el de historias."""
    n_criteria = len(criteria)
    n_stories = len(stories)
    if n_criteria < n_stories:
        print(
            f"  ⚠ {label} has {n_criteria} entries but {n_stories} stories "
            f"— padding with empty lists."
        )
        criteria.extend([] for _ in range(n_stories - n_criteria))
    elif n_criteria > n_stories:
        print(
            f"  ⚠ {label} has {n_criteria} entries but only {n_stories} stories "
            f"— truncating to match."
        )
        criteria = criteria[:n_stories]
    return criteria
