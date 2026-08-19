
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union


def load_stories(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        stories = [line.strip() for line in f if line.strip()]
    if not stories:
        raise ValueError(f"{path.name}: no stories found (file is empty or all blank lines).")
    return stories


def load_criteria(path: Union[str, Path]) -> List[List[str]]:
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
        cleaned = [
            c.strip() for c in entry
            if c.strip().lower() not in placeholder_phrases
        ]
        criteria.append(cleaned)

    return criteria


def load_aspects(path: Union[str, Path]) -> List[Tuple[str, str]]:
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
    n_criteria = len(criteria)
    n_stories = len(stories)
    if n_criteria < n_stories:
        print(
            f"  ⚠ {label} has {n_criteria} entries but {n_stories} stories "
            f"— padding with empty lists."
        )
        criteria = criteria + [[] for _ in range(n_stories - n_criteria)]
    elif n_criteria > n_stories:
        print(
            f"  ⚠ {label} has {n_criteria} entries but only {n_stories} stories "
            f"— truncating to match."
        )
        criteria = criteria[:n_stories]
    return criteria
