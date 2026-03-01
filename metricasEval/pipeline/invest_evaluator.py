"""
InvestEvaluator — LEVEL 5

Evalúa la calidad intrínseca de historias de usuario generadas
usando un LLM (gpt-4o) bajo el marco INVEST:
  I — Independiente
  N — Negociable
  V — Valiosa
  E — Estimable
  S — Pequeña (Small)
  T — Testeable

Basado en la metodología del paper:
"Evaluación de la calidad de historias de usuario usando modelos
de lenguaje de gran tamaño: un estudio en la industria"
(Hernández-Agüero, Quesada-López, Chaves-Sánchez)

A diferencia del paper, este evaluador no compara con expertos:
solo obtiene la calificación del LLM para cada historia generada.
"""

import os
import json
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    raise ImportError(
        "El paquete 'openai' no está instalado. "
        "Ejecutá: pip install openai>=1.0.0"
    )

try:
    from langdetect import detect as _langdetect_detect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


@dataclass
class InvestStoryResult:
    """Resultado de evaluación INVEST para una historia de usuario."""
    story_index: int
    story_text: str
    classification: str
    scores: Dict[str, int]          # {"I": 4, "N": 3, "V": 5, "E": 3, "S": 4, "T": 2}
    justifications: Dict[str, str]  # {"I": "...", "N": "...", ...}
    invest_mean: float              # promedio de los 6 criterios


@dataclass
class InvestEvaluationResult:
    """Resultado agregado de la evaluación INVEST de un conjunto de historias."""
    story_results: List[InvestStoryResult]
    aggregate: Dict
    language_detected: str
    model_used: str
    total_stories: int


# ─── Prompts ───────────────────────────────────────────────────────────────────

_PROFILE_ES = (
    "Eres un ingeniero de requerimientos experto en INVEST, con amplia experiencia "
    "en análisis de historias de usuario en entornos ágiles de desarrollo de software."
)

_PROFILE_EN = (
    "You are a requirements engineering expert proficient in INVEST, with extensive "
    "experience analyzing user stories in agile software development environments."
)

_CRITERIA_DESC_ES = {
    "I": "Independiente: es autónoma, comprensible por sí sola y no depende de otras historias.",
    "N": "Negociable: tiene un nivel de detalle adecuado que permite ajustes y priorización.",
    "V": "Valiosa: proporciona un beneficio claro y medible para el sistema o usuario final.",
    "E": "Estimable: es técnicamente alcanzable y se puede estimar en términos de tiempo o esfuerzo.",
    "S": "Pequeña: presenta un alcance acotado para completarse dentro de un sprint (máximo 4 semanas).",
    "T": "Testeable: incluye criterios de aceptación que permiten validar su implementación con pruebas.",
}

_CRITERIA_DESC_EN = {
    "I": "Independent: it is self-contained, understandable on its own, and does not depend on other stories.",
    "N": "Negotiable: it has an appropriate level of detail that allows for adjustments and prioritization.",
    "V": "Valuable: it provides a clear and measurable benefit for the system or end user.",
    "E": "Estimable: it is technically achievable and can be estimated in terms of time or effort.",
    "S": "Small: it has a limited scope to be completed within a sprint (maximum 4 weeks).",
    "T": "Testeable: it includes acceptance criteria that allow its implementation to be validated with tests.",
}

_LIKERT_SCALE_ES = (
    "Escala Likert 1-5: "
    "1 = Fuerte desacuerdo (no cumple el criterio), "
    "2 = Desacuerdo, "
    "3 = Neutral, "
    "4 = Acuerdo, "
    "5 = Fuerte acuerdo (cumple completamente el criterio)."
)

_LIKERT_SCALE_EN = (
    "Likert scale 1-5: "
    "1 = Strongly disagree (does not meet the criterion), "
    "2 = Disagree, "
    "3 = Neutral, "
    "4 = Agree, "
    "5 = Strongly agree (fully meets the criterion)."
)

_CLASSIFICATIONS_ES = (
    "- Historia de Usuario Válida\n"
    "- Historia de Usuario Incompleta\n"
    "- Reporte de error o incidente\n"
    "- Solicitud incompleta"
)

_CLASSIFICATIONS_EN = (
    "- Valid User Story\n"
    "- Incomplete User Story\n"
    "- Error or incident report\n"
    "- Incomplete request"
)

_JSON_SCHEMA_ES = (
    '[\n'
    '  {\n'
    '    "index": 0,\n'
    '    "clasificacion": "Historia de Usuario Válida",\n'
    '    "scores": {"I": 4, "N": 3, "V": 5, "E": 3, "S": 4, "T": 2},\n'
    '    "justificaciones": {\n'
    '      "I": "Justificación para Independiente...",\n'
    '      "N": "Justificación para Negociable...",\n'
    '      "V": "Justificación para Valiosa...",\n'
    '      "E": "Justificación para Estimable...",\n'
    '      "S": "Justificación para Pequeña...",\n'
    '      "T": "Justificación para Testeable..."\n'
    '    }\n'
    '  }\n'
    ']'
)

_JSON_SCHEMA_EN = (
    '[\n'
    '  {\n'
    '    "index": 0,\n'
    '    "clasificacion": "Valid User Story",\n'
    '    "scores": {"I": 4, "N": 3, "V": 5, "E": 3, "S": 4, "T": 2},\n'
    '    "justificaciones": {\n'
    '      "I": "Justification for Independent...",\n'
    '      "N": "Justification for Negotiable...",\n'
    '      "V": "Justification for Valuable...",\n'
    '      "E": "Justification for Estimable...",\n'
    '      "S": "Justification for Small...",\n'
    '      "T": "Justification for Testeable..."\n'
    '    }\n'
    '  }\n'
    ']'
)


def _build_prompt_es(batch: List[str], batch_offset: int, dominio: str, descripcion: str) -> str:
    criteria_lines = "\n".join(f"  - {v}" for v in _CRITERIA_DESC_ES.values())
    stories_lines = "\n".join(
        f'{i + batch_offset}. "{s}"' for i, s in enumerate(batch)
    )
    return (
        f"Rol: {_PROFILE_ES}\n\n"
        f"Contexto: Estás evaluando historias del dominio \"{dominio}\" — {descripcion}.\n\n"
        f"Tarea: Evalúa INDIVIDUALMENTE cada historia de usuario del listado adjunto. "
        f"Procesa cada una de forma independiente.\n\n"
        f"[FASE 1] Clasifica cada historia en una de estas categorías:\n"
        f"{_CLASSIFICATIONS_ES}\n\n"
        f"[FASE 2] Evalúa cada criterio INVEST con {_LIKERT_SCALE_ES}\n"
        f"Criterios:\n{criteria_lines}\n\n"
        f"IMPORTANTE: Responde ÚNICAMENTE con un JSON array válido siguiendo este esquema "
        f"(un objeto por historia, en el mismo orden):\n{_JSON_SCHEMA_ES}\n\n"
        f"Historias a evaluar:\n{stories_lines}"
    )


def _build_prompt_en(batch: List[str], batch_offset: int, dominio: str, descripcion: str) -> str:
    criteria_lines = "\n".join(f"  - {v}" for v in _CRITERIA_DESC_EN.values())
    stories_lines = "\n".join(
        f'{i + batch_offset}. "{s}"' for i, s in enumerate(batch)
    )
    return (
        f"Role: {_PROFILE_EN}\n\n"
        f"Context: You are evaluating user stories from the \"{dominio}\" domain — {descripcion}.\n\n"
        f"Task: Evaluate EACH user story from the list below INDIVIDUALLY. "
        f"Process each one independently.\n\n"
        f"[PHASE 1] Classify each story into one of these categories:\n"
        f"{_CLASSIFICATIONS_EN}\n\n"
        f"[PHASE 2] Evaluate each INVEST criterion using {_LIKERT_SCALE_EN}\n"
        f"Criteria:\n{criteria_lines}\n\n"
        f"IMPORTANT: Respond ONLY with a valid JSON array following this schema "
        f"(one object per story, in the same order):\n{_JSON_SCHEMA_EN}\n\n"
        f"Stories to evaluate:\n{stories_lines}"
    )


# ─── Evaluador ─────────────────────────────────────────────────────────────────

class InvestEvaluator:
    """
    Evalúa historias de usuario bajo el marco INVEST usando gpt-4o.

    Procesa las historias en batches de BATCH_SIZE para respetar
    los límites de tokens, siguiendo la metodología del paper.
    """

    CRITERIA = ["I", "N", "V", "E", "S", "T"]
    BATCH_SIZE = 5

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No se encontró una API key de OpenAI. "
                "Pasá api_key al constructor o configurá la variable de entorno OPENAI_API_KEY."
            )
        self._client = OpenAI(api_key=resolved_key)
        self._model = model

    def evaluate(
        self,
        stories: List[str],
        metadata: Dict = None
    ) -> InvestEvaluationResult:
        """
        Evalúa un conjunto de historias de usuario bajo INVEST.

        Args:
            stories: Lista de historias generadas.
            metadata: Diccionario con al menos 'dominio' y 'descripcion'
                      (de metadata.json del caso de prueba).

        Returns:
            InvestEvaluationResult con resultados por historia y métricas agregadas.
        """
        metadata = metadata or {}
        dominio = metadata.get("dominio", "software")
        descripcion = metadata.get("descripcion", "proyecto de software")

        lang = self._detect_language(stories)
        total_batches = (len(stories) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        print(f"  [LEVEL 5] Idioma detectado: {lang} | {len(stories)} historias en {total_batches} batches")

        all_results: List[InvestStoryResult] = []
        for batch_num, batch_start in enumerate(range(0, len(stories), self.BATCH_SIZE), start=1):
            batch = stories[batch_start: batch_start + self.BATCH_SIZE]
            batch_end = min(batch_start + len(batch) - 1, len(stories) - 1)
            print(f"  [LEVEL 5] Batch {batch_num}/{total_batches} (historias {batch_start}–{batch_end})...", end=" ", flush=True)
            t0 = time.time()
            prompt = self._build_prompt(batch, batch_start, dominio, descripcion, lang)
            raw_response = self._call_api(prompt)
            batch_results = self._parse_response(raw_response, batch, batch_start)
            all_results.extend(batch_results)
            print(f"OK ({time.time() - t0:.1f}s)")

        aggregate = self._aggregate(all_results)

        return InvestEvaluationResult(
            story_results=all_results,
            aggregate=aggregate,
            language_detected=lang,
            model_used=self._model,
            total_stories=len(stories),
        )

    # ── Métodos internos ───────────────────────────────────────────────────────

    def _detect_language(self, stories: List[str]) -> str:
        if not _LANGDETECT_AVAILABLE or not stories:
            return "en"
        try:
            sample = " ".join(stories[:3])
            detected = _langdetect_detect(sample)
            return detected if detected in ("es", "en") else "en"
        except Exception:
            return "en"

    def _build_prompt(
        self,
        batch: List[str],
        batch_offset: int,
        dominio: str,
        descripcion: str,
        lang: str,
    ) -> str:
        if lang == "es":
            return _build_prompt_es(batch, batch_offset, dominio, descripcion)
        return _build_prompt_en(batch, batch_offset, dominio, descripcion)

    def _call_api(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un evaluador experto de calidad de historias de usuario. "
                        "Responde ÚNICAMENTE con JSON válido, sin texto adicional, "
                        "sin bloques de código, sin explicaciones fuera del JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _parse_response(
        self,
        response: str,
        batch: List[str],
        offset: int,
    ) -> List[InvestStoryResult]:
        # Limpiar posibles bloques de código markdown
        clean = response
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Si el parsing falla, retornar resultados de error para todo el batch
            return [
                self._error_result(offset + i, story, "Error al parsear respuesta del LLM")
                for i, story in enumerate(batch)
            ]

        results: List[InvestStoryResult] = []
        for i, story in enumerate(batch):
            absolute_index = offset + i
            # Buscar el objeto correspondiente al índice
            item = next(
                (d for d in data if isinstance(d, dict) and d.get("index") == absolute_index),
                None,
            )
            if item is None and i < len(data):
                # Fallback: tomar por posición si el índice no coincide
                item = data[i]

            if item is None:
                results.append(
                    self._error_result(absolute_index, story, "No se encontró resultado en la respuesta")
                )
                continue

            scores_raw = item.get("scores", {})
            scores = {}
            for c in self.CRITERIA:
                val = scores_raw.get(c)
                try:
                    scores[c] = max(1, min(5, int(val)))
                except (TypeError, ValueError):
                    scores[c] = 1

            justifications = {c: item.get("justificaciones", {}).get(c, "") for c in self.CRITERIA}
            invest_mean = round(statistics.mean(scores.values()), 3)

            results.append(
                InvestStoryResult(
                    story_index=absolute_index,
                    story_text=story,
                    classification=item.get("clasificacion", "Desconocida"),
                    scores=scores,
                    justifications=justifications,
                    invest_mean=invest_mean,
                )
            )

        return results

    def _error_result(self, index: int, story: str, reason: str) -> InvestStoryResult:
        return InvestStoryResult(
            story_index=index,
            story_text=story,
            classification=f"Error: {reason}",
            scores={c: 0 for c in self.CRITERIA},
            justifications={c: "" for c in self.CRITERIA},
            invest_mean=0.0,
        )

    def _aggregate(self, results: List[InvestStoryResult]) -> Dict:
        valid_results = [r for r in results if r.invest_mean > 0]
        if not valid_results:
            return {
                "overall_mean": 0.0,
                "scores_per_criterion": {},
                "error": "No hay resultados válidos para agregar",
            }

        scores_per_criterion: Dict[str, Dict] = {}
        for criterion in self.CRITERIA:
            vals = [r.scores[criterion] for r in valid_results]
            distribution = {str(k): vals.count(k) for k in range(1, 6)}
            scores_per_criterion[criterion] = {
                "mean": round(statistics.mean(vals), 3),
                "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "distribution": distribution,
            }

        all_means = [r.invest_mean for r in valid_results]
        return {
            "overall_mean": round(statistics.mean(all_means), 3),
            "scores_per_criterion": scores_per_criterion,
            "valid_stories": len(valid_results),
            "error_stories": len(results) - len(valid_results),
        }
