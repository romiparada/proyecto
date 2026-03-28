"""Refinamiento asistido por LLM basado en los resultados del pipeline."""

import json
import urllib.request
from urllib.error import HTTPError
from typing import List, Dict, Any


def call_llm(prompt: str, api_key: str, model: str) -> str:
    """Llama a un LLM externo vía REST. Soporta OpenAI, Gemini y OpenRouter."""
    if not api_key:
        raise ValueError("Se requiere API key para llamar al LLM.")

    if "gemini" in model.lower():
        # Google Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except HTTPError as e:
            raise RuntimeError(f"Gemini API Error: {e.code} - {e.read().decode('utf-8')}")
    
    elif "sk-or-v1" in api_key or "/" in model:  # formato OpenRouter
        # OpenRouter
        url = "https://openrouter.ai/api/v1/chat/completions"
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except HTTPError as e:
            raise RuntimeError(f"OpenRouter API Error: {e.code} - {e.read().decode('utf-8')}")
        
    else:
        # OpenAI (fallback por defecto)
        url = "https://api.openai.com/v1/chat/completions"
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"))
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {api_key}")
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except HTTPError as e:
            raise RuntimeError(f"OpenAI API Error: {e.code} - {e.read().decode('utf-8')}")


def identify_issues(coverage_data: Dict, matching_data: Dict, hallucination_data: Dict) -> Dict[str, List[Any]]:
    """Extrae los problemas detectados por el pipeline de evaluación."""
    issues = {
        "uncovered": [],
        "partial": [],
        "epics": [],
        "hallucinations": []
    }
    
    # 1. Problemas de cobertura
    for c in coverage_data.get("results", []):
        if c.get("coverage_status") == "not_covered":
            issues["uncovered"].append(f"[{c['category']}] {c['functionality']}")
        elif c.get("coverage_status") == "possibly_covered":
            issues["partial"].append(f"[{c['category']}] {c['functionality']} (Mejor match: {c['best_match']['story_text']})")

    # 2. Historias posiblemente demasiado amplias (épicas)
    for m in matching_data.get("results", []):
        # Si matchea alto con más de 1 historia esperada, podría ser un épico
        high_matches = [match for match in m.get("top_matches", []) if match.get("similarity", 0) > 0.70]
        if len(high_matches) > 1:
            issues["epics"].append(m["generated_story"])

    # 3. Alucinaciones
    for h in hallucination_data.get("results", []):
        if h.get("hallucination_flag") in ["possible_hallucination", "uncertain"]:
            issues["hallucinations"].append(h["generated_story"])
            
    return issues


def run_refinement(
    generated_stories: List[str],
    functionalities: List[Dict],
    coverage_data: Dict,
    matching_data: Dict,
    hallucination_data: Dict,
    prd_text: str,
    api_key: str,
    model: str
) -> Dict[str, Any]:
    """Ejecuta refinamiento asistido por LLM sobre las historias generadas."""
    
    # 1. Identificar problemas
    issues = identify_issues(coverage_data, matching_data, hallucination_data)

    # 2. Construir prompt
    prompt_template = """You are an expert Requirements Engineer and Agile Product Owner. 
We have evaluated a set of LLM-generated user stories against a Product Requirements Document (PRD) and expected domain functionalities.
Your task is to provide CONCRETE SUGGESTIONS for refining the generated user stories.

PRD TEXT:
\"\"\"
{prd}
\"\"\"

---
ISSUES FOUND DURING THE SEMANTIC PIPELINE EVALUATION:

1. Functionalities NOT completely covered (Need new stories to fill the gap):
{uncovered}

2. Functionalities only partially covered (Need story improvements/clarifications):
{partial}

3. Generated stories that might be too broad (matched multiple distinct expected criteria, possible EPICS to split):
{epics}

4. Generated stories flagged as HALLUCINATION or UNCERTAIN (Not grounded in PRD or functionalities):
{hallucinations}

---
TASK:

Based on the PRD and the identified issues, please generate a structured JSON containing ONLY your suggestions. 
Do not include any markdown block markers like ```json or any additional text outside the JSON object. 
The JSON must EXACTLY follow this structure:

{{
  "new_stories": [
    {{
      "functionality": "...",
      "suggested_story": "..."
    }}
  ],
  "improvements": [
    {{
      "original_story": "...",
      "suggestion": "..."
    }}
  ],
  "splits": [
    {{
      "original_story": "...",
      "suggested_stories": ["...", "..."]
    }}
  ],
  "reclassification": [
    {{
      "story": "...",
      "issue": "e.g., non-functional requirement / vague / hallucinated",
      "suggestion": "..."
    }}
  ]
}}
"""

    prompt = prompt_template.format(
        prd=prd_text,
        uncovered="\\n".join(f"- {s}" for s in issues["uncovered"]) if issues["uncovered"] else "None",
        partial="\\n".join(f"- {s}" for s in issues["partial"]) if issues["partial"] else "None",
        epics="\\n".join(f"- {s}" for s in issues["epics"]) if issues["epics"] else "None",
        hallucinations="\\n".join(f"- {s}" for s in issues["hallucinations"]) if issues["hallucinations"] else "None"
    )
    
    # 3. Llamar al LLM
    try:
        response_text = call_llm(prompt, api_key, model)

        # Limpiar formato markdown si lo hay
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        json_output = json.loads(response_text.strip())
        return json_output
        
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {e}\nRaw Response: {response_text}")
    except Exception as e:
        raise RuntimeError(f"Error during LLM refinement: {str(e)}")
