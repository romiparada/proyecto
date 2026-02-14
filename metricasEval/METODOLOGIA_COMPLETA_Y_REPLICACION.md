# Metodología de Evaluación de Historias de Usuario - Completa y Replicable

## 📋 Índice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Parte I: Explicación de la Metodología](#parte-i-explicación-de-la-metodología)
   - [Arquitectura General](#arquitectura-general)
   - [LEVEL 0: Semantic Encoder](#level-0-semantic-encoder)
   - [LEVEL 1: Story Alignment](#level-1-story-alignment)
   - [LEVEL 2: Story Coverage](#level-2-story-coverage)
   - [LEVEL 3: Acceptance Criteria](#level-3-acceptance-criteria)
   - [LEVEL 4: Concept Coverage](#level-4-concept-coverage)
3. [Parte II: Guía de Replicación Paso a Paso](#parte-ii-guía-de-replicación-paso-a-paso)
   - [Fase 0: Setup del Entorno](#fase-0-setup-del-entorno)
   - [Fase 1: Crear Estructura del Proyecto](#fase-1-crear-estructura-del-proyecto)
   - [Fase 2: Implementar LEVEL 0](#fase-2-implementar-level-0)
   - [Fase 3: Implementar LEVEL 1](#fase-3-implementar-level-1)
   - [Fase 4: Implementar LEVEL 2](#fase-4-implementar-level-2)
   - [Fase 5: Implementar LEVEL 3](#fase-5-implementar-level-3)
   - [Fase 6: Implementar LEVEL 4](#fase-6-implementar-level-4)
   - [Fase 7: Crear el Pipeline Orquestador](#fase-7-crear-el-pipeline-orquestador)
   - [Fase 8: Crear Scripts de Ejecución](#fase-8-crear-scripts-de-ejecución)
   - [Fase 9: Preparar Casos de Prueba](#fase-9-preparar-casos-de-prueba)
   - [Fase 10: Ejecutar y Validar](#fase-10-ejecutar-y-validar)

---

## ❓ FAQ - Preguntas Frecuentes

### P1: ¿Por qué usar SemanticEncoder si SBERT ya codifica internamente?

**R**: `SemanticEncoder` **ES** SBERT, no un paso adicional. Es un wrapper (patrón de diseño Facade) que:
- Encapsula `SentenceTransformer` sin agregar procesamiento extra
- Centraliza configuración y manejo de errores (Windows DLL, device, etc.)
- Provee métodos con nombres más claros (`encode_stories()` vs `encode()`)
- Facilita futuros cambios de modelo

Cuando llamas `encoder.encode()`, internamente ejecuta `SentenceTransformer.encode()` directamente.

### P2: ¿Los embeddings se calculan antes de pasar los textos a SBERT?

**R**: No. SBERT es quien calcula los embeddings. El flujo es:
```
Texto → encoder.encode(texto) → SentenceTransformer.encode(texto) → Embedding
```

### P3: ¿Los CA se codifican todos juntos o por pares?

**R**: **Por pares**. Para cada par de historias alineadas (gen_i, esp_j):
1. Se extraen SOLO los CA de ese par específico
2. Se codifican SOLO los CA de ese par
3. Se comparan CA dentro del par (all-vs-all)
4. **NO** se mezclan CA de diferentes pares

### P4: ¿Por qué usar batch encoding?

**R**: Eficiencia. Codificar 23 historias en 1 llamada es 10-20x más rápido que 23 llamadas individuales:
- ✅ **Batch**: `encode([h1, h2, ..., h23])` = 1 llamada
- ❌ **Individual**: `encode(h1) + encode(h2) + ...` = 23 llamadas

---

# PARTE I: Explicación de la Metodología

## Resumen Ejecutivo

Esta metodología evalúa la calidad de historias de usuario generadas automáticamente (por ejemplo, mediante LLMs) comparándolas con historias esperadas (ground truth). La evaluación se realiza en **5 niveles jerárquicos** que van desde la representación semántica básica hasta la cobertura conceptual del dominio.

### Objetivos Clave

1. **Evaluar alineación**: ¿Las historias generadas coinciden semánticamente con las esperadas?
2. **Medir cobertura**: ¿Qué proporción de historias esperadas están representadas?
3. **Validar criterios de aceptación**: ¿Los CA generados cubren los requisitos funcionales?
4. **Verificar cobertura conceptual**: ¿Los conceptos del dominio están representados?

### Tecnologías Core

- **SBERT** (Sentence-BERT): `all-mpnet-base-v2` para embeddings semánticos
- **BERTScore** (opcional): Validación secundaria con `microsoft/deberta-xlarge-mnli`
- **Python 3.8+**: PyTorch, sentence-transformers, bert-score
- **Similitud coseno**: Métrica principal de comparación semántica

### 📌 Nota Importante sobre "Encoding" y SBERT

En esta metodología, **`SemanticEncoder` ES SBERT**, no un paso adicional:
- `SemanticEncoder` es un wrapper (patrón Facade) alrededor de `SentenceTransformer`
- Cuando llamamos `encoder.encode(texts)`, internamente llama directamente a `SentenceTransformer.encode(texts)`
- **NO hay encoding previo ni procesamiento extra**
- El wrapper solo centraliza configuración y provee nombres de métodos más claros

Por tanto, cuando veas "codificar con SBERT" significa simplemente llamar al encoder, que internamente usa SBERT directamente.

---

## Arquitectura General

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                       │
│                  (pipeline/evaluation_pipeline.py)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
      ┌───────────────────────────────────────────────┐
      │  LEVEL 0: Semantic Encoder (SBERT)            │
      │  • Codifica textos a embeddings vectoriales   │
      │  • Modelo: all-mpnet-base-v2                  │
      │  • Dimensión: 768 valores por texto           │
      └───────────────────────────────────────────────┘
                              │
                              ▼
      ┌───────────────────────────────────────────────┐
      │  LEVEL 1: Story Alignment                     │
      │  • SBERT: matching argmax + clasificación     │
      │  • BERTScore: validación secundaria OPCIONAL  │
      │  • Filtrado: solo Strong (≥0.80) y           │
      │    Conservative (≥0.85) pasan                 │
      └───────────────────────────────────────────────┘
                              │
                              ▼
      ┌───────────────────────────────────────────────┐
      │  LEVEL 2: Story Coverage                      │
      │  • Coverage(Gen, Esp): % esperadas cubiertas  │
      │  • Diversity = 100 - Coverage(Esp, Gen)       │
      │  • Umbral: 0.75 para considerar "cubierta"    │
      └───────────────────────────────────────────────┘
                              │
                              ▼
      ┌───────────────────────────────────────────────┐
      │  LEVEL 3: Acceptance Criteria (solo alineadas)│
      │  • Cobertura funcional: all-vs-all dentro par │
      │  • Verificabilidad: condición + resultado     │
      │  • Ambigüedad: términos vagos detectados      │
      └───────────────────────────────────────────────┘
                              │
                              ▼
      ┌───────────────────────────────────────────────┐
      │  LEVEL 4: Conceptual Coverage (solo alineadas)│
      │  • Mide cobertura de aspectos/conceptos       │
      │  • Umbral: 0.70 para considerar "cubierto"    │
      └───────────────────────────────────────────────┘
```

### Flujo de Datos

```
INPUT → LEVEL 0 (encoding) → LEVEL 1 (alignment + filter)
                                      │
                                      ├─→ Aligned stories → LEVEL 3 → LEVEL 4
                                      │
                                      └─→ All stories → LEVEL 2
```

**⚠️ IMPORTANTE**: LEVEL 3 y LEVEL 4 SOLO evalúan historias que pasaron el filtro de alineación en LEVEL 1 (Strong o Conservative).

---

## LEVEL 0: Semantic Encoder

**Archivo**: `pipeline/semantic_encoder.py`

### Responsabilidad

Transformar texto en lenguaje natural a vectores numéricos (embeddings de 768 dimensiones) para poder calcular similitud semántica entre textos usando **similitud coseno**.

**⚠️ ACLARACIÓN IMPORTANTE**: `SemanticEncoder` **ES** SBERT. No es un paso previo ni una capa extra de encoding. Es simplemente un wrapper (capa de abstracción) que encapsula `SentenceTransformer` para:
- Centralizar configuración
- Manejar fixes de Windows
- Proveer métodos semánticamente claros (`encode_stories()`, `encode_acceptance_criteria()`)
- Facilitar cambios futuros de modelo

Cuando llamamos a `encoder.encode()`, internamente llama directamente a `SentenceTransformer.encode()` sin procesamiento adicional.

### Configuración

```python
MODELO = "sentence-transformers/all-mpnet-base-v2"
DIMENSIÓN = 768 valores numéricos por texto
DEVICE = CPU (con fix para Windows DLL loading)
BATCH_SIZE = 16 (configurable)
```

### ¿Qué textos se codifican?

**LEVEL 1 y LEVEL 2 (Historias)**:
1. **Historias generadas** (todas en batch: 1 llamada)
2. **Historias esperadas** (todas en batch: 1 llamada)

**LEVEL 3 (Criterios de Aceptación)**:
3. **CA por pares** (para cada par de historias alineadas, se codifican SOLO los CA de ese par)
   - **NO** se codifican todos los CA juntos
   - Se itera sobre cada par (gen_i, esp_j) y se codifican los CA de ese par específico

**LEVEL 4 (Conceptos del Dominio)**:
4. **Descripciones de conceptos** (todas en batch)
5. **Historias alineadas** (todas en batch)

### Ejemplo de Codificación

```python
# Input
texts = [
  "As administrative staff, I want to register patients",
  "As a doctor, I want to view patient records"
]

# Proceso
encoder = SemanticEncoder()
embeddings = encoder.encode(texts, convert_to_tensor=True)

# Output
# Tensor shape: [2, 768]
# [[0.12, -0.34, 0.56, ..., 0.23],   # Historia 1 → 768 números
#  [0.08, -0.29, 0.44, ..., 0.19]]   # Historia 2 → 768 números
```

### Cálculo de Similitud

```python
from sentence_transformers import util

# Similitud entre 2 textos
sim = util.cos_sim(embedding1, embedding2)  # Resultado: 0.0 a 1.0

# Matriz de similitudes (23 generadas vs 28 esperadas)
matriz = util.cos_sim(emb_generadas, emb_esperadas)  # Shape: [23, 28]
# matriz[i][j] = similitud entre historia_gen[i] y historia_esp[j]
```

### Fix para Windows

```python
# Necesario para evitar errores de DLL en Windows
if sys.platform == "win32":
    dll_path = os.path.join(sys.prefix, "Lib", "site-packages", "numpy.libs")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
```

---

## LEVEL 1: Story Alignment

**Archivo**: `pipeline/story_alignment_evaluator.py`

### Responsabilidad

Alinear cada historia generada con su mejor match en las esperadas usando **SBERT exclusivamente** para decidir el matching y clasificación. Filtrar historias débilmente alineadas.

### Input

- **Historias generadas** (lista de strings)
- **Historias esperadas** (lista de strings, ground truth)

### Proceso Paso a Paso

#### 1. Codificar Ambas Listas en Batch (Eficiente)

```python
# Codificar TODAS las historias de una vez (batch encoding)
emb_gen = encoder.encode_stories(stories_generated)  # [23, 768]
emb_exp = encoder.encode_stories(stories_expected)   # [28, 768]
```

**⚠️ IMPORTANTE**: El encoding se hace en **batch** (todas a la vez), NO una por una. Esto es 10-20x más rápido.

#### 2. Calcular Matriz de Similitudes Completa

```python
# Matriz de similitud de una sola vez
similarity_matrix = util.cos_sim(emb_gen, emb_exp)   # [23, 28]

# similarity_matrix[i][j] = similitud entre historia_gen[i] y historia_esp[j]
# Para cada historia generada, tenemos 28 similitudes (una por cada esperada)
```

#### 3. Para Cada Historia: Argmax (Mejor Match)

```python
# Para CADA historia generada, buscar en la matriz
for i in range(len(stories_generated)):
    similarities = similarity_matrix[i]  # Fila i: [28 valores]
    
    # Argmax: índice del valor máximo
    best_match_idx = similarities.argmax()  # → índice de la esperada con mayor similitud
    best_similarity = similarities[best_match_idx]  # → valor de esa similitud
    
    # Resultado:
    # Historia generada i ↔ Historia esperada best_match_idx
    # Similitud SBERT: best_similarity
```

#### 4. Clasificación por Umbral

```python
if best_similarity >= 0.85:
    nivel = "conservative"  # ✓ ALINEADA (pasa a LEVEL 3/4)
elif best_similarity >= 0.80:
    nivel = "strong"        # ✓ ALINEADA (pasa a LEVEL 3/4)
else:
    nivel = "weak"          # ✗ FILTRADA (no continúa)
```

#### 5. BERTScore (OPCIONAL, Secundario)

```python
# Se calcula SOLO entre la generada y su match SBERT
# NO afecta la alineación ni el filtrado
bertscore_f1 = bert_score(
    cands=["historia generada"],
    refs=["historia esperada (match SBERT)"],
    model="microsoft/deberta-xlarge-mnli"
)
# Resultado: 0.91 (o null si no disponible)
# ⚠️ Este valor es INFORMATIVO, NO decide nada
```

### Umbrales Clave

```python
UMBRAL_STRONG       = 0.80  # Alineación fuerte
UMBRAL_CONSERVATIVE = 0.85  # Alineación conservadora (más estricta)
# Weak: sim < 0.80 → NO PASA a niveles siguientes
```

### Interpretación

- **Conservative (≥0.85)**: Historias casi idénticas semánticamente
- **Strong (≥0.80)**: Historias claramente relacionadas
- **Weak (<0.80)**: Historias diferentes o poco relacionadas → **FILTRADAS**

### Ejemplo Real: Historia #7

**Texto generado**:
> "As a patient, I want to receive notifications when my appointment is modified or canceled, so that I am always informed about my schedule."

**Proceso**:
1. Se codifica EN BATCH con todas las demás (23 historias a la vez)
2. Se calcula matriz de similitud [23 x 28] de una vez
3. Para la historia #7, se busca en la fila 7 de la matriz
4. Mejor match: esperada #7 con similitud **0.9751** (argmax de la fila)
5. Clasificación: 0.9751 ≥ 0.85 → **"conservative"**
6. BERTScore: null (opcional, calculado después)
7. **Resultado: ✓ ALINEADA** → pasa a LEVEL 3 y 4

**Texto esperado matched**:
> "As a patient, I want to be notified when my appointment is modified or canceled to stay informed."

### Output

```json
{
  "aligned_stories": [
    {
      "index_generated": 7,
      "text_generated": "As a patient, I want to receive notifications...",
      "index_matched": 7,
      "text_matched": "As a patient, I want to be notified...",
      "sbert_similarity": 0.9751,
      "bertscore_f1": null,
      "alignment_level": "conservative",
      "is_aligned": true
    }
  ],
  "metrics": {
    "total_generated": 23,
    "aligned_count": 10,
    "strong_count": 2,
    "conservative_count": 8,
    "weak_count": 13,
    "alignment_rate": 0.4348,  // 10/23
    "sbert_mean_aligned": 0.8654,
    "sbert_mean_all": 0.7123
  }
}
```

### Ejemplo de Historia Filtrada (Weak)

```json
{
  "index_generated": 3,
  "text_generated": "As an authorized user, I want to visualize a patient's complete information...",
  "index_matched": 8,
  "text_matched": "As a physician, I want to access the patient's electronic health record...",
  "sbert_similarity": 0.6331,  // ← Muy baja
  "alignment_level": "weak",
  "is_aligned": false  // ← NO PASA a LEVEL 3/4
}
```

---

## LEVEL 2: Story Coverage

**Archivo**: `pipeline/story_coverage_calculator.py`

### Responsabilidad

Calcular qué porcentaje de historias esperadas (ground truth) están cubiertas por las historias generadas.

**⚠️ IMPORTANTE**: Este nivel usa **TODAS** las historias generadas, incluyendo las filtradas como "weak" en LEVEL 1.

### Input

- **Historias generadas** (todas, 23 en el ejemplo)
- **Historias esperadas** (todas, 28 en el ejemplo)
- **Umbral**: 0.75 (para considerar "cubierta")

### Fórmula

```python
Coverage(Gen, Esp) = |{e ∈ Esp : max_sim(e, Gen) ≥ 0.75}| / |Esp|

# En español:
# Para cada historia esperada:
#   Encuentra la similitud máxima con cualquier historia generada
#   Si max_similitud ≥ 0.75 → esperada está "cubierta"
# Score = esperadas_cubiertas / total_esperadas
```

### Proceso Paso a Paso

#### Para cada historia esperada:

```python
# Historia esperada #0
esperada = "As administrative staff, I want to register the data of a new patient..."

# 1. Codificar
emb_esperada = encoder.encode([esperada])  # [1, 768]
emb_todas_gen = encoder.encode(stories_generated)  # [23, 768]

# 2. Calcular similitud con TODAS las generadas
similitudes = util.cos_sim(emb_esperada, emb_todas_gen)[0]
# [0.8596, 0.6421, 0.5234, ..., 0.3912]  # 23 valores

# 3. Encontrar máximo
max_sim = max(similitudes)  # → 0.8596
best_match_idx = 0
best_match_text = stories_generated[0]

# 4. Verificar umbral
if max_sim >= 0.75:
    cubierta = True  # ✓ CUBIERTA
else:
    cubierta = False  # ✗ NO CUBIERTA
```

### Ejemplo Real

De **28 historias esperadas**:
- **16 cubiertas** (similitud máxima ≥ 0.75)
- **12 no cubiertas** (similitud máxima < 0.75)
- **Coverage score**: 16/28 = **57.14%**

### Output

```json
{
  "story_coverage_score": 0.5714,
  "diversity_complement": 42.86,
  "covered": 16,
  "total": 28,
  "threshold": 0.75,
  "details": [
    {
      "expected_idx": 0,
      "expected_text": "As administrative staff, I want to register...",
      "max_similarity": 0.8596,
      "covered": true,
      "best_match_idx": 0,
      "best_match_text": "As an administrative staff member..."
    },
    {
      "expected_idx": 9,
      "expected_text": "As a physician, I want to record the diagnosis...",
      "max_similarity": 0.7032,  // ← Por debajo del umbral
      "covered": false,
      "best_match_idx": -1,
      "best_match_text": null
    }
  ]
}
```

### Diversity (Métrica Complementaria)

```python
Diversity = 100 - Coverage(Esp, Gen)

# Para el ejemplo:
Diversity = 100 - 57.14 = 42.86%
```

**Interpretación**:
- **Coverage alto (>70%)**: El modelo genera las historias esperadas
- **Diversity alto (>40%)**: El modelo genera historias adicionales no esperadas
  - Puede ser BUENO: Cubre casos no previstos
  - Puede ser MALO: Genera historias irrelevantes

### Diferencia con LEVEL 1

| Aspecto | LEVEL 1 (Alignment) | LEVEL 2 (Coverage) |
|---------|---------------------|-------------------|
| Dirección | Gen → Esp | Esp → Gen |
| Objetivo | Clasificar generadas | Cubrir esperadas |
| Filtro | Sí (weak rechazadas) | No (usa todas) |
| Umbral | 0.80/0.85 | 0.75 |
| Usa para LEVEL 3/4 | Sí (solo alineadas) | No |

---

## LEVEL 3: Acceptance Criteria

**Archivo**: `pipeline/acceptance_criteria_evaluator.py`

### ⚠️ REGLA CRÍTICA

**Solo evalúa criterios de aceptación de historias ALINEADAS** (Strong + Conservative del LEVEL 1).

### Input

**Del LEVEL 1 vienen pares alineados**:
```python
aligned_pairs = [
  (gen_idx=0, esp_idx=0),   # Par #1
  (gen_idx=1, esp_idx=1),   # Par #2
  (gen_idx=2, esp_idx=3),   # Par #3
  # ... más pares
]

# Y los CA completos
ca_generated = [
  ["CA 1 de historia 0", "CA 2 de historia 0", ...],  # Historia 0
  ["CA 1 de historia 1", ...],                        # Historia 1
  # ...
]

ca_expected = [
  ["CA 1 esperado historia 0", ...],  # Historia 0
  # ...
]
```

### Métricas Evaluadas

#### 3.1 Functional Coverage (Cobertura Funcional)

**Fórmula**:
```python
Coverage_CA = covered_expected_CA / total_expected_CA

# Para un par (historia generada i, historia esperada j):
# Para cada CA esperado del par:
#   Encuentra la similitud máxima con cualquier CA generado del par
#   Si max_similitud ≥ 0.75 → CA esperado está "cubierto"
```

**⚠️ IMPORTANTE**: Los CA se codifican **PAR POR PAR**, NO todos juntos.

**Proceso (para cada par de historias alineadas)**:
```python
# Iterar sobre cada par de historias alineadas
for (gen_idx, esp_idx) in aligned_pairs:
    # Extraer SOLO los CA de este par específico
    ca_gen = ca_generated[gen_idx]  # Ej: 3 CA de esta historia generada
    ca_exp = ca_expected[esp_idx]   # Ej: 8 CA de esta historia esperada
    
    # Codificar SOLO los CA de este par (no todos los CA de todas las historias)
    emb_gen = encoder.encode(ca_gen)  # [3, 768]
    emb_exp = encoder.encode(ca_exp)  # [8, 768]
    
    # Comparar CA dentro del par: ALL-vs-ALL
    covered = 0
    for i, emb_e in enumerate(emb_exp):
        # Similitud del CA esperado i con TODOS los CA generados del par
        sims = util.cos_sim(emb_e, emb_gen)[0]  # [3 similitudes]
        max_sim = max(sims)  # Mejor match
        
        if max_sim >= 0.75:
            covered += 1
    
    coverage_par = covered / len(ca_exp)  # Ej: 6/8 = 0.75
    
# Agregación: promedio de coverage de todos los pares
avg_coverage = sum(coverage_par_i) / num_pares
```

**Ejemplo concreto**:
```
Par 1: Historia gen #0 ↔ Historia esp #0
  CA generados: 3 CA
  CA esperados: 8 CA
  Coverage: 6/8 = 0.75 (6 CA esperados tienen match ≥0.75)
  
Par 2: Historia gen #1 ↔ Historia esp #1
  CA generados: 2 CA
  CA esperados: 5 CA
  Coverage: 4/5 = 0.80
  
Promedio: (0.75 + 0.80) / 2 = 0.775
```

**✗ LO QUE NO SE HACE**: NO se codifican todos los CA de todas las historias juntos para luego compararlos globalmente. Cada par se evalúa independientemente.

#### 3.2 Verificabilidad

Mide si los CA tienen:
- **Condición** (Given/When/If)
- **Resultado observable** (Then/Must/Shall/Displays)

**Patrones de detección**:
```python
CONDITION_PATTERNS = [
    r"\bif\b", r"\bwhen\b", r"\bgiven\b", 
    r"\bprovided\b", r"\bassuming\b"
]

RESULT_PATTERNS = [
    r"\bshall\b", r"\bmust\b", r"\bshould\b", r"\bthen\b",
    r"\bdisplays?\b", r"\breturns?\b", r"\bshows?\b",
    r"\bcreates?\b", r"\bupdates?\b", r"\bstores?\b"
]
```

**Scoring**:
```python
for ca in ca_generated:
    tiene_condicion = any(re.search(p, ca, re.I) for p in CONDITION_PATTERNS)
    tiene_resultado = any(re.search(p, ca, re.I) for p in RESULT_PATTERNS)
    
    if tiene_condicion and tiene_resultado:
        score = 1.0  # ✓ Verificable
    elif tiene_resultado:
        score = 0.5  # Parcialmente verificable
    else:
        score = 0.0  # ✗ No verificable

verificabilidad_global = promedio(scores)
```

#### 3.3 No-Ambigüedad

Detecta términos vagos en los CA generados.

**Términos ambiguos**:
```python
AMBIGUOUS_TERMS = [
    "fast", "quick", "adequate", "appropriate",
    "correct", "proper", "user-friendly", "efficient",
    "intuitive", "as soon as possible", "reasonable time",
    "easily", "simple", "good", "bad", "optimal", "best", "worst"
]
```

**Scoring**:
```python
for ca in ca_generated:
    num_ambiguous = sum(1 for term in AMBIGUOUS_TERMS if term in ca.lower())
    
    if num_ambiguous == 0:
        score = 1.0  # ✓ Sin ambigüedad
    else:
        score = 0.0  # ✗ Tiene términos ambiguos

no_ambiguity_global = promedio(scores)
ambiguity_global = 1.0 - no_ambiguity_global
```

### Output

```json
{
  "pairs_evaluated": 10,
  "avg_functional_coverage": 0.7234,
  "global_verifiability": 0.8523,
  "global_no_ambiguity": 0.9145,
  "global_ambiguity": 0.0855,
  "pairs_detail": [
    {
      "story_generated_idx": 0,
      "story_expected_idx": 0,
      "functional_coverage": 0.75,  // 6/8 CA cubiertos
      "verifiability_score": 0.8333,
      "ambiguity_score": 0.0,
      "functional_coverage_details": [
        {
          "expected_ca": "The system must request full name...",
          "covered": true,
          "max_similarity": 0.8721,
          "best_match": "The system allows entry of personal data"
        }
      ]
    }
  ]
}
```

---

## LEVEL 4: Concept Coverage

**Archivo**: `pipeline/concept_coverage_evaluator.py`

### Responsabilidad

Evaluar qué proporción de conceptos del dominio están representados en las historias alineadas.

**⚠️ IMPORTANTE**: Solo evalúa sobre historias que pasaron el filtro de alineación en LEVEL 1.

### Input

- **Historias alineadas** (solo las Strong + Conservative, textos)
- **Conceptos del dominio** (diccionario `{nombre: [descripciones]}`)
- **Umbral**: 0.70 (para considerar concepto "cubierto")

### Estructura de Conceptos

```json
{
  "pacientes": [
    "registrar pacientes con datos personales",
    "editar informacion de pacientes",
    "buscar pacientes por identificador"
  ],
  "turnos": [
    "crear turnos medicos",
    "modificar fecha y hora de turno",
    "cancelar turnos"
  ],
  "inventario": [
    "registrar medicamentos e insumos",
    "actualizar niveles de stock"
  ]
}
```

### Proceso

```python
# Para cada concepto
for concept_name, descriptions in concepts.items():
    # Codificar descripciones del concepto
    emb_concept = encoder.encode(descriptions)  # [n_desc, 768]
    
    # Codificar historias alineadas
    emb_stories = encoder.encode(aligned_stories)  # [n_aligned, 768]
    
    # Buscar máxima similitud entre cualquier descripción
    # del concepto y cualquier historia alineada
    max_sim = 0.0
    for emb_desc in emb_concept:
        sims = util.cos_sim(emb_desc, emb_stories)[0]
        sim_max = max(sims)
        if sim_max > max_sim:
            max_sim = sim_max
    
    # Verificar umbral
    if max_sim >= 0.70:
        cubierto = True
    else:
        cubierto = False
```

### Ejemplo Real

**Conceptos**: 7 (pacientes, turnos, historia_clinica, personal, facturacion, inventario, reportes)
**Historias alineadas**: 10
**Umbral**: 0.70

**Resultado**:
- **Conceptos cubiertos**: 5/7
- **Coverage**: 71.43%

```json
{
  "concept_coverage_score": 0.7143,
  "concepts_covered": 5,
  "concepts_total": 7,
  "threshold": 0.70,
  "details": [
    {
      "concept": "pacientes",
      "covered": true,
      "max_similarity": 0.8234,
      "best_description": "registrar pacientes con datos personales",
      "best_story_idx": 0,
      "best_story": "As an administrative staff member, I want to register patients..."
    },
    {
      "concept": "reportes",
      "covered": false,
      "max_similarity": 0.6521,  // ← Por debajo del umbral 0.70
      "best_description": "generar estadisticas de consultas",
      "best_story_idx": -1
    }
  ]
}
```

---

# PARTE II: Guía de Replicación Paso a Paso

Esta guía te permitirá construir el sistema de evaluación **desde cero**, paso a paso, de manera limpia y estructurada.

## Fase 0: Setup del Entorno

### 0.1 Requisitos Previos

- **Python**: 3.8 o superior
- **Sistema operativo**: Windows, Linux o macOS
- **RAM**: Mínimo 8GB (recomendado 16GB para modelos grandes)
- **Espacio en disco**: ~5GB para modelos y dependencias

### 0.2 Crear Entorno Virtual

```bash
# Crear carpeta del proyecto
mkdir metricasEval
cd metricasEval

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 0.3 Instalar Dependencias

Crear `requirements.txt`:

```txt
# Core ML
torch==2.10.0
sentence-transformers==5.2.2
transformers==5.1.0

# Métricas adicionales
bert-score==0.3.13

# Procesamiento de datos
numpy==2.4.2
pandas==3.0.0
scikit-learn==1.8.0

# Utilidades
tqdm==4.67.3
```

Instalar:
```bash
pip install -r requirements.txt
```

**⚠️ Nota Windows**: Si usas Windows, asegúrate de tener instaladas las Visual C++ Redistributables.

---

## Fase 1: Crear Estructura del Proyecto

### 1.1 Estructura de Directorios

```bash
metricasEval/
├── pipeline/
│   ├── __init__.py
│   ├── semantic_encoder.py
│   ├── story_alignment_evaluator.py
│   ├── story_coverage_calculator.py
│   ├── acceptance_criteria_evaluator.py
│   ├── concept_coverage_evaluator.py
│   └── evaluation_pipeline.py
├── casos_prueba/
│   └── caso_A_alineadas/
│       ├── metadata.json
│       ├── generadas.txt
│       ├── esperadas.txt
│       ├── ca_generados.json
│       ├── ca_esperados.json
│       └── (otros casos...)
├── resultados/
├── cargador_datos.py
├── ejecutar_pipeline.py
├── requirements.txt
└── README.md
```

Crear directorios:
```bash
mkdir pipeline casos_prueba resultados
```

---

## Fase 2: Implementar LEVEL 0

### 2.1 Crear `pipeline/semantic_encoder.py`

```python
"""
LEVEL 0 — Representación Semántica

Responsabilidad única: Codificar textos usando SBERT.
"""

import os
import sys
import warnings
import logging

# Suprimir warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# FIX Windows DLL loading
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

# Modelo SBERT recomendado
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
```

### 2.2 Probar LEVEL 0

```python
# test_level0.py
from pipeline.semantic_encoder import SemanticEncoder

encoder = SemanticEncoder()

texts = [
    "As a user, I want to register patients",
    "As administrative staff, I want to create patient records"
]

embeddings = encoder.encode(texts)
print(f"Shape: {embeddings.shape}")  # Debe ser [2, 768]
print(f"Tipo: {type(embeddings)}")   # Debe ser torch.Tensor
```

**✅ ¿Qué acabamos de hacer?**
- Creamos un wrapper simple alrededor de `SentenceTransformer`
- `encoder.encode()` internamente llama a `SentenceTransformer.encode()` sin procesamiento adicional
- Los 2 textos se codificaron en **batch** (simultáneamente), no uno por uno
- Cada texto se convirtió en un vector de 768 números

**⚠️ NO estamos "codificando antes de SBERT"** - el encoder **ES** SBERT.

---

## Fase 3: Implementar LEVEL 1

### 3.1 Crear `pipeline/story_alignment_evaluator.py`

```python
"""
LEVEL 1 — Evaluación de Historias de Usuario (HU)

Responsabilidades:
- Matching SBERT (cosine similarity) contra HU_expected
- Selección del mejor match (argmax)
- BERTScore F1 como métrica secundaria (OPCIONAL)
- Clasificación: Strong ≥ 0.80, Conservative ≥ 0.85, Weak < 0.80
- Filtrado: Solo historias alineadas continúan al LEVEL 3/4
"""

import os
import sys
import warnings
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# FIX Windows DLL loading
if sys.platform == "win32":
    dll_path = os.path.join(sys.prefix, "Lib", "site-packages", "numpy.libs")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
    dll_path2 = os.path.join(sys.prefix, "Lib", "site-packages", "scipy.libs")
    if os.path.exists(dll_path2):
        os.add_dll_directory(dll_path2)

import numpy as np
from sentence_transformers import util
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Umbrales según metodología experimental
UMBRAL_STRONG = 0.80
UMBRAL_CONSERVATIVE = 0.85


@dataclass
class AlignmentResult:
    """Resultado de alineación para una historia generada."""
    index_generated: int
    text_generated: str
    index_matched: int
    text_matched: str
    sbert_similarity: float
    bertscore_f1: Optional[float]
    alignment_level: str  # "strong", "conservative", "weak"
    
    @property
    def is_aligned(self) -> bool:
        """Historia está alineada si es strong o conservative."""
        return self.alignment_level in ("strong", "conservative")


class StoryAlignmentEvaluator:
    """
    LEVEL 1: Evaluador de alineación de HU.
    
    Evalúa qué tan bien las historias generadas se alinean
    con las historias esperadas usando SBERT.
    
    BERTScore se calcula como métrica informativa pero
    NO participa en la decisión de alineación.
    """
    
    def __init__(
        self,
        encoder,
        umbral_strong: float = UMBRAL_STRONG,
        umbral_conservative: float = UMBRAL_CONSERVATIVE
    ):
        self.encoder = encoder
        self.umbral_strong = umbral_strong
        self.umbral_conservative = umbral_conservative
    
    def _classify_alignment(self, sbert_similarity: float) -> str:
        """
        Clasifica nivel de alineación basado en SBERT.
        
        - conservative: ≥ 0.85
        - strong: ≥ 0.80
        - weak: < 0.80
        """
        if sbert_similarity >= self.umbral_conservative:
            return "conservative"
        elif sbert_similarity >= self.umbral_strong:
            return "strong"
        else:
            return "weak"
    
    def evaluate(
        self,
        stories_generated: List[str],
        stories_expected: List[str]
    ) -> Dict:
        """
        Evalúa alineación de historias generadas con esperadas.
        
        Args:
            stories_generated: Historias generadas.
            stories_expected: Historias esperadas (ground truth).
            
        Returns:
            Diccionario con resultados de alineación.
        """
        # LEVEL 0: Codificar ambas listas en batch (eficiente)
        emb_generated = self.encoder.encode_stories(stories_generated)  # [N_gen, 768]
        emb_expected = self.encoder.encode_stories(stories_expected)    # [N_exp, 768]
        
        # Calcular matriz de similitud completa de una vez
        similarity_matrix = util.cos_sim(emb_generated, emb_expected)  # [N_gen, N_exp]
        
        aligned_stories = []
        
        for i, story_gen in enumerate(stories_generated):
            # Obtener fila i: similitudes de historia_gen[i] con todas las esperadas
            similarities = similarity_matrix[i]
            
            # Argmax: mejor match
            sbert_idx = int(similarities.argmax().item())
            sbert_sim = float(similarities[sbert_idx].item())
            sbert_match = stories_expected[sbert_idx]
            
            # Clasificar alineación
            alignment_level = self._classify_alignment(sbert_sim)
            
            # Crear resultado
            result = AlignmentResult(
                index_generated=i,
                text_generated=story_gen,
                index_matched=sbert_idx,
                text_matched=sbert_match,
                sbert_similarity=sbert_sim,
                bertscore_f1=None,  # Opcional
                alignment_level=alignment_level
            )
            
            aligned_stories.append(result)
        
        # Calcular métricas
        aligned_count = sum(1 for a in aligned_stories if a.is_aligned)
        strong_count = sum(1 for a in aligned_stories if a.alignment_level == "strong")
        conservative_count = sum(1 for a in aligned_stories if a.alignment_level == "conservative")
        weak_count = sum(1 for a in aligned_stories if a.alignment_level == "weak")
        
        sbert_scores_aligned = [a.sbert_similarity for a in aligned_stories if a.is_aligned]
        sbert_scores_all = [a.sbert_similarity for a in aligned_stories]
        
        return {
            "aligned_stories": aligned_stories,
            "metrics": {
                "total_generated": len(stories_generated),
                "aligned_count": aligned_count,
                "strong_count": strong_count,
                "conservative_count": conservative_count,
                "weak_count": weak_count,
                "alignment_rate": aligned_count / len(stories_generated),
                "sbert_mean_aligned": np.mean(sbert_scores_aligned) if sbert_scores_aligned else 0.0,
                "sbert_mean_all": np.mean(sbert_scores_all)
            }
        }
    
    def get_aligned_pairs(self, evaluation_result: Dict) -> List[Tuple[int, int]]:
        """
        Extrae pares (gen_idx, exp_idx) de historias alineadas.
        
        Args:
            evaluation_result: Resultado de evaluate().
            
        Returns:
            Lista de tuplas (índice_generada, índice_esperada).
        """
        pairs = []
        for alignment in evaluation_result["aligned_stories"]:
            if alignment.is_aligned:
                pairs.append((alignment.index_generated, alignment.index_matched))
        return pairs
```

### 3.2 Probar LEVEL 1

```python
# test_level1.py
from pipeline.semantic_encoder import SemanticEncoder
from pipeline.story_alignment_evaluator import StoryAlignmentEvaluator

encoder = SemanticEncoder()
evaluator = StoryAlignmentEvaluator(encoder)

stories_gen = [
    "As admin, I want to register patients with data",
    "As doctor, I want to view records"
]

stories_exp = [
    "As administrative staff, I want to register patient data",
    "As physician, I want to access patient records",
    "As pharmacist, I want to manage inventory"
]

result = evaluator.evaluate(stories_gen, stories_exp)
print(f"Aligned: {result['metrics']['aligned_count']}/{result['metrics']['total_generated']}")
for alignment in result["aligned_stories"]:
    print(f"  Story {alignment.index_generated}: {alignment.alignment_level} (sim={alignment.sbert_similarity:.3f})")
```

---

## Fase 4: Implementar LEVEL 2

### 4.1 Crear `pipeline/story_coverage_calculator.py`

```python
"""
LEVEL 2 — Coverage de Historias

Responsabilidades:
- Coverage(X,Y) = expected_stories_with_match / total_expected_stories
- Diversity(X,Y) = 100 - Coverage(X,Y)

Esta métrica evalúa qué proporción de historias esperadas
tienen al menos un match en las historias generadas.
"""

import numpy as np
from sentence_transformers import util
from typing import List, Dict, Tuple

# Umbral por defecto
COVERAGE_THRESHOLD = 0.75


class StoryCoverageCalculator:
    """
    LEVEL 2: Calculador de Coverage y Diversidad de HU.
    """
    
    def __init__(self, encoder, threshold: float = COVERAGE_THRESHOLD):
        self.encoder = encoder
        self.threshold = threshold
    
    def calculate_coverage(
        self,
        stories_generated: List[str],
        stories_expected: List[str],
        threshold: float = None
    ) -> Tuple[float, Dict]:
        """
        Calcula coverage de historias esperadas.
        
        Coverage = covered_expected / total_expected
        
        Args:
            stories_generated: HU generadas.
            stories_expected: HU esperadas.
            threshold: Umbral de similitud.
            
        Returns:
            (score, detalle)
        """
        threshold = threshold or self.threshold
        
        if not stories_expected:
            return 1.0, {"covered": 0, "total": 0, "details": []}
        
        if not stories_generated:
            return 0.0, {"covered": 0, "total": len(stories_expected), "details": []}
        
        # Codificar ambas listas
        emb_gen = self.encoder.encode_stories(stories_generated)
        emb_exp = self.encoder.encode_stories(stories_expected)
        
        covered = 0
        details = []
        
        for i, emb_e in enumerate(emb_exp):
            # Máxima similitud con cualquier historia generada
            sims = util.cos_sim(emb_e, emb_gen)[0]
            max_sim = float(sims.max().item())
            is_covered = max_sim >= threshold
            
            if is_covered:
                covered += 1
                best_idx = int(sims.argmax().item())
            else:
                best_idx = -1
            
            details.append({
                "expected_idx": i,
                "expected_text": stories_expected[i],
                "max_similarity": max_sim,
                "covered": is_covered,
                "best_match_idx": best_idx,
                "best_match_text": stories_generated[best_idx] if is_covered else None
            })
        
        score = covered / len(stories_expected)
        
        return score, {
            "covered": covered,
            "total": len(stories_expected),
            "threshold": threshold,
            "details": details
        }
```

### 4.2 Probar LEVEL 2

```python
# test_level2.py
from pipeline.semantic_encoder import SemanticEncoder
from pipeline.story_coverage_calculator import StoryCoverageCalculator

encoder = SemanticEncoder()
calculator = StoryCoverageCalculator(encoder, threshold=0.75)

stories_gen = [
    "As admin, I want to register patients",
    "As doctor, I want to view records"
]

stories_exp = [
    "As administrative staff, I want to register patient data",
    "As physician, I want to access patient records",
    "As pharmacist, I want to manage inventory"  # Esta no está cubierta
]

coverage_score, detail = calculator.calculate_coverage(stories_gen, stories_exp)
print(f"Coverage: {coverage_score:.2%}")
print(f"Covered: {detail['covered']}/{detail['total']}")
```

---

## Fase 5: Implementar LEVEL 3

### 5.1 Crear `pipeline/acceptance_criteria_evaluator.py`

```python
"""
LEVEL 3 — Evaluación de Criterios de Aceptación (CA)

SOLO evalúa CA de historias alineadas.

Métricas:
- Functional Coverage
- Verificabilidad
- Ambigüedad
"""

import os
import sys
import re
from sentence_transformers import util
from typing import List, Dict, Tuple
from dataclasses import dataclass

# FIX Windows
if sys.platform == "win32":
    dll_path = os.path.join(sys.prefix, "Lib", "site-packages", "numpy.libs")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)

SCENARIO_COVERAGE_THRESHOLD = 0.75

AMBIGUOUS_TERMS = [
    "fast", "quick", "adequate", "appropriate",
    "correct", "proper", "user-friendly", "efficient",
    "intuitive", "as soon as possible", "reasonable time",
    "easily", "simple", "good", "bad", "optimal"
]

CONDITION_PATTERNS = [
    r"\bif\b", r"\bwhen\b", r"\bgiven\b",
    r"\bprovided\b", r"\bassuming\b"
]

RESULT_PATTERNS = [
    r"\bshall\b", r"\bmust\b", r"\bshould\b", r"\bthen\b",
    r"\bdisplays?\b", r"\breturns?\b", r"\bshows?\b",
    r"\bcreates?\b", r"\bupdates?\b", r"\bstores?\b"
]


@dataclass
class CriteriaEvaluationResult:
    """Resultado de evaluación de CA para un par de historias."""
    story_generated_idx: int
    story_expected_idx: int
    functional_coverage: float
    functional_coverage_details: List[Dict]
    ambiguity_score: float
    ambiguity_details: List[Dict]
    verifiability_score: float
    verifiability_details: List[Dict]


class AcceptanceCriteriaEvaluator:
    """
    LEVEL 3: Evaluador de Criterios de Aceptación.
    
    SOLO evalúa CA de historias alineadas (del LEVEL 1).
    """
    
    def __init__(
        self,
        encoder,
        coverage_threshold: float = SCENARIO_COVERAGE_THRESHOLD
    ):
        self.encoder = encoder
        self.coverage_threshold = coverage_threshold
        self.ambiguous_terms = AMBIGUOUS_TERMS
        self.condition_patterns = CONDITION_PATTERNS
        self.result_patterns = RESULT_PATTERNS
    
    def evaluate_functional_coverage(
        self,
        ca_generated: List[str],
        ca_expected: List[str]
    ) -> Tuple[float, List[Dict]]:
        """
        Evalúa cobertura funcional de CA.
        
        Args:
            ca_generated: CA generados para una historia.
            ca_expected: CA esperados para la historia match.
            
        Returns:
            (score, detalle)
        """
        if not ca_expected:
            return 1.0, []
        
        if not ca_generated:
            return 0.0, [
                {"expected": ca, "covered": False, "max_similarity": 0.0}
                for ca in ca_expected
            ]
        
        # Codificar
        emb_gen = self.encoder.encode(ca_generated, convert_to_tensor=True)
        emb_exp = self.encoder.encode(ca_expected, convert_to_tensor=True)
        
        covered = 0
        details = []
        
        for i, emb_e in enumerate(emb_exp):
            sims = util.cos_sim(emb_e, emb_gen)[0]
            max_sim = float(sims.max().item())
            is_covered = max_sim >= self.coverage_threshold
            
            if is_covered:
                covered += 1
                best_idx = int(sims.argmax().item())
                best_match = ca_generated[best_idx]
            else:
                best_idx = -1
                best_match = None
            
            details.append({
                "expected": ca_expected[i],
                "covered": is_covered,
                "max_similarity": max_sim,
                "best_match_idx": best_idx,
                "best_match": best_match
            })
        
        score = covered / len(ca_expected)
        return score, details
    
    def evaluate_verifiability(
        self,
        ca_generated: List[str]
    ) -> Tuple[float, List[Dict]]:
        """
        Evalúa verificabilidad de CA generados.
        
        Args:
            ca_generated: CA generados.
            
        Returns:
            (score, detalle)
        """
        if not ca_generated:
            return 1.0, []
        
        scores = []
        details = []
        
        for ca in ca_generated:
            has_condition = any(
                re.search(p, ca, re.IGNORECASE) 
                for p in self.condition_patterns
            )
            has_result = any(
                re.search(p, ca, re.IGNORECASE) 
                for p in self.result_patterns
            )
            
            if has_condition and has_result:
                score = 1.0
            elif has_result:
                score = 0.5
            else:
                score = 0.0
            
            scores.append(score)
            details.append({
                "criteria": ca,
                "has_condition": has_condition,
                "has_result": has_result,
                "score": score
            })
        
        avg_score = sum(scores) / len(scores)
        return avg_score, details
    
    def evaluate_ambiguity(
        self,
        ca_generated: List[str]
    ) -> Tuple[float, List[Dict]]:
        """
        Evalúa ambigüedad de CA generados.
        
        Args:
            ca_generated: CA generados.
            
        Returns:
            (ambiguity_score, detalle)
        """
        if not ca_generated:
            return 0.0, []
        
        scores = []
        details = []
        
        for ca in ca_generated:
            ca_lower = ca.lower()
            found_terms = [
                term for term in self.ambiguous_terms 
                if term in ca_lower
            ]
            
            if found_terms:
                ambiguity = 1.0  # Tiene ambigüedad
            else:
                ambiguity = 0.0  # Sin ambigüedad
            
            scores.append(ambiguity)
            details.append({
                "criteria": ca,
                "ambiguous_terms_found": found_terms,
                "ambiguity_score": ambiguity
            })
        
        avg_ambiguity = sum(scores) / len(scores)
        return avg_ambiguity, details
    
    def evaluate_aligned_stories(
        self,
        aligned_pairs: List[Tuple[int, int]],
        all_ca_generated: List[List[str]],
        all_ca_expected: List[List[str]]
    ) -> Dict:
        """
        Evalúa CA para todos los pares de historias alineadas.
        
        Args:
            aligned_pairs: Lista de tuplas (gen_idx, exp_idx).
            all_ca_generated: Lista completa de CA generados.
            all_ca_expected: Lista completa de CA esperados.
            
        Returns:
            Diccionario con resultados agregados.
        """
        results = []
        
        for gen_idx, exp_idx in aligned_pairs:
            ca_gen = all_ca_generated[gen_idx]
            ca_exp = all_ca_expected[exp_idx]
            
            coverage, coverage_details = self.evaluate_functional_coverage(ca_gen, ca_exp)
            verif, verif_details = self.evaluate_verifiability(ca_gen)
            amb, amb_details = self.evaluate_ambiguity(ca_gen)
            
            result = CriteriaEvaluationResult(
                story_generated_idx=gen_idx,
                story_expected_idx=exp_idx,
                functional_coverage=coverage,
                functional_coverage_details=coverage_details,
                ambiguity_score=amb,
                ambiguity_details=amb_details,
                verifiability_score=verif,
                verifiability_details=verif_details
            )
            
            results.append(result)
        
        # Agregación global
        avg_coverage = sum(r.functional_coverage for r in results) / len(results) if results else 0.0
        avg_verif = sum(r.verifiability_score for r in results) / len(results) if results else 0.0
        avg_amb = sum(r.ambiguity_score for r in results) / len(results) if results else 0.0
        
        return {
            "pairs_evaluated": len(results),
            "avg_functional_coverage": avg_coverage,
            "global_verifiability": avg_verif,
            "global_ambiguity": avg_amb,
            "global_no_ambiguity": 1.0 - avg_amb,
            "pairs_detail": results
        }
```

---

## Fase 6: Implementar LEVEL 4

### 6.1 Crear `pipeline/concept_coverage_evaluator.py`

```python
"""
LEVEL 4 — Coverage Conceptual

Evalúa qué proporción de conceptos del dominio
están representados en las historias alineadas.
"""

from sentence_transformers import util
from typing import List, Dict, Tuple
import numpy as np

CONCEPT_COVERAGE_THRESHOLD = 0.70


class ConceptCoverageEvaluator:
    """
    LEVEL 4: Evaluador de Cobertura Conceptual.
    
    SOLO evalúa sobre historias alineadas del LEVEL 1.
    """
    
    def __init__(self, encoder, threshold: float = CONCEPT_COVERAGE_THRESHOLD):
        self.encoder = encoder
        self.threshold = threshold
    
    def evaluate_concept_coverage(
        self,
        aligned_stories: List[str],
        concepts: Dict[str, List[str]],
        threshold: float = None
    ) -> Tuple[float, Dict]:
        """
        Evalúa cobertura de conceptos del dominio.
        
        Args:
            aligned_stories: Textos de historias alineadas.
            concepts: Dict {nombre_concepto: [descripciones]}.
            threshold: Umbral de similitud.
            
        Returns:
            (score, detalle)
        """
        threshold = threshold or self.threshold
        
        if not concepts:
            return 1.0, {"covered": 0, "total": 0, "details": []}
        
        if not aligned_stories:
            return 0.0, {
                "covered": 0,
                "total": len(concepts),
                "details": [
                    {"concept": name, "covered": False, "max_similarity": 0.0}
                    for name in concepts.keys()
                ]
            }
        
        # Codificar historias alineadas
        emb_stories = self.encoder.encode_stories(aligned_stories)
        if emb_stories.dim() == 1:
            emb_stories = emb_stories.unsqueeze(0)
        
        covered = 0
        details = []
        
        for concept_name, descriptions in concepts.items():
            # Codificar descripciones del concepto
            emb_concept = self.encoder.encode(descriptions, convert_to_tensor=True)
            if emb_concept.dim() == 1:
                emb_concept = emb_concept.unsqueeze(0)
            
            # Buscar máxima similitud
            max_sim = 0.0
            best_story_idx = -1
            best_description = None
            
            for desc_idx, emb_desc in enumerate(emb_concept):
                emb_desc_2d = emb_desc if emb_desc.dim() == 2 else emb_desc.unsqueeze(0)
                sims = util.cos_sim(emb_desc_2d, emb_stories)[0]
                sim_max = float(sims.max().item())
                
                if sim_max > max_sim:
                    max_sim = sim_max
                    best_story_idx = int(sims.argmax().item())
                    best_description = descriptions[desc_idx]
            
            is_covered = max_sim >= threshold
            if is_covered:
                covered += 1
            
            details.append({
                "concept": concept_name,
                "covered": is_covered,
                "max_similarity": max_sim,
                "best_description": best_description,
                "best_story_idx": best_story_idx,
                "best_story": aligned_stories[best_story_idx] if best_story_idx >= 0 else None
            })
        
        score = covered / len(concepts)
        
        return score, {
            "covered": covered,
            "total": len(concepts),
            "threshold": threshold,
            "details": details
        }
    
    def evaluate_aspect_coverage(
        self,
        aligned_stories: List[str],
        aspects: Dict[str, List[str]],
        threshold: float = None
    ) -> Dict:
        """
        Evalúa cobertura de aspectos funcionales.
        
        Wrapper de evaluate_concept_coverage.
        """
        score, detail = self.evaluate_concept_coverage(
            aligned_stories, aspects, threshold
        )
        
        return {
            "concept_coverage_score": score,
            "concepts_covered": detail["covered"],
            "concepts_total": detail["total"],
            "threshold": detail["threshold"],
            "details": detail["details"]
        }
```

---

## Fase 7: Crear el Pipeline Orquestador

### 7.1 Crear `pipeline/__init__.py`

```python
"""
Pipeline de evaluación metodológica
LEVEL 0 → LEVEL 4
"""

from .semantic_encoder import SemanticEncoder
from .story_alignment_evaluator import StoryAlignmentEvaluator
from .story_coverage_calculator import StoryCoverageCalculator
from .acceptance_criteria_evaluator import AcceptanceCriteriaEvaluator
from .concept_coverage_evaluator import ConceptCoverageEvaluator
from .evaluation_pipeline import EvaluationPipeline, EvaluationInput, EvaluationOutput

__all__ = [
    "SemanticEncoder",
    "StoryAlignmentEvaluator",
    "StoryCoverageCalculator",
    "AcceptanceCriteriaEvaluator",
    "ConceptCoverageEvaluator",
    "EvaluationPipeline",
    "EvaluationInput",
    "EvaluationOutput",
]
```

### 7.2 Crear `pipeline/evaluation_pipeline.py`

```python
"""
EvaluationPipeline — Orquestador LEVEL 0 → LEVEL 4

Ejecuta el pipeline completo de evaluación.
"""

from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

from .semantic_encoder import SemanticEncoder
from .story_alignment_evaluator import StoryAlignmentEvaluator
from .story_coverage_calculator import StoryCoverageCalculator
from .acceptance_criteria_evaluator import AcceptanceCriteriaEvaluator
from .concept_coverage_evaluator import ConceptCoverageEvaluator


@dataclass
class EvaluationInput:
    """Datos de entrada para el pipeline."""
    stories_generated: List[str]
    stories_expected: List[str]
    ca_generated: List[List[str]] = field(default_factory=list)
    ca_expected: List[List[str]] = field(default_factory=list)
    domain_concepts: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


@dataclass 
class EvaluationOutput:
    """Resultado completo del pipeline."""
    timestamp: str
    metadata: Dict
    config: Dict
    level_1_alignment: Dict
    level_2_coverage: Dict
    level_3_criteria: Optional[Dict]
    level_4_concepts: Optional[Dict]
    summary: Dict


class EvaluationPipeline:
    """
    Orquestador del pipeline de evaluación.
    
    Ejecuta los 5 niveles en secuencia correcta.
    """
    
    def __init__(
        self,
        sbert_model: str = None,
        device: str = None,
        umbral_strong: float = 0.80,
        umbral_conservative: float = 0.85,
        coverage_threshold: float = 0.75,
        ca_coverage_threshold: float = 0.75,
        concept_threshold: float = 0.70
    ):
        # LEVEL 0: Encoder
        self.encoder = SemanticEncoder(
            model_name=sbert_model,
            device=device
        ) if sbert_model else SemanticEncoder(device=device)
        
        # LEVEL 1: Alineación
        self.alignment_evaluator = StoryAlignmentEvaluator(
            encoder=self.encoder,
            umbral_strong=umbral_strong,
            umbral_conservative=umbral_conservative
        )
        
        # LEVEL 2: Coverage
        self.coverage_calculator = StoryCoverageCalculator(
            encoder=self.encoder,
            threshold=coverage_threshold
        )
        
        # LEVEL 3: CA
        self.criteria_evaluator = AcceptanceCriteriaEvaluator(
            encoder=self.encoder,
            coverage_threshold=ca_coverage_threshold
        )
        
        # LEVEL 4: Concepts
        self.concept_evaluator = ConceptCoverageEvaluator(
            encoder=self.encoder,
            threshold=concept_threshold
        )
        
        self._config = {
            "umbral_strong": umbral_strong,
            "umbral_conservative": umbral_conservative,
            "coverage_threshold": coverage_threshold,
            "ca_coverage_threshold": ca_coverage_threshold,
            "concept_threshold": concept_threshold,
            "encoder": self.encoder.get_config()
        }
    
    def run(self, input_data: EvaluationInput) -> EvaluationOutput:
        """
        Ejecuta pipeline completo LEVEL 0 → LEVEL 4.
        
        Args:
            input_data: Datos de entrada.
            
        Returns:
            EvaluationOutput con resultados de todos los niveles.
        """
        timestamp = datetime.now().isoformat()
        
        # LEVEL 1: Alineación
        level_1_result = self.alignment_evaluator.evaluate(
            stories_generated=input_data.stories_generated,
            stories_expected=input_data.stories_expected
        )
        
        # Extraer pares alineados
        aligned_pairs = self.alignment_evaluator.get_aligned_pairs(level_1_result)
        aligned_stories_text = [
            a.text_generated for a in level_1_result["aligned_stories"]
            if a.is_aligned
        ]
        
        # LEVEL 2: Coverage
        coverage_score, coverage_detail = self.coverage_calculator.calculate_coverage(
            stories_generated=input_data.stories_generated,
            stories_expected=input_data.stories_expected
        )
        
        level_2_result = {
            "story_coverage_score": coverage_score,
            "diversity_complement": 100.0 * (1.0 - coverage_score),
            **coverage_detail
        }
        
        # LEVEL 3: CA (solo alineadas)
        level_3_result = None
        if input_data.ca_generated and input_data.ca_expected and aligned_pairs:
            level_3_result = self.criteria_evaluator.evaluate_aligned_stories(
                aligned_pairs=aligned_pairs,
                all_ca_generated=input_data.ca_generated,
                all_ca_expected=input_data.ca_expected
            )
        
        # LEVEL 4: Concepts (solo alineadas)
        level_4_result = None
        if input_data.domain_concepts and aligned_stories_text:
            level_4_result = self.concept_evaluator.evaluate_aspect_coverage(
                aligned_stories=aligned_stories_text,
                aspects=input_data.domain_concepts
            )
        
        # Resumen
        summary = self._build_summary(
            input_data, level_1_result, level_2_result,
            level_3_result, level_4_result
        )
        
        # Serializar para JSON
        level_1_serializable = self._serialize_level_1(level_1_result)
        
        return EvaluationOutput(
            timestamp=timestamp,
            metadata=input_data.metadata,
            config=self._config,
            level_1_alignment=level_1_serializable,
            level_2_coverage=level_2_result,
            level_3_criteria=level_3_result,
            level_4_concepts=level_4_result,
            summary=summary
        )
    
    def _build_summary(self, input_data, l1, l2, l3, l4) -> Dict:
        """Construye resumen ejecutivo."""
        summary = {
            "input_stats": {
                "stories_generated": len(input_data.stories_generated),
                "stories_expected": len(input_data.stories_expected),
                "stories_aligned": l1["metrics"]["aligned_count"],
                "alignment_rate": l1["metrics"]["alignment_rate"]
            },
            "level_1_alignment": l1["metrics"],
            "level_2_coverage": {
                "story_coverage": l2["story_coverage_score"],
                "stories_covered": l2["covered"],
                "stories_total": l2["total"]
            }
        }
        
        if l3:
            summary["level_3_criteria"] = {
                "pairs_evaluated": l3["pairs_evaluated"],
                "avg_functional_coverage": l3["avg_functional_coverage"],
                "global_verifiability": l3["global_verifiability"],
                "global_no_ambiguity": l3["global_no_ambiguity"],
                "global_ambiguity": l3["global_ambiguity"]
            }
        
        if l4:
            summary["level_4_concepts"] = {
                "concept_coverage": l4["concept_coverage_score"],
                "concepts_covered": l4["concepts_covered"],
                "concepts_total": l4["concepts_total"]
            }
        
        return summary
    
    def _serialize_level_1(self, level_1_result: Dict) -> Dict:
        """Serializa resultados de LEVEL 1 para JSON."""
        aligned_stories_list = []
        for alignment in level_1_result["aligned_stories"]:
            aligned_stories_list.append({
                "index_generated": alignment.index_generated,
                "text_generated": alignment.text_generated,
                "index_matched": alignment.index_matched,
                "text_matched": alignment.text_matched,
                "sbert_similarity": alignment.sbert_similarity,
                "bertscore_f1": alignment.bertscore_f1,
                "alignment_level": alignment.alignment_level,
                "is_aligned": alignment.is_aligned
            })
        
        return {
            "aligned_stories": aligned_stories_list,
            "metrics": level_1_result["metrics"]
        }
    
    def to_dict(self, output: EvaluationOutput) -> Dict:
        """Convierte EvaluationOutput a diccionario."""
        return asdict(output)
```

---

## Fase 8: Crear Scripts de Ejecución

### 8.1 Crear `cargador_datos.py`

```python
"""
Utilidades para cargar datos de casos de prueba.
"""

import os
import json
from typing import List, Dict


def cargar_historias(ruta: str) -> List[str]:
    """Carga historias desde archivo .txt (una por línea)."""
    with open(ruta, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
    return [linea.strip() for linea in lineas if linea.strip()]


def cargar_aspectos(ruta: str) -> Dict[str, List[str]]:
    """Carga aspectos desde archivo .json."""
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)


def cargar_criterios(ruta: str) -> List[List[str]]:
    """Carga criterios de aceptación desde archivo .json."""
    with open(ruta, 'r', encoding='utf-8') as f:
        return json.load(f)


def cargar_metadata(dir_caso: str) -> Dict:
    """Carga metadata.json de un caso."""
    ruta_metadata = os.path.join(dir_caso, "metadata.json")
    
    if not os.path.exists(ruta_metadata):
        raise FileNotFoundError(f"metadata.json no encontrado en: {dir_caso}")
    
    with open(ruta_metadata, 'r', encoding='utf-8') as f:
        return json.load(f)


def descubrir_casos(dir_casos: str) -> List[str]:
    """Descubre todos los casos en el directorio."""
    if not os.path.isdir(dir_casos):
        return []
    
    casos = []
    for item in os.listdir(dir_casos):
        ruta_item = os.path.join(dir_casos, item)
        if os.path.isdir(ruta_item) and item.startswith("caso_"):
            casos.append(item)
    
    return sorted(casos)


def guardar_json(resultados: Dict, ruta: str):
    """Guarda resultados en archivo JSON."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
```

### 8.2 Crear `ejecutar_pipeline.py`

```python
"""
Ejecutor de casos usando el pipeline refactorizado.

Uso:
    python ejecutar_pipeline.py --caso caso_A_alineadas
    python ejecutar_pipeline.py --caso todos
"""

import os
import argparse
from datetime import datetime

from pipeline import EvaluationPipeline, EvaluationInput
from cargador_datos import (
    cargar_historias,
    cargar_aspectos,
    cargar_metadata,
    cargar_criterios,
    descubrir_casos,
    guardar_json
)


def timestamp():
    return datetime.now().isoformat()


def ejecutar_caso(
    nombre_caso: str,
    dir_casos: str,
    archivo_aspectos: str,
    pipeline: EvaluationPipeline,
    dir_salida: str
) -> dict:
    """Ejecuta evaluación completa para un caso."""
    dir_caso = os.path.join(dir_casos, nombre_caso)
    
    print(f"\n{'='*50}")
    print(f"Caso: {nombre_caso}")
    print(f"{'='*50}")
    
    # Cargar datos
    metadata = cargar_metadata(dir_caso)
    print(f"Descripción: {metadata['descripcion']}")
    print(f"Dominio: {metadata['dominio']}")
    
    ruta_generadas = os.path.join(dir_caso, "generadas.txt")
    historias_generadas = cargar_historias(ruta_generadas)
    print(f"Historias generadas: {len(historias_generadas)}")
    
    ruta_esperadas = os.path.join(dir_caso, "esperadas.txt")
    if not os.path.exists(ruta_esperadas):
        print("Sin historias esperadas - abortando caso")
        return None
    
    historias_esperadas = cargar_historias(ruta_esperadas)
    print(f"Historias esperadas: {len(historias_esperadas)}")
    
    # CA
    ruta_ca_gen = os.path.join(dir_caso, "ca_generados.json")
    ruta_ca_exp = os.path.join(dir_caso, "ca_esperados.json")
    
    criterios_generados = []
    criterios_esperados = []
    
    if os.path.exists(ruta_ca_gen):
        criterios_generados = cargar_criterios(ruta_ca_gen)
        print(f"CA generados: {len(criterios_generados)} historias")
    
    if os.path.exists(ruta_ca_exp):
        criterios_esperados = cargar_criterios(ruta_ca_exp)
        print(f"CA esperados: {len(criterios_esperados)} historias")
    
    # Aspectos
    aspectos = cargar_aspectos(archivo_aspectos)
    print(f"Aspectos: {len(aspectos)}")
    
    # Preparar input
    input_data = EvaluationInput(
        stories_generated=historias_generadas,
        stories_expected=historias_esperadas,
        ca_generated=criterios_generados,
        ca_expected=criterios_esperados,
        domain_concepts=aspectos,
        metadata=metadata
    )
    
    # Ejecutar pipeline
    print("\nEjecutando pipeline de evaluación...")
    resultado = pipeline.run(input_data)
    
    # Mostrar resumen
    print("\n--- RESUMEN ---")
    summary = resultado.summary
    
    print(f"\nLEVEL 1 - Alineación:")
    print(f"  Alineadas: {summary['input_stats']['stories_aligned']}/{summary['input_stats']['stories_generated']}")
    print(f"  Tasa: {summary['input_stats']['alignment_rate']:.2%}")
    
    print(f"\nLEVEL 2 - Coverage:")
    print(f"  Coverage: {summary['level_2_coverage']['story_coverage']:.2%}")
    print(f"  Cubiertas: {summary['level_2_coverage']['stories_covered']}/{summary['level_2_coverage']['stories_total']}")
    
    if "level_3_criteria" in summary:
        l3 = summary["level_3_criteria"]
        print(f"\nLEVEL 3 - CA:")
        print(f"  Cobertura funcional: {l3['avg_functional_coverage']:.2%}")
        print(f"  Verificabilidad: {l3['global_verifiability']:.2%}")
        print(f"  No-ambigüedad: {l3['global_no_ambiguity']:.2%}")
    
    if "level_4_concepts" in summary:
        l4 = summary["level_4_concepts"]
        print(f"\nLEVEL 4 - Conceptos:")
        print(f"  Coverage: {l4['concept_coverage']:.2%}")
        print(f"  Cubiertos: {l4['concepts_covered']}/{l4['concepts_total']}")
    
    # Guardar resultados
    ts = timestamp()
    ts_safe = ts.replace(":", "-").replace(".", "-")
    
    resultado_dict = pipeline.to_dict(resultado)
    resultado_dict["caso"] = nombre_caso
    resultado_dict["num_generadas"] = len(historias_generadas)
    resultado_dict["num_esperadas"] = len(historias_esperadas)
    
    ruta_salida = os.path.join(dir_salida, f"{nombre_caso}_{ts_safe}.json")
    guardar_json(resultado_dict, ruta_salida)
    
    print(f"\n✓ Resultados guardados en: {ruta_salida}")
    
    return resultado_dict


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutor de pipeline de evaluación"
    )
    parser.add_argument(
        "--caso",
        required=True,
        help="Nombre del caso (ej: caso_A_alineadas) o 'todos'"
    )
    parser.add_argument(
        "--dir-casos",
        default="casos_prueba",
        help="Directorio de casos de prueba"
    )
    parser.add_argument(
        "--dir-salida",
        default="resultados",
        help="Directorio de resultados"
    )
    parser.add_argument(
        "--aspectos",
        default="casos_prueba/aspectos_hospital.json",
        help="Archivo de aspectos del dominio"
    )
    
    args = parser.parse_args()
    
    # Crear pipeline
    pipeline = EvaluationPipeline()
    
    # Ejecutar caso(s)
    if args.caso == "todos":
        casos = descubrir_casos(args.dir_casos)
        print(f"Encontrados {len(casos)} casos")
        for caso in casos:
            ejecutar_caso(caso, args.dir_casos, args.aspectos, pipeline, args.dir_salida)
    else:
        ejecutar_caso(args.caso, args.dir_casos, args.aspectos, pipeline, args.dir_salida)


if __name__ == "__main__":
    main()
```

---

## Fase 9: Preparar Casos de Prueba

### 9.1 Crear Caso de Prueba Ejemplo

Crear estructura:
```
casos_prueba/
└── caso_A_alineadas/
    ├── metadata.json
    ├── esperadas.txt
    ├── generadas.txt
    ├── ca_esperados.json
    ├── ca_generados.json
```

### 9.2 `metadata.json`

```json
{
    "descripcion": "Historias alineadas - dominio hospitalario",
    "dominio": "hospital",
    "archivo_aspectos": "aspectos_hospital.json",
    "evaluar_alineacion": true,
    "evaluar_completitud": true,
    "evaluar_consistencia": true
}
```

### 9.3 `esperadas.txt`

```
As administrative staff, I want to register the data of a new patient to admit them into the system.
As administrative staff, I want to edit the data of a registered patient to keep the information up to date.
As a physician, I want to access the patient's electronic health record to review their previous information.
```

(Una historia por línea)

### 9.4 `generadas.txt`

```
As an administrative staff member, I want to register patients with personal data and medical background.
As an administrative staff member, I want to modify the personal and contact details of a registered patient.
As a doctor, I want to register diagnoses, treatments, and patient's clinical progress in the electronic clinical record.
```

### 9.5 `ca_esperados.json`

```json
[
  [
    "The system must request full name of the patient",
    "The system must request date of birth of the patient",
    "The system must generate a unique patient identifier"
  ],
  [
    "The system must allow editing the full name of a registered patient",
    "The system must allow editing patient contact information"
  ],
  [
    "The system must allow recording diagnoses",
    "The system must allow recording treatments"
  ]
]
```

### 9.6 `ca_generados.json`

```json
[
  [
    "The system allows entry of personal and medical background data",
    "Each patient receives a unique identifier"
  ],
  [
    "The system allows editing of existing patient records"
  ],
  [
    "Diagnoses and clinical progress can be documented"
  ]
]
```

### 9.7 `aspectos_hospital.json` (en casos_prueba/)

```json
{
    "pacientes": [
        "registrar pacientes con datos personales",
        "editar informacion de pacientes",
        "buscar pacientes por identificador"
    ],
    "historia_clinica": [
        "acceder a historia clinica electronica",
        "registrar diagnosticos y tratamientos"
    ]
}
```

---

## Fase 10: Ejecutar y Validar

### 10.1 Ejecutar Pipeline

```bash
# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Ejecutar un caso
python ejecutar_pipeline.py --caso caso_A_alineadas

# Ejecutar todos los casos
python ejecutar_pipeline.py --caso todos
```

### 10.2 Verificar Resultados

Revisar archivo en `resultados/caso_A_alineadas_2026-XX-XX....json`:

```json
{
  "timestamp": "2026-XX-XXTXX:XX:XX",
  "caso": "caso_A_alineadas",
  "summary": {
    "input_stats": {
      "stories_generated": 3,
      "stories_expected": 3,
      "stories_aligned": 3,
      "alignment_rate": 1.0
    },
    "level_1_alignment": {
      "sbert_mean_aligned": 0.85,
      ...
    },
    "level_2_coverage": {
      "story_coverage": 1.0,
      ...
    }
  }
}
```

### 10.3 Validación de Resultados

**Verificar**:
1. ✓ Todas las historias tienen similitud SBERT calculada
2. ✓ Las alineadas son ≥0.80 (Strong) o ≥0.85 (Conservative)
3. ✓ LEVEL 2 muestra coverage esperado
4. ✓ LEVEL 3 solo evalúa historias alineadas
5. ✓ LEVEL 4 solo evalúa historias alineadas
6. ✓ Los resultados son consistentes entre niveles

---

## 🔧 Troubleshooting Común

### Error: "DLL load failed"
**Windows**: Instalar Visual C++ Redistributables 2015-2022
**Solución temporal**: Ya incluido en semantic_encoder.py (add_dll_directory)

### Error: "Model not found"
**Causa**: Primera ejecución descarga modelos (~500MB)
**Solución**: Esperar descarga automática o descargar manualmente

### Error: "CUDA out of memory"
**Solución**: Usar CPU en lugar de GPU:
```python
encoder = SemanticEncoder(device="cpu")
```

### Similitudes todas muy bajas
**Causa**: Historias en idiomas diferentes o muy distintas
**Solución**: Verificar que generadas y esperadas son del mismo dominio

---

## 📊 Extensiones Posibles

### 1. Agregar BERTScore Real

```python
# En story_alignment_evaluator.py
from bert_score import score as bert_score

bertscore_f1 = bert_score(
    cands=[text_generated],
    refs=[text_matched],
    lang="en",
    model_type="microsoft/deberta-xlarge-mnli",
    device=self.encoder.device,
    verbose=False
)[2].item()
```

### 2. Comparación Multi-Modelo

Crear `ejecutar_diversidad.py` para comparar outputs de diferentes LLMs:
```python
from pipeline.story_coverage_calculator import StoryCoverageCalculator

calculator = StoryCoverageCalculator(encoder)
diversity = calculator.calculate_diversity(stories_gpt4, stories_claude)
```

### 3. Visualizaciones

Agregar gráficos de métricas con matplotlib:
```python
import matplotlib.pyplot as plt

def plot_alignment_distribution(sbert_scores):
    plt.hist(sbert_scores, bins=20)
    plt.xlabel("SBERT Similarity")
    plt.ylabel("Frequency")
    plt.title("Distribution of Alignment Scores")
    plt.savefig("alignment_dist.png")
```

---

## ✅ Checklist de Implementación Completa

- [ ] Fase 0: Entorno virtual creado y dependencias instaladas
- [ ] Fase 1: Estructura de directorios creada
- [ ] Fase 2: LEVEL 0 implementado y probado
- [ ] Fase 3: LEVEL 1 implementado y probado
- [ ] Fase 4: LEVEL 2 implementado y probado
- [ ] Fase 5: LEVEL 3 implementado y probado
- [ ] Fase 6: LEVEL 4 implementado y probado
- [ ] Fase 7: Pipeline orquestador creado
- [ ] Fase 8: Scripts de ejecución creados
- [ ] Fase 9: Al menos un caso de prueba preparado
- [ ] Fase 10: Pipeline ejecutado exitosamente y resultados validados

---

## 📚 Referencias

- **SBERT**: [https://www.sbert.net/](https://www.sbert.net/)
- **BERTScore**: [https://github.com/Tiiiger/bert_score](https://github.com/Tiiiger/bert_score)
- **Sentence-Transformers**: [https://www.sbert.net/docs/pretrained_models.html](https://www.sbert.net/docs/pretrained_models.html)

---

## 📄 Licencia y Contacto

Proyecto desarrollado para evaluación de historias de usuario generadas automáticamente.

**Fecha de creación**: Febrero 2026
**Versión**: 1.0

---

**FIN DEL DOCUMENTO**
