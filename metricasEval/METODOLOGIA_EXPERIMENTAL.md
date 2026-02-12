# Metodología Experimental - Evaluación de Historias de Usuario

## 🎯 Objetivo General

Evaluar la calidad de historias de usuario generadas automáticamente comparándolas con historias esperadas (ground truth) usando una metodología de evaluación jerárquica en 5 niveles.

---

## 🏗️ Arquitectura General

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

---

## 📊 LEVEL 0: Semantic Encoder

**Archivo**: `pipeline/semantic_encoder.py`

### ¿Qué hace?
Transforma texto en lenguaje natural a vectores numéricos (embeddings de 768 dimensiones) para poder calcular similitud semántica entre textos.

### Configuración
- **Modelo**: `sentence-transformers/all-mpnet-base-v2`
- **Dimensión**: 768 valores numéricos por texto
- **Device**: CPU (con fix para Windows DLL loading)

### ¿Qué textos se codifican EXACTAMENTE?

En el caso `caso_A_alineadas`:

#### 1. Historias generadas (23 textos)
```python
# Ejemplo de las primeras 2 historias generadas:
stories_gen = [
  "As an administrative staff member, I want to register patients with personal data and medical background, so that the hospital can maintain accurate and complete patient records.",
  "As an administrative staff member, I want to modify the personal and contact details of a registered patient, so that patient information remains up to date.",
  # ... 21 más
]
# Se codifica → tensor shape [23, 768]
```

#### 2. Historias esperadas (28 textos)
```python
# Ejemplo de las primeras 2 historias esperadas:
stories_exp = [
  "As administrative staff, I want to register the data of a new patient to admit them into the system.",
  "As administrative staff, I want to edit the data of a registered patient to keep the information up to date.",
  # ... 26 más  
]
# Se codifica → tensor shape [28, 768]
```

#### 3. Criterios de aceptación generados (suma de todos los CA)
```python
# Ejemplo: CA de las primeras 2 historias generadas
ca_gen = [
  ["The system allows entry of personal and medical background data",    # CA de historia 0
   "Each patient receives a unique identifier",
   "Duplicate identity documents or identifiers are not allowed"],
  ["The system allows editing of existing patient records"],              # CA de historia 1
  # ... más
]
# Se aplanan y codifican TODOS los CA → tensor shape [n_criterios, 768]
```

#### 4. Criterios de aceptación esperados (suma de todos los CA)
```python
# Estructura idéntica a ca_gen
# Se aplanan y codifican TODOS los CA → tensor shape [m_criterios, 768]
```

#### 5. Descripciones de conceptos del dominio
```python
# Del archivo aspectosHospital2.json:
concepts = {
  "patients": "patient registration and management",
  "appointments": "appointments and consultation scheduling",
  "medical_records": "electronic medical records that patients can consult, but not modify",
  # ... más conceptos
}
# Se codifican las 7 descripciones → tensor shape [7, 768]
```

### Métodos principales
```python
encode(texts: List[str]) → torch.Tensor
  # Transforma cada texto en un vector de 768 números
  # Input:  ["historia 1", "historia 2", ...]
  # Output: tensor shape [n, 768]
  # Ejemplo: ["register patients", "edit patient"] → [[0.12, -0.34, ..., 0.56], [0.08, -0.29, ..., 0.44]]
```

### Uso de embeddings
Una vez codificados, se calculan similitudes usando **cosine similarity**:
```python
from sentence_transformers import util

# Similitud entre 2 textos
sim = util.cos_sim(embedding1, embedding2)  # Resultado: 0.0 a 1.0

# Matriz de similitudes (23 generadas vs 28 esperadas)
matriz = util.cos_sim(emb_generadas, emb_esperadas)  # Shape: [23, 28]
# matriz[i][j] = similitud entre historia_gen[i] y historia_esp[j]
```

---

## 🎯 LEVEL 1: Story Alignment Evaluator

**Archivo**: `pipeline/story_alignment_evaluator.py`

### ¿Qué hace?
Alinea cada historia generada con su mejor match en las esperadas usando **SBERT exclusivamente** para decidir el matching y clasificación.

### Input EXACTO (caso_A_alineadas)
```python
stories_generated = [
  "As an administrative staff member, I want to register patients with personal data...",
  "As an administrative staff member, I want to modify the personal and contact details...",
  # ... 21 más (total: 23)
]

stories_expected = [
  "As administrative staff, I want to register the data of a new patient...",
  "As administrative staff, I want to edit the data of a registered patient...",
  # ... 26 más (total: 28)
]
```

### Proceso PASO A PASO (ejemplo con historia generada #0)

#### Paso 1: Codificar con SBERT
```python
emb_gen = encoder.encode(stories_generated)    # [23, 768]
emb_exp = encoder.encode(stories_expected)     # [28, 768]
```

#### Paso 2: Matriz de similitudes
```python
matriz_sim = util.cos_sim(emb_gen, emb_exp)   # [23, 28]

# Para historia generada #0:
similitudes_historia_0 = matriz_sim[0]  # Vector de 28 valores
# [0.8596, 0.6234, 0.5421, 0.7012, ..., 0.3421]
#  ↑ match con esperada #0
```

#### Paso 3: Argmax (mejor match)
```python
best_match_idx = torch.argmax(similitudes_historia_0)  # → 0
best_similarity = similitudes_historia_0[best_match_idx]  # → 0.8596

# Resultado:
# Historia generada #0 ↔ Historia esperada #0
# Similitud SBERT: 0.8596
```

#### Paso 4: Clasificación por umbral
```python
if best_similarity >= 0.85:
    nivel = "conservative"  # ✓ PASA
elif best_similarity >= 0.80:
    nivel = "strong"        # ✓ PASA
else:
    nivel = "weak"          # ✗ FILTRADA

# Historia #0: sim=0.8596 → "conservative" → PASA
```

#### Paso 5: BERTScore (OPCIONAL, secundario)
```python
# Se calcula SOLO entre la generada y su match SBERT
bertscore_f1 = bert_score(
    cands=["historia generada #0"],
    refs=["historia esperada #0 (match SBERT)"],
    model="microsoft/deberta-xlarge-mnli"
)
# Resultado: 0.91 (o null si no disponible)
# ⚠️ Este valor NO afecta la alineación ni el filtrado
```

### Flujo de ejecución

```
23 Historias Generadas  ──┐
                          │
28 Historias Esperadas   ─┴─→ [Matriz SBERT 23x28]
                                      │
                                      ▼
                          Para cada generada: argmax(fila)
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │ Clasificación umbral  │
                          │ SBERT (NO BERTScore)  │
                          └───────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            Conservative (≥0.85)  Strong (≥0.80)    Weak (<0.80)
            PASA → LEVEL 3/4      PASA → LEVEL 3/4  FILTRADA
```

### Ejemplo REAL: Historia generada #7

**Texto generado**:
> "As a patient, I want to receive notifications when my appointment is modified or canceled, so that I am always informed about my schedule."

**Proceso**:
1. Se codifica → embedding [768 valores]
2. Se compara con las 28 esperadas → 28 similitudes SBERT
3. Mejor match: esperada #7 con similitud **0.9751**
4. Clasificación: 0.9751 ≥ 0.85 → **"conservative"**
5. BERTScore (opcional): null (no calculado)
6. **Resultado: ✓ ALINEADA** → pasa a LEVEL 3 y 4

**Texto esperado matched**:
> "As a patient, I want to be notified when my appointment is modified or canceled to stay informed."

### Umbrales clave (configurables)
```python
UMBRAL_STRONG       = 0.80  # Alineación fuerte
UMBRAL_CONSERVATIVE = 0.85  # Alineación conservadora (más estricta)
# Weak: sim < 0.80 → NO PASA a niveles siguientes
```

**Interpretación**:
- **Conservative** (≥0.85): historias casi idénticas semánticamente
- **Strong** (≥0.80): historias claramente relacionadas
- **Weak** (<0.80): historias diferentes o poco relacionadas → **FILTRADAS**

### BERTScore (secundario y opcional)
- **Función**: validación adicional basada en contexto
- **Modelo**: `microsoft/deberta-xlarge-mnli`
- **Cuándo se calcula**: SOLO contra el match elegido por SBERT
- **Cuándo NO se calcula**: si el modelo no está disponible o falla la carga
- **Decisión**: NUNCA decide alineación, matching ni filtrado
- **Resultado si no disponible**: `bertscore_f1 = null`

**Ejemplo**:
```json
{
  "index_generated": 7,
  "index_matched": 7,        // ← Decidido por SBERT
  "sbert_similarity": 0.9751, // ← Decide clasificación
  "bertscore_f1": null,       // ← Informativo, no afecta nada
  "alignment_level": "conservative", // ← Basado en SBERT
  "is_aligned": true         // ← Basado en SBERT
}
```

### Output REAL (caso_A_alineadas)

#### Resultado completo
De **23 historias generadas**:
- **10 alineadas** (Strong + Conservative) → pasan a LEVEL 3/4
  - 2 Strong (≥0.80)
  - 8 Conservative (≥0.85)
- **13 filtradas** (Weak < 0.80) → no continúan

#### Ejemplo de alineaciones exitosas
```json
{
  "aligned_stories": [
    {
      "index_generated": 0,
      "index_matched": 0,
      "sbert_similarity": 0.8596,
      "alignment_level": "conservative"
    },
    {
      "index_generated": 2,
      "index_matched": 3,
      "sbert_similarity": 0.8135,
      "alignment_level": "strong"
    },
    {
      "index_generated": 7,
      "index_matched": 7,
      "sbert_similarity": 0.9751,  // ← Muy alta similitud
      "alignment_level": "conservative"
    }
    // ... 7 más
  ]
}
```

#### Ejemplo de historia filtrada (weak)
```json
{
  "index_generated": 3,
  "text_generated": "As an authorized user, I want to visualize a patient's complete information...",
  "index_matched": 8,  // Mejor match encontrado
  "text_matched": "As a physician, I want to access the patient's electronic health record...",
  "sbert_similarity": 0.6331,  // ← Muy baja
  "alignment_level": "weak",
  "is_aligned": false  // ← NO PASA a LEVEL 3/4
}
```

---

## 📈 LEVEL 2: Story Coverage Calculator

**Archivo**: `pipeline/story_coverage_calculator.py`

### ¿Qué hace?
Calcula qué porcentaje de historias esperadas (ground truth) están cubiertas por las historias generadas, sin importar si están alineadas o no.

**⚠️ IMPORTANTE**: Este nivel usa TODAS las 23 historias generadas, no solo las 10 alineadas.

### Input EXACTO (caso_A_alineadas)
```python
# Usa todas, incluyendo las weak filtradas en LEVEL 1
stories_generated = 23 historias
stories_expected = 28 historias
threshold = 0.75  # Umbral para considerar "cubierta"
```

### Fórmula
```python
Coverage(Gen, Esp) = |{e ∈ Esp : max_sim(e, Gen) ≥ 0.75}| / |Esp|

# En español:
# Para cada historia esperada:
#   Encuentra la similitud máxima con cualquier historia generada
#   Si max_similitud ≥ 0.75 → esperada está "cubierta"
# Score = esperadas_cubiertas / total_esperadas
```

### Proceso PASO A PASO (ejemplo con historia esperada #0)

#### Historia esperada #0:
> "As administrative staff, I want to register the data of a new patient to admit them into the system."

**Paso 1**: Codificar con SBERT
```python
emb_esperada_0 = encoder.encode([historia_esperada_0])  # [1, 768]
emb_todas_gen = encoder.encode(stories_generated)       # [23, 768]
```

**Paso 2**: Calcular similitud con TODAS las generadas
```python
similitudes = util.cos_sim(emb_esperada_0, emb_todas_gen)[0]
# [0.8596, 0.6421, 0.5234, ..., 0.3912]  # 23 valores
```

**Paso 3**: Encontrar máximo
```python
max_sim = max(similitudes)  # → 0.8596
best_match_idx = 0  # Historia generada #0
best_match_text = "As an administrative staff member, I want to register patients with personal data..."
```

**Paso 4**: Verificar umbral
```python
if max_sim >= 0.75:
    cubierta = True
else:
    cubierta = False

# Historia esperada #0: 0.8596 ≥ 0.75 → ✓ CUBIERTA
```

### Resultado REAL (caso_A_alineadas)

De **28 historias esperadas**:
- **16 cubiertas** (similitud máxima ≥ 0.75)
- **12 no cubiertas** (similitud máxima < 0.75)
- **Coverage score**: 16/28 = **57.14%**

#### Ejemplos de historias esperadas CUBIERTAS
```json
{
  "expected_idx": 0,
  "expected_text": "As administrative staff, I want to register the data of a new patient...",
  "max_similarity": 0.8596,  // ← Por encima del umbral
  "covered": true,
  "best_match_idx": 0,
  "best_match_text": "As an administrative staff member, I want to register patients..."
},
{
  "expected_idx": 7,
  "expected_text": "As a patient, I want to be notified when my appointment is modified...",
  "max_similarity": 0.9751,  // ← Muy alta similitud
  "covered": true,
  "best_match_idx": 7
}
```

#### Ejemplos de historias esperadas NO CUBIERTAS
```json
{
  "expected_idx": 9,
  "expected_text": "As a physician, I want to record the diagnosis made during the appointment...",
  "max_similarity": 0.7032,  // ← Por debajo del umbral 0.75
  "covered": false,
  "best_match_idx": -1,
  "best_match_text": null
},
{
  "expected_idx": 12,
  "expected_text": "As administrative staff, I want to deactivate a registered physician...",
  "max_similarity": 0.6037,  // ← Ninguna historia generada cubre esto
  "covered": false,
  "best_match_idx": -1
}
```

### Diversity (métrica complementaria)
```python
Diversity = 100 - Coverage(Esp, Gen)
# O también: Coverage inverso (Gen vs Esp)

# Para caso_A_alineadas:
Diversity = 100 - 57.14 = 42.86%
```

**Interpretación**:
- **Coverage alto** (>70%): el modelo genera las historias esperadas
- **Diversity alto** (>40%): el modelo genera historias adicionales no esperadas
  - Puede ser BUENO: cubre casos no previstos
  - Puede ser MALO: genera historias irrelevantes

### Diferencia con LEVEL 1
| Aspecto | LEVEL 1 (Alignment) | LEVEL 2 (Coverage) |
|---------|---------------------|-------------------|
| Dirección | Gen → Esp | Esp → Gen |
| Objetivo | Clasificar generadas | Cubrir esperadas |
| Filtro | Sí (weak rechazadas) | No (usa todas) |
| Umbral | 0.80/0.85 | 0.75 |
| Usa para LEVEL 3/4 | Sí (solo alineadas) | No |

**Ejemplo concreto**:
- Historia generada #3 en LEVEL 1: **weak** (0.6331) → NO pasa a LEVEL 3
- Historia esperada #8 en LEVEL 2: **cubierta** por generada #10 (0.8693) → contribuye al coverage

---

## ✅ LEVEL 3: Acceptance Criteria Evaluator

**Archivo**: `pipeline/acceptance_criteria_evaluator.py`

### ⚠️ REGLA CRÍTICA
**Solo evalúa criterios de aceptación de historias ALINEADAS** (10 historias Strong + Conservative del LEVEL 1).

### Input EXACTO (caso_A_alineadas)

**Del LEVEL 1 vienen 10 pares alineados**:
```python
aligned_pairs = [
  (gen_idx=0, esp_idx=0),   # Par #1
  (gen_idx=1, esp_idx=1),   # Par #2
  (gen_idx=2, esp_idx=3),   # Par #3
  # ... 7 pares más
]
```

**Estructura de CA**:
```python
# ca_generados.json - Lista de listas [23]
ca_gen = [
  ["The system allows entry of personal and medical background data",  # CA de historia generada #0
   "Each patient receives a unique identifier",
   "Duplicate identity documents or identifiers are not allowed"],
  ["The system allows editing of existing patient records"],            # CA de historia generada #1
  # ... 21 más
]

# ca_esperados.json - Lista de listas [28]
ca_esp = [
  ["The system must request full name of the patient",                 # CA de historia esperada #0
   "The system must request date of birth of the patient",
   "The system must request sex of the patient",
   "The system must request an identity document number",
   "The system must request contact information including email and phone number",
   "The identity document must be unique in the system",
   "If the identity document already exists, an error message is displayed",
   "The system must generate a unique patient identifier"],
  ["The system must allow editing the full name of a registered patient",  # CA de historia esperada #1
   "The system must allow editing the date of birth of a registered patient",
   "The system must allow editing the sex of a registered patient",
   "The system must allow editing the identity document of a registered patient",
   "The system must allow editing patient contact information including email and phone number"],
  # ... 26 más
]
```

---

### Métrica 1: Cobertura Funcional

#### ¿Qué evalúa?
¿Los CA generados de una historia cubren funcionalmente los CA esperados de su par alineado?

#### Proceso: All-vs-All DENTRO del par

**Ejemplo con Par #1** (historia gen #0 ↔ historia esp #0):

**Paso 1**: Extraer CA del par
```python
ca_gen_0 = [
  "The system allows entry of personal and medical background data",
  "Each patient receives a unique identifier",
  "Duplicate identity documents or identifiers are not allowed"
]  # 3 CA generados

ca_esp_0 = [
  "The system must request full name of the patient",
  "The system must request date of birth of the patient",
  "The system must request sex of the patient",
  "The system must request an identity document number",
  "The system must request contact information including email and phone number",
  "The identity document must be unique in the system",
  "If the identity document already exists, an error message is displayed",
  "The system must generate a unique patient identifier"
]  # 8 CA esperados
```

**Paso 2**: Codificar con SBERT
```python
emb_gen_0 = encoder.encode(ca_gen_0)  # [3, 768]
emb_esp_0 = encoder.encode(ca_esp_0)  # [8, 768]
```

**Paso 3**: Matriz de similitudes (all-vs-all)
```python
matriz = util.cos_sim(emb_esp_0, emb_gen_0)  # [8, 3]
# Cada fila: similitudes de 1 CA esperado vs los 3 CA generados
```

**Paso 4**: Para cada CA esperado, buscar max similitud
```python
# CA esperado #0: "The system must request full name of the patient"
similitudes_ca_0 = [0.6395, 0.4123, 0.3891]  # vs 3 CA generados
max_sim = 0.6395  # ← Por debajo del umbral 0.75
cubierto = False

# CA esperado #5: "The identity document must be unique in the system"
similitudes_ca_5 = [0.4521, 0.4234, 0.7971]  # vs 3 CA generados
max_sim = 0.7971  # ← Por encima del umbral 0.75
best_match = "Duplicate identity documents or identifiers are not allowed"
cubierto = True

# CA esperado #7: "The system must generate a unique patient identifier"
similitudes_ca_7 = [0.5123, 0.8235, 0.6789]
max_sim = 0.8235  # ← Match con "Each patient receives a unique identifier"
cubierto = True
```

**Paso 5**: Calcular cobertura del par
```python
cubiertas = 3  # CA esp #5, #6, #7 están cubiertos
total = 8
coverage_par_0 = 3 / 8 = 0.375 = 37.5%
```

#### Resultado REAL (caso_A_alineadas)

**Coverage por par**:
```json
{
  "pair_results": [
    {
      "story_generated_idx": 0,
      "story_expected_idx": 0,
      "functional_coverage": 0.375,  // 3 de 8 CA cubiertos
      "functional_coverage_details": [
        {
          "expected_ca": "The system must request full name of the patient",
          "max_similarity": 0.6395,
          "covered": false,  // < 0.75
          "best_match": null
        },
        {
          "expected_ca": "The identity document must be unique in the system",
          "max_similarity": 0.7971,
          "covered": true,
          "best_match": "Duplicate identity documents or identifiers are not allowed"
        },
        {
          "expected_ca": "The system must generate a unique patient identifier",
          "max_similarity": 0.8235,
          "covered": true,
          "best_match": "Each patient receives a unique identifier"
        }
        // ... 5 más
      ]
    },
    // ... 9 pares más
  ],
  "aggregate": {
    "num_aligned_pairs": 10,
    "avg_functional_coverage": 0.1136  // Promedio de los 10 pares = 11.36%
  }
}
```

---

### Métrica 2: Verificabilidad

#### ¿Qué evalúa?
¿Los CA generados son testables? (tienen condición + resultado observable)

**⚠️ IMPORTANTE**: Se evalúan TODOS los CA de las 10 historias alineadas (no todos los CA de todas las historias).

#### Extracción de CA alineados
```python
# Solo CA de historias generadas alineadas
ca_alineados = []
for (gen_idx, esp_idx) in aligned_pairs:
    ca_alineados.extend(ca_gen[gen_idx])

# Resultado: 15 CA totales de las 10 historias alineadas
```

#### Criterios de verificabilidad
Un CA es **verificable** si tiene:
1. **Condición explícita**: palabras clave
   ```python
   CONDITION_PATTERNS = [
       r'\bif\b', r'\bwhen\b', r'\bgiven\b', r'\bafter\b',
       r'\bbefore\b', r'\bupon\b', r'\bas soon as\b',
       r'\bonce\b', r'\bin case\b', r'\bunless\b'
   ]
   ```

2. **Resultado observable**: palabras clave
   ```python
   RESULT_PATTERNS = [
       r'\bmust\b', r'\bshall\b', r'\bshould\b', r'\bwill\b',
       r'\bdisplay\b', r'\bshow\b', r'\bnotif\w+\b', r'\balert\b',
       r'\ballows?\b', r'\bprevent\b', r'\benable\b'
   ]
   ```

#### Ejemplo: Análisis de CA
```python
# CA #1: "The system allows entry of personal and medical background data"
has_condition = False  # No tiene if/when/given
has_result = True      # Tiene "allows"
verificable = False    # Necesita ambas → Parcialmente verificable

# CA #2: "Each patient receives a unique identifier"
has_condition = False  # No condición
has_result = False     # "receives" no está en patterns
verificable = False    # No verificable

# CA #3: "Duplicate identity documents or identifiers are not allowed"
has_condition = False
has_result = True      # Tiene "allowed"
verificable = False    # Parcialmente verificable

# Hipotético verificable:
# "If the identity document already exists, the system must display an error message"
has_condition = True   # "if"
has_result = True      # "must display"
verificable = True     # ✓ VERIFICABLE
```

#### Resultado REAL (caso_A_alineadas)
```json
{
  "global_metrics": {
    "total_criteria_evaluated": 15,  // CA de 10 historias alineadas
    "verifiability": {
      "score": 0.0,  // 0% verificables
      "details": [
        {
          "criteria": "The system allows entry of personal and medical background data",
          "has_condition": false,
          "has_observable_result": true,
          "is_verifiable": false  // Falta condición
        },
        {
          "criteria": "Each patient receives a unique identifier",
          "has_condition": false,
          "has_observable_result": false,
          "is_verifiable": false
        }
        // ... 13 más, todos no verificables
      ]
    }
  }
}
```

**Interpretación**:
- 0% verificables es común cuando los CA no están escritos en formato *Given-When-Then* o similar
- No es necesariamente un error, solo indica que los CA no son testables automáticamente

---

### Métrica 3: Ambigüedad

#### ¿Qué evalúa?
¿Los CA generados contienen términos vagos o ambiguos?

#### Términos ambiguos detectados
```python
AMBIGUOUS_TERMS = [
    "appropriate", "suitable", "adequate", "reasonable", "efficient",
    "fast", "slow", "quickly", "soon", "large", "small", "many", "few",
    "significant", "important", "relevant", "user-friendly", "easy",
    "flexible", "robust", "clear", "simple", "complex", "optimal",
    "minimal", "maximum", "sufficient", "necessary"
]
```

#### Ejemplo: Análisis de CA
```python
# CA: "The system allows entry of personal and medical background data"
ambiguous_terms_found = []  # No contiene términos vagos
is_ambiguous = False

# CA: "Notifications are sent promptly after changes"
ambiguous_terms_found = ["promptly"]  # "promptly" es vago
is_ambiguous = True

# CA hipotético: "The system should be fast and user-friendly"
ambiguous_terms_found = ["fast", "user-friendly"]
is_ambiguous = True
```

#### Resultado REAL (caso_A_alineadas)
```json
{
  "global_metrics": {
    "total_criteria_evaluated": 15,
    "ambiguity": {
      "score": 1.0,  // 100% sin ambigüedad = 0% ambiguos
      "details": [
        {
          "criteria": "The system allows entry of personal and medical background data",
          "is_ambiguous": false,
          "ambiguous_terms_found": []
        },
        {
          "criteria": "Notifications are sent promptly after changes",
          "is_ambiguous": false,  // "promptly" no detectado (puede necesitar actualización)
          "ambiguous_terms_found": []
        }
        // ... 13 más, todos sin ambigüedad detectada
      ]
    }
  }
}
```

**Interpretación**:
- **Score = 1.0**: ningún CA contiene términos ambiguos detectados
- **Score < 0.8**: muchos CA con términos vagos → revisar especificaciones

---

## 🧩 LEVEL 4: Concept Coverage Evaluator

**Archivo**: `pipeline/concept_coverage_evaluator.py`

### ⚠️ REGLA CRÍTICA
**Solo evalúa historias ALINEADAS** (10 historias Strong + Conservative del LEVEL 1).

### ¿Qué hace?
Mide si las historias alineadas cubren los aspectos/conceptos clave del dominio.

### Input EXACTO (caso_A_alineadas)

**Historias alineadas** (solo textos de las 10 historias que pasaron LEVEL 1):
```python
aligned_stories = [
  "As an administrative staff member, I want to register patients with personal data...",  # gen #0
  "As an administrative staff member, I want to modify the personal and contact details...",  # gen #1
  "As an administrative staff member, I want to search for patients by name...",  # gen #2
  # ... 7 más (total: 10)
]
```

**Conceptos del dominio** (`aspectosHospital2.json` - INGLÉS):
```json
{
  "patients": "patient registration and management",
  "appointments": "appointments and consultation scheduling",
  "medical_records": "electronic medical records that patients can consult, but not modify",
  "staff": "physician registration and management",
  "invoice": "invoice generation and notification",
  "inventory": "inventory control of supplies in the hospital internal pharmacy",
  "reports": "statistics of consultations and financial reports"
}
```

### Proceso PASO A PASO

**Ejemplo con concepto "patients"**:

**Paso 1**: Extraer descripción del concepto
```python
concept_name = "patients"
concept_description = "patient registration and management"
```

**Paso 2**: Codificar con SBERT
```python
emb_concept = encoder.encode([concept_description])  # [1, 768]
emb_stories = encoder.encode(aligned_stories)         # [10, 768]
```

**Paso 3**: Calcular similitud con TODAS las historias alineadas
```python
similitudes = util.cos_sim(emb_concept, emb_stories)[0]
# [0.7234, 0.6891, 0.7012, 0.4521, 0.3234, 0.5123, 0.7891, 0.4234, 0.3891, 0.6234]
#  ↑ sim con cada una de las 10 historias alineadas
```

**Paso 4**: Encontrar máxima similitud
```python
max_sim = max(similitudes)  # → 0.7891
best_story_idx = 6  # Historia alineada #6 (gen_idx=11)
best_story = "As an administrator, I want to register medical professionals with their personal and contact details..."
```

**Paso 5**: Verificar umbral
```python
threshold = 0.70
if max_sim >= threshold:
    concepto_cubierto = True
else:
    concepto_cubierto = False

# Concepto "patients": 0.7891 ≥ 0.70 → ✓ CUBIERTO
```

### Resultado REAL (caso_A_alineadas con aspectos en INGLÉS)

Supongamos que se ejecutó con `aspectosHospital2.json`:

```json
{
  "aspect_coverage_score": 0.4286,  // 3 de 7 conceptos cubiertos = 42.86%
  "aspects_covered": 3,
  "aspects_total": 7,
  "threshold": 0.7,
  "details": [
    {
      "concept": "patients",
      "covered": true,
      "max_similarity": 0.7891,
      "best_description": "patient registration and management",
      "best_story_idx": 0,  // Índice en aligned_stories
      "best_story": "As an administrative staff member, I want to register patients..."
    },
    {
      "concept": "appointments",
      "covered": true,
      "max_similarity": 0.7423,
      "best_description": "appointments and consultation scheduling",
      "best_story_idx": 3,
      "best_story": "As an administrative staff member, I want to modify or cancel existing appointments..."
    },
    {
      "concept": "medical_records",
      "covered": true,
      "max_similarity": 0.7234,
      "best_description": "electronic medical records that patients can consult, but not modify",
      "best_story_idx": 5,
      "best_story": "As a patient, I want to consult my electronic clinical record..."
    },
    {
      "concept": "staff",
      "covered": false,
      "max_similarity": 0.6823,  // ← Por debajo del umbral 0.70
      "best_description": "physician registration and management",
      "best_story_idx": 6,
      "best_story": "As an administrator, I want to register medical professionals..."
    },
    {
      "concept": "invoice",
      "covered": false,
      "max_similarity": 0.6234,
      "best_description": "invoice generation and notification",
      "best_story_idx": 8
    },
    {
      "concept": "inventory",
      "covered": false,
      "max_similarity": 0.6512,
      "best_description": "inventory control of supplies in the hospital internal pharmacy",
      "best_story_idx": 9
    },
    {
      "concept": "reports",
      "covered": false,
      "max_similarity": 0.5823,
      "best_description": "statistics of consultations and financial reports",
      "best_story_idx": 6
    }
  ]
}
```

### ⚠️ CRÍTICO: Idioma de conceptos

**Problema con `aspectos_hospital.json` (ESPAÑOL)**:
```json
{
  "pacientes": "registrar pacientes con datos personales",
  "turnos": "crear turnos medicos",
  ...
}
```

Si las historias están en **INGLÉS** pero los conceptos en **ESPAÑOL**:
```python
# Similitud entre:
emb_concept = encode("registrar pacientes con datos personales")  # Español
emb_story = encode("register patients with personal data...")     # Inglés

# Resultado: similitud MUY BAJA (0.30-0.50) por diferencia de idioma
# Todos los conceptos aparecerán como NO CUBIERTOS → Coverage = 0%
```

**Solución**:
```bash
# Usar aspectos en el MISMO idioma que las historias
python ejecutar_pipeline.py --caso caso_A_alineadas --aspectos casos_prueba/aspectosHospital2.json
                                                                          ^^^^^^^^^^^^^^^^^^^^^^
                                                                          INGLÉS
```

### Formato esperado de aspectos

#### Opción 1: Diccionario simple (usado en el código)
```json
{
  "concept_key": "description of the concept",
  "another_concept": "another description"
}
```

#### Opción 2: Lista con descripciones múltiples (soportado)
```json
[
  {
    "concept": "patients",
    "descriptions": [
      "register patients with personal data",
      "edit patient information",
      "search for patients in the system"
    ]
  },
  {
    "concept": "appointments",
    "descriptions": [
      "create medical appointments",
      "cancel appointments",
      "reschedule appointments"
    ]
  }
]
```

**Con descripciones múltiples**, el proceso es:
```python
# Para cada descripción del concepto:
for desc in concept_descriptions:
    emb_desc = encode([desc])
    sims = cos_sim(emb_desc, emb_stories)[0]
    max_sims.append(max(sims))

# Similitud del concepto = promedio de las similitudes máximas
concept_similarity = mean(max_sims)

if concept_similarity >= threshold:
    concepto_cubierto = True
```

### Interpretación de scores

- **Coverage 0-30%**: muy bajo, faltan conceptos clave del dominio
- **Coverage 30-60%**: moderado, cubre algunos aspectos principales
- **Coverage 60-80%**: bueno, la mayoría de conceptos están representados
- **Coverage >80%**: excelente, cobertura completa del dominio

**En caso_A_alineadas** (42.86%):
- ✓ Cubre: pacientes, turnos, historia clínica
- ✗ No cubre suficientemente: personal, facturación, inventario, reportes
- Interpretación: el modelo generó historias para los casos de uso principales, pero no para aspectos administrativos/operativos

---

## 🔄 Flujo Completo de Datos (caso_A_alineadas REAL)

### 1. INPUT INICIAL
```python
# Archivos fuente:
casos_prueba/caso_A_alineadas/
├── generadas.txt           → 23 historias en inglés
├── esperadas.txt           → 28 historias en inglés  
├── ca_generados.json       → [[ca1, ca2], [ca3], ...] - 23 listas
├── ca_esperados.json       → [[ca1], [ca2, ca3], ...] - 28 listas
└── metadata.json

casos_prueba/aspectosHospital2.json  → 7 conceptos en inglés
```

### 2. LEVEL 0: Encoding
**Qué se codifica**:
```python
✓ 23 historias generadas      → tensor [23, 768]
✓ 28 historias esperadas       → tensor [28, 768]
✓ Todos los CA generados       → tensor [n_ca_gen, 768]
✓ Todos los CA esperados       → tensor [m_ca_esp, 768]
✓ 7 descripciones de conceptos → tensor [7, 768]
```

### 3. LEVEL 1: Alignment
**Input**:
- Embeddings de 23 generadas [23, 768]
- Embeddings de 28 esperadas [28, 768]

**Proceso**:
```python
# Matriz de similitudes SBERT
matriz = util.cos_sim(emb_gen, emb_exp)  # [23, 28]

# Para cada historia generada (fila):
for i in range(23):
    similitudes = matriz[i]  # Vector de 28 valores
    best_match_idx = argmax(similitudes)
    best_sim = similitudes[best_match_idx]
    
    # Clasificación
    if best_sim >= 0.85:
        nivel = "conservative"  # PASA
    elif best_sim >= 0.80:
        nivel = "strong"        # PASA
    else:
        nivel = "weak"          # FILTRADA
```

**Output REAL**:
```
De 23 historias generadas:
├─ 10 ALINEADAS (43.48%)
│  ├─ 2 Strong
│  └─ 8 Conservative
└─ 13 FILTRADAS (weak)

Ejemplos:
✓ Gen #0 ↔ Esp #0: sim=0.8596 → conservative → PASA
✓ Gen #2 ↔ Esp #3: sim=0.8135 → strong → PASA
✓ Gen #7 ↔ Esp #7: sim=0.9751 → conservative → PASA
✗ Gen #3 ↔ Esp #8: sim=0.6331 → weak → FILTRADA
✗ Gen #4 ↔ Esp #10: sim=0.7802 → weak → FILTRADA
```

**Pares alineados** (índices):
```
10 pares que pasan:
(gen=0, esp=0), (gen=1, esp=1), (gen=2, esp=3), (gen=6, esp=5),
(gen=7, esp=7), (gen=10, esp=18), (gen=11, esp=10), (gen=13, esp=14),
(gen=15, esp=17), (gen=16, esp=19)
```

### 4. LEVEL 2: Coverage
**Input**:
- TODAS las 23 historias generadas (incluye las 13 filtradas)
- Las 28 esperadas
- Umbral: 0.75

**Proceso**:
```python
# Para cada historia esperada:
for esp in esperadas (28):
    # Buscar similitud máxima con CUALQUIER generada
    max_sim = max(cos_sim(esp, gen) for gen in generadas)
    if max_sim >= 0.75:
        cubierta = True
```

**Output REAL**:
```
De 28 historias esperadas:
├─ 16 CUBIERTAS (57.14%)
└─ 12 NO CUBIERTAS (42.86%)

Ejemplos cubiertas:
✓ Esp #0: max_sim=0.8596 (con Gen #0)
✓ Esp #7: max_sim=0.9751 (con Gen #7)
✓ Esp #10: max_sim=0.8541 (con Gen #11)

Ejemplos NO cubiertas:
✗ Esp #9: max_sim=0.7032 < 0.75
✗ Esp #12: max_sim=0.6037 < 0.75
✗ Esp #15: max_sim=0.7148 < 0.75
```

**Diferencia con LEVEL 1**:
- LEVEL 1: Gen #4 (sim=0.7802) → weak → FILTRADA
- LEVEL 2: Gen #4 contribuye al coverage de Esp #4 (sim=0.7598 ≥ 0.75)

### 5. LEVEL 3: CA Evaluation
**Input**:
- Solo CA de las **10 historias alineadas**
- ca_generados[0, 1, 2, 6, 7, 10, 11, 13, 15, 16]
- ca_esperados[0, 1, 3, 5, 7, 18, 10, 14, 17, 19]

**Ejemplo Par #1** (gen=0, esp=0):
```python
# CA generados de historia #0 (3 criterios)
ca_gen_0 = [
  "The system allows entry of personal and medical background data",
  "Each patient receives a unique identifier",
  "Duplicate identity documents or identifiers are not allowed"
]

# CA esperados de historia #0 (8 criterios)
ca_esp_0 = [
  "The system must request full name of the patient",
  "The system must request date of birth of the patient",
  "The system must request sex of the patient",
  "The system must request an identity document number",
  "The system must request contact information including email and phone number",
  "The identity document must be unique in the system",
  "If the identity document already exists, an error message is displayed",
  "The system must generate a unique patient identifier"
]

# Cobertura funcional all-vs-all
matriz = cos_sim(emb_esp_0, emb_gen_0)  # [8, 3]
for ca_esp in ca_esp_0:
    max_sim = max(similitudes con los 3 ca_gen)
    if max_sim >= 0.75:
        cubierto += 1

# Resultado: 3 de 8 cubiertos = 37.5%
```

**Output REAL**:
```
10 pares evaluados:
├─ Cobertura funcional promedio: 11.36%
│  (muy baja: CA generados son genéricos)
├─ Verificabilidad global: 0.00%
│  (ningún CA tiene condición+resultado)
└─ No-ambigüedad: 100.00%
   (ningún CA tiene términos vagos)

Total CA evaluados: 15
(suma de CA de las 10 historias generadas alineadas)
```

### 6. LEVEL 4: Conceptual Coverage
**Input**:
- Solo textos de las **10 historias alineadas**
- 7 conceptos del dominio (inglés)

**Ejemplo concepto "patients"**:
```python
concept_desc = "patient registration and management"
aligned_stories = [
  "As an administrative staff member, I want to register patients...",    # gen #0
  "As an administrative staff member, I want to modify the personal...",  # gen #1
  # ... 8 más
]

# Similitud del concepto con cada historia
sims = cos_sim(encode(concept_desc), encode(aligned_stories))[0]
# [0.7891, 0.6234, 0.5123, ..., 0.4521]

max_sim = 0.7891  # Con historia gen #0
if max_sim >= 0.70:
    concepto_cubierto = True
```

**Output REAL** (con aspectosHospital2.json en inglés):
```
De 7 conceptos:
├─ 3 CUBIERTOS (42.86%)
│  ✓ patients: 0.7891 (gen #0)
│  ✓ appointments: 0.7423 (gen #6)
│  ✓ medical_records: 0.7234 (gen #10)
└─ 4 NO CUBIERTOS (57.14%)
   ✗ staff: 0.6823 < 0.70
   ✗ invoice: 0.6234 < 0.70
   ✗ inventory: 0.6512 < 0.70
   ✗ reports: 0.5823 < 0.70
```

**Con aspectos_hospital.json en español** (ERROR):
```
De 7 conceptos:
└─ 0 CUBIERTOS (0.00%)  ← Idioma mismatch
   ✗ pacientes: 0.4684 < 0.70
   ✗ turnos: 0.2121 < 0.70
   ✗ historia_clinica: 0.5070 < 0.70
   ...
```

### 7. RESUMEN FINAL
```json
{
  "input_stats": {
    "stories_generated": 23,
    "stories_expected": 28,
    "stories_aligned": 10,
    "alignment_rate": 0.4348  // 43.48%
  },
  "level_1_alignment": {
    "sbert_mean": 0.7521,
    "aligned_count": 10,
    "strong_count": 2,
    "conservative_count": 8,
    "weak_count": 13
  },
  "level_2_coverage": {
    "story_coverage": 0.5714,  // 57.14%
    "stories_covered": 16,
    "stories_total": 28
  },
  "level_3_criteria": {
    "pairs_evaluated": 10,
    "avg_functional_coverage": 0.1136,  // 11.36%
    "avg_verifiability": 0.0,
    "global_no_ambiguity": 1.0
  },
  "level_4_concepts": {
    "concept_coverage": 0.4286,  // 42.86%
    "concepts_covered": 3,
    "concepts_total": 7
  }
}
```

### Diagrama de filtrado de datos

```
INPUT:
23 generadas, 28 esperadas

    ↓ LEVEL 1: Alignment

10 alineadas ────────┐
(strong+conservative)│
                     │
13 filtradas (weak)  │
                     │
    ↓ LEVEL 2: Coverage (usa 23)
                     │
16 esperadas cubiertas
                     │
    ↓ LEVEL 3: CA Evaluation
                     │
Solo CA de las ─────→ 10 alineadas
                     │
15 CA totales evaluados
                     │
    ↓ LEVEL 4: Conceptual Coverage
                     │
Solo textos de las ─→ 10 alineadas
                     │
7 conceptos evaluados vs 10 historias
```

---

## 🛠️ Runners Disponibles

### 1. `ejecutar_pipeline.py` (RECOMENDADO)
Pipeline refactorizado con metodología estricta.

```bash
python ejecutar_pipeline.py --caso caso_A_alineadas --aspectos casos_prueba/aspectosHospital2.json
```

**Características**:
- Arquitectura modular limpia
- SBERT exclusivo para filtrado
- BERTScore siempre secundario (sin flag CLI)
- CA y conceptos solo para alineadas

### 2. `ejecutar_casos.py` (LEGACY)
Runner original, menos estructurado pero funcional.

```bash
python ejecutar_casos.py --caso caso_A_alineadas
```

---

## 📁 Estructura de Casos de Prueba

```
casos_prueba/
├── caso_A_alineadas/
│   ├── metadata.json           # Descripción del caso
│   ├── generadas.txt           # Historias generadas
│   ├── esperadas.txt           # Historias esperadas (ground truth)
│   ├── ca_generados.json       # Lista de listas: [[ca1, ca2], [ca3], ...]
│   ├── ca_esperados.json       # Lista de listas: [[ca1], [ca2, ca3], ...]
│   └── historias_por_herramienta.json  # Historias por modelo (opcional)
│
├── aspectos_hospital.json      # Conceptos en ESPAÑOL
└── aspectosHospital2.json      # Conceptos en INGLÉS
```

### Formato `metadata.json`
```json
{
  "descripcion": "Historias alineadas - dominio hospitalario",
  "dominio": "hospital",
  "evaluar_alineacion": true,
  "evaluar_completitud": true,
  "evaluar_consistencia": true
}
```

### Formato `ca_generados.json` / `ca_esperados.json`
```json
[
  ["CA1 de historia 0", "CA2 de historia 0"],
  ["CA1 de historia 1"],
  ["CA1 de historia 2", "CA2 de historia 2", "CA3 de historia 2"]
]
```
- Lista de listas
- Índice externo = índice de historia
- Índice interno = CA dentro de esa historia

### Formato `aspectos.json`
```json
[
  {
    "concept": "patients",
    "descriptions": [
      "register patients with personal data",
      "edit patient information"
    ]
  },
  {
    "concept": "appointments",
    "descriptions": ["create medical appointments"]
  }
]
```

---

## 📊 Interpretación de Resultados

### Resultado exitoso típico
```
LEVEL 1 - Alineación:
  Historias alineadas: 10/23 (43.48%)
  SBERT media: 0.752
  Strong: 2, Conservative: 8, Weak: 13

LEVEL 2 - Coverage:
  Story coverage: 57.14% (16/28)

LEVEL 3 - Criterios de Aceptación:
  Pares evaluados: 10 (solo alineadas)
  Cobertura funcional promedio: 11.36%
  Verificabilidad global: 0.00%
  No-ambigüedad global: 100.00%

LEVEL 4 - Coverage Conceptual:
  Cobertura: 42.86% (3/7)
```

### Interpretación

#### LEVEL 1
- **Alineadas**: % de historias generadas que son suficientemente similares a las esperadas
- **SBERT media**: similitud semántica promedio (0.70-0.80 típico)
- **Strong/Conservative/Weak**: distribución de calidad de matches

#### LEVEL 2
- **Coverage alto**: el modelo cubre bien los requisitos esperados
- **Coverage bajo**: faltan historias importantes

#### LEVEL 3
- **Cobertura funcional**: ¿los CA generados cubren los esperados?
  - <20%: muy baja
  - 20-50%: moderada
  - >50%: buena
- **Verificabilidad**: % de CA que son testables (0% común si no hay keywords)
- **No-ambigüedad**: % de CA sin términos vagos (100% = sin ambigüedad)

#### LEVEL 4
- **Cobertura conceptual**: ¿las historias cubren todos los aspectos del dominio?
  - <30%: muy baja
  - 30-60%: moderada
  - >60%: buena

---

## ⚙️ Configuración de Umbrales

Todos los umbrales están en `pipeline/evaluation_pipeline.py`:

```python
DEFAULT_CONFIG = {
    "umbral_strong": 0.80,           # LEVEL 1: Strong alignment
    "umbral_conservative": 0.85,     # LEVEL 1: Conservative alignment
    "coverage_threshold": 0.75,      # LEVEL 2: Story coverage
    "ca_coverage_threshold": 0.75,   # LEVEL 3: CA functional coverage
    "concept_threshold": 0.70,       # LEVEL 4: Concept coverage
}
```

### Recomendaciones
- **Stricter** (más exigente): aumentar umbrales → menos alineaciones/cobertura
- **Looser** (más permisivo): disminuir umbrales → más alineaciones/cobertura

---

## 🐛 Diagnóstico de Problemas Comunes

### 1. Conceptual coverage = 0%
**Causa**: idioma de aspectos ≠ idioma de historias
**Solución**: usar `aspectosHospital2.json` (inglés) con historias en inglés

### 2. Verificabilidad = 0%
**Causa**: CA no tienen keywords de condición (`if`, `when`, etc.)
**Solución**: metodológica, no bug; los CA pueden no estar escritos en formato testable

### 3. BERTScore = null
**Causa**: modelo no disponible o error al cargar
**Solución**: no afecta resultados; BERTScore es opcional y secundario

### 4. Cobertura funcional muy baja (<10%)
**Causa**: CA generados muy diferentes a esperados, o umbral muy estricto
**Solución**: revisar calidad de generación o ajustar `ca_coverage_threshold`

---

## 📝 Salida JSON Completa

```json
{
  "timestamp": "2026-02-12T20:44:01",
  "metadata": {...},
  "config": {...},
  
  "level_1_alignment": {
    "alignments": [...],         // Todos los matches
    "aligned_stories": [...],    // Solo Strong + Conservative
    "aggregate": {...}
  },
  
  "level_2_coverage": {
    "story_coverage_score": 0.57,
    "covered": 16,
    "total": 28,
    "details": [...]
  },
  
  "level_3_criteria": {
    "pair_results": [...],       // 1 por cada par alineado
    "aggregate": {...},
    "global_metrics": {...}
  },
  
  "level_4_concepts": {
    "aspect_coverage_score": 0.43,
    "aspects_covered": 3,
    "aspects_total": 7,
    "details": [...]
  },
  
  "summary": {
    "input_stats": {...},
    "level_1_alignment": {...},
    "level_2_coverage": {...},
    "level_3_criteria": {...},
    "level_4_concepts": {...}
  }
}
```

---

## 🚀 Quick Start

```bash
# 1. Activar entorno
.\.venv\Scripts\Activate.ps1

# 2. Ejecutar evaluación
python ejecutar_pipeline.py --caso caso_A_alineadas --aspectos casos_prueba/aspectosHospital2.json

# 3. Ver resultados
# JSON en: resultados/caso_A_alineadas_TIMESTAMP.json
# TXT  en: resultados/caso_A_alineadas_TIMESTAMP.txt
```

---

## 📚 Referencias de Código

### Módulos principales
- `pipeline/semantic_encoder.py` → LEVEL 0
- `pipeline/story_alignment_evaluator.py` → LEVEL 1
- `pipeline/story_coverage_calculator.py` → LEVEL 2
- `pipeline/acceptance_criteria_evaluator.py` → LEVEL 3
- `pipeline/concept_coverage_evaluator.py` → LEVEL 4
- `pipeline/evaluation_pipeline.py` → Orquestador

### Evaluadores legacy
- `evaluador_metricas.py` → alineación y coverage (legacy)
- `evaluador_criterios_aceptacion.py` → CA metrics (legacy)
- `cargador_datos.py` → I/O de casos de prueba

---

## ✅ Validación del Pipeline

Pipeline correcto si:
1. ✅ SBERT decide alineación (no BERTScore)
2. ✅ Solo Strong + Conservative pasan
3. ✅ LEVEL 3 evalúa solo historias alineadas
4. ✅ LEVEL 4 evalúa solo historias alineadas
5. ✅ CA evaluation es all-vs-all dentro de cada par
6. ✅ No hay comparación global de CA entre todas las historias
