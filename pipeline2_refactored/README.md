# pipeline2_refactored

Pipeline de evaluación semántica de historias de usuario generadas por un LLM. Utiliza SBERT (Sentence-BERT) para medir la calidad de las historias generadas en tres dimensiones ortogonales.

---

## ¿Qué hace?

Dado un conjunto de **historias generadas por un LLM** y un conjunto de **historias esperadas** (ground truth humano), el sistema responde tres preguntas:

| Dimensión | Pregunta |
|-----------|----------|
| **1 — Cobertura Funcional** | ¿El LLM cubrió todas las funcionalidades del dominio? |
| **2 — Similitud de Solución** | ¿Qué tan parecidas son las historias generadas a las esperadas? |
| **2b — Alineación de CA** | ¿Los criterios de aceptación también se alinean semánticamente? |
| **3 — Detección de Alucinaciones** | ¿Alguna historia generada no tiene respaldo en las esperadas? |

Toda comparación usa **similitud coseno sobre embeddings SBERT** — no búsqueda por palabras clave.

---

## Estructura del proyecto

```
pipeline2_refactored/
│
├── main.py                        # Orquestador principal + CLI
├── __main__.py                    # Permite: python -m pipeline2_refactored
├── semantic_encoder.py            # Clase SemanticEncoder (wrapper de SBERT)
│
├── config/
│   └── settings.py                # PipelineConfig: todos los umbrales
│
├── embeddings/
│   └── encoder.py                 # create_encoder(): fábrica del encoder
│
├── io/
│   ├── loaders.py                 # Parsers de entrada (txt, json)
│   └── reporters.py               # Escritores de salida (JSON, CSV, grafo)
│
├── similarity/
│   └── metrics.py                 # compute_similarity_matrix(), top_k_matches()
│
├── coverage/
│   └── functional_coverage.py     # Dimensión 1: Cobertura Funcional
│
├── matching/
│   ├── story_matching.py          # Dimensión 2: Similitud de Solución
│   └── ca_alignment.py            # Dimensión 2b: Alineación de CA
│
├── hallucination/
│   └── detector.py                # Dimensión 3: Detección de Alucinaciones
│
└── refinement/
    └── refinement_module.py       # Opcional: sugerencias de mejora via LLM
```

---

## Flujo de ejecución

```
Argumentos CLI
      │
      ▼
[main.py] ──► [loaders.py]         Carga archivos de entrada
      │
      ├──► [encoder.py]            Instancia SemanticEncoder (SBERT)
      │
      ├──► [functional_coverage]   DIM 1: aspectos × generadas → matriz similitud
      │         └──► [metrics.py]
      │
      ├──► [story_matching]        DIM 2: generadas × esperadas → matriz similitud
      │         └──► [metrics.py]
      │
      ├──► [ca_alignment]          DIM 2b: CA generados × CA esperados (por historia)
      │
      ├──► [detector]              DIM 3: reutiliza scores de DIM 2, sin re-encoding
      │
      ├──► [reporters.py]          Serializa resultados → JSON + CSV + grafo
      │
      └──► [refinement_module]     Opcional: llama LLM externo con issues detectados
```

### Paso a paso detallado

1. **Validación de entradas** — Se verifican rutas de archivos antes de hacer cualquier cómputo.
2. **Carga de datos** — `loaders.py` parsea los `.txt` y `.json`. Si los CA tienen distinto largo que las historias, se alinean con padding/truncamiento.
3. **Carga del modelo** — `SemanticEncoder` carga SBERT, auto-detecta CUDA/CPU y silencia logs de librerías.
4. **Dimensión 1** — Se encodean aspectos e historias generadas. Se calcula la matriz `(aspectos × generadas)`. Cada aspecto recibe su mejor historia y un estado: `covered`, `possibly_covered` o `not_covered`.
5. **Dimensión 2** — Se encodean historias generadas y esperadas. Matriz `(generadas × esperadas)`. Cada historia generada obtiene su top-k de historias esperadas más similares.
6. **Dimensión 2b** — Para las historias con similitud ≥ umbral, se compara cada CA generado contra **todos** los CA de la historia esperada rank-1. Resultado: matriz completa de pares `(gen_CA × exp_CA)`.
7. **Dimensión 3** — Se reusan los scores de la Dimensión 2 (sin re-encoding). Cada historia se clasifica como `aligned`, `uncertain` o `possible_hallucination`.
8. **Guardado** — `reporters.py` escribe todos los reportes al directorio de salida.
9. **Refinamiento** *(opcional)* — Se extraen los problemas (no cubiertos, épicos, alucinaciones), se construye un prompt y se llama al LLM externo.
10. **Resumen** — Se imprime un resumen tabulado en consola.

---

## Sistema de umbrales

Todos los umbrales viven en `config/settings.py` (`PipelineConfig`) y son sobreescribibles via CLI:

| Campo | Valor por defecto | Uso |
|-------|------------------|-----|
| `coverage_threshold_covered` | `0.75` | Aspecto considerado cubierto |
| `coverage_threshold_partial` | `0.60` | Aspecto posiblemente cubierto |
| `hallucination_threshold_aligned` | `0.75` | Historia alineada con esperadas |
| `hallucination_threshold_uncertain` | `0.60` | Historia incierta |
| `criteria_threshold_matching` | `0.75` | CA par coincidente |
| `criteria_threshold_low` | `0.50` | CA par con baja similitud |
| `ca_alignment_story_threshold` | `0.60` | Mínimo para entrar en análisis CA |
| `graph_edge_threshold` | `0.60` | Similitud mínima para aristas del grafo |
| `top_k` | `3` | Cantidad de matches a recuperar |

**Lógica de clasificación (igual en todas las dimensiones):**

```
0.0 ──────── 0.50 ──────── 0.60 ──────── 0.75 ──── 1.0
             │              │              │
           missing      low_sim /       matching /
                        uncertain       covered /
                                        aligned
```

---

## Entradas

| Argumento | Formato | Descripción |
|-----------|---------|-------------|
| `--generadas` | `.txt`, una por línea | Historias generadas por el LLM |
| `--esperadas` | `.txt`, una por línea | Historias esperadas (ground truth) |
| `--aspectos` | `.json` (dict o lista) | Funcionalidades del dominio |
| `--ca-gen` *(opcional)* | `.json` lista de listas | CA de cada historia generada |
| `--ca-esp` *(opcional)* | `.json` lista de listas | CA de cada historia esperada |

**Formato de `--aspectos`:**
```json
// Con categorías:
{ "pacientes": ["Registrar paciente", "Asignar ID único"] }

// Lista plana:
["Registrar paciente", "Asignar ID único"]
```

**Formato de `--ca-gen` / `--ca-esp`:**
```json
[
  ["El sistema debe validar el DNI", "El formulario debe tener campo de fecha"],
  ["El médico puede cancelar la cita con 24h de anticipación"],
  []
]
```
> Cada sublista corresponde a una historia en el mismo orden. Las listas vacías o con texto placeholder (`"No Acceptance Criteria defined"`) se normalizan automáticamente.

---

## Salidas

Todos los archivos se guardan en el directorio indicado por `--output`:

| Archivo | Descripción |
|---------|-------------|
| `coverage_report.json` / `.csv` | Resultado por funcionalidad (estado + top-k matches) |
| `matching_report.json` / `.csv` | Resultado por historia generada (similitud + top-k) |
| `hallucination_report.json` / `.csv` | Clasificación de alucinación por historia |
| `ca_alignment_report.json` | Resultado completo de alineación CA |
| `ca_alignment_matrix.csv` | Un registro por par `(CA_gen, CA_esp)` |
| `ca_alignment_story_summary.csv` | Resumen por historia: tasa de coincidencia CA |
| `semantic_graph.json` | Grafo tripartito (aspectos, generadas, esperadas) |
| `semantic_graph_nodes.csv` | Nodos para Gephi |
| `semantic_graph_edges.csv` | Aristas para Gephi |
| `refinement_report.json` *(opcional)* | Sugerencias del LLM |

---

## Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd codigo/

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

> La primera ejecución descarga el modelo SBERT (~420 MB). Las siguientes usan la caché local.

---

## Cómo correrlo

### Ejecución mínima (sin CA)

```bash
python -m pipeline2_refactored \
  --generadas  casos_prueba/caso_A_alineadas/generadas.txt \
  --esperadas  casos_prueba/caso_A_alineadas/esperadas.txt \
  --aspectos   casos_prueba/caso_A_alineadas/aspectosHospital.json \
  --output     resultados/
```

### Ejecución completa (con CA)

```bash
python -m pipeline2_refactored \
  --generadas  casos_prueba/caso_A_alineadas/generadas.txt \
  --esperadas  casos_prueba/caso_A_alineadas/esperadas.txt \
  --aspectos   casos_prueba/caso_A_alineadas/aspectosHospital.json \
  --ca-gen     casos_prueba/caso_A_alineadas/ca_generados.json \
  --ca-esp     casos_prueba/caso_A_alineadas/ca_esperados.json \
  --output     resultados/
```

### Con refinamiento LLM (OpenAI)

```bash
python -m pipeline2_refactored \
  --generadas  casos_prueba/caso_A_alineadas/generadas.txt \
  --esperadas  casos_prueba/caso_A_alineadas/esperadas.txt \
  --aspectos   casos_prueba/caso_A_alineadas/aspectosHospital.json \
  --refinement \
  --prd        casos_prueba/caso_A_alineadas/pdr.txt \
  --api-key    sk-... \
  --llm-model  gpt-4o \
  --output     resultados/
```

### Ajuste de umbrales

```bash
python -m pipeline2_refactored \
  --generadas  ... \
  --esperadas  ... \
  --aspectos   ... \
  --coverage-threshold      0.70 \
  --hallucination-threshold 0.70 \
  --ca-alignment-threshold  0.65 \
  --top-k                   5 \
  --output     resultados_tuneado/
```

### Forzar dispositivo o modelo

```bash
# Forzar CPU aunque haya GPU
python -m pipeline2_refactored ... --device cpu

# Usar modelo más liviano
python -m pipeline2_refactored ... --model sentence-transformers/all-MiniLM-L6-v2
```

---

## Ejemplo de salida en consola

```
--- Cargando entradas ---
  Historias generadas : 23
  Historias esperadas : 57
  Funcionalidades     : 54
  CA generados        : 23 entradas
  CA esperados        : 57 entradas

Cargando modelo SBERT...
  Dispositivo : cuda:0
  Modelo      : sentence-transformers/all-mpnet-base-v2

--- Dimensión 1: Cobertura Funcional ---
  Cobertura calculada para 54 funcionalidades

--- Dimensión 2: Similitud de Solución (Story Matching) ---
  23 historias generadas comparadas contra 57 esperadas

--- Dimensión 2b: Alineación de CA ---
  Historias evaluadas: 21, omitidas: 2
  Tasa CA: 5.0% (5/100 pares)

--- Dimensión 3: Detección de Alucinaciones ---
  23 historias clasificadas

=================================================================
PIPELINE2 REFACTORED — RESUMEN DE EVALUACIÓN
=================================================================

[DIM 1] Cobertura Funcional
  Total funcionalidades    : 54
  ✔ Cubiertas              : 17  (31.5%)
  ⚠ Posiblemente cubiertas : 24  (44.4%)
  ✖ No cubiertas           : 13  (24.1%)

[DIM 2] Similitud de Solución (Story Matching)
  Total historias generadas : 23
  Similitud promedio        : 0.7714

[DIM 3] Detección de Alucinaciones
  Total historias       : 23
  ✔ Alineadas           : 17  (73.9%)
  ⚠ Inciertas           : 4   (17.4%)
  ✖ Posible alucinación : 2   (8.7%)
```

---

## Referencia de argumentos CLI

| Argumento | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--generadas` | str | *requerido* | Path historias generadas (.txt) |
| `--esperadas` | str | *requerido* | Path historias esperadas (.txt) |
| `--aspectos` | str | *requerido* | Path funcionalidades dominio (.json) |
| `--ca-gen` | str | None | Path CA generados (.json) |
| `--ca-esp` | str | None | Path CA esperados (.json) |
| `--output` | str | `resultados_pipeline2_refactored` | Directorio de salida |
| `--model` | str | `all-mpnet-base-v2` | Nombre modelo SBERT |
| `--device` | str | auto | `cpu` o `cuda` |
| `--top-k` | int | `3` | Cantidad de matches a recuperar |
| `--coverage-threshold` | float | `0.75` | Umbral "cubierto" |
| `--hallucination-threshold` | float | `0.75` | Umbral "alineado" |
| `--ca-alignment-threshold` | float | `0.60` | Umbral mínimo historia para análisis CA |
| `--refinement` | flag | False | Activar refinamiento LLM |
| `--prd` | str | None | Path PRD texto plano (requerido con `--refinement`) |
| `--api-key` | str | None | API key del proveedor LLM |
| `--llm-model` | str | `gpt-4o` | Modelo LLM (`gpt-4o`, `gemini-1.5-pro`, etc.) |
