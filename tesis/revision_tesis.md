# Revision de la Tesis - Informe de Hallazgos

Fecha de revision: 2026-03-25

> **Nota:** Este informe fue generado automaticamente y luego verificado manualmente contra los archivos fuente. ~70% de los hallazgos fueron verificados directamente. Algunos errores reportados pueden estar en codigo LaTeX comentado (indicado donde corresponda). Se recomienda revisar cada hallazgo en contexto antes de aplicar correcciones.

---

## Tabla de contenidos

1. [Errores ortograficos](#1-errores-ortograficos)
2. [Repeticion de conceptos](#2-repeticion-de-conceptos)
3. [Faltas de referencia](#3-faltas-de-referencia)

---

## 1. Errores ortograficos

Se encontraron **~105 instancias** de errores ortograficos en codigo activo. Los problemas mas comunes son:
- **Tildes faltantes** en palabras comunes (articulo, capitulo, generacion, evaluacion, ingles, ademas, sera, esta, mas, pagina)
- **Typos** (letras transpuestas, letras faltantes)
- **Errores de concordancia** de genero/numero

### resumen.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 10 | "el para relevamiento" | "para el relevamiento" o "el relevamiento" | Palabras invertidas o sobrante |

### introduccion.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 9 | "Capitulo" (x2) | "Capitulo" → **"Capítulo"** | "el Capitulo \ref{cap:antecedentes}" |
| 9 | "evaluacion" | **"evaluación"** | "propuesta de evaluacion de historias" |

### revision/conceptos/main.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 17 | "serian" | **"serían"** | "criterios de aceptacion serian" |
| 89 | "articulo" | **"artículo"** | "el articulo Language Models" |
| 89 | "presento" | **"presentó"** | "presento el modelo GPT-3" |
| 89 | "popularizo" | **"popularizó"** | "que popularizo el uso" |
| 97 | "ingles" (x2) | **"inglés"** | "Traducir al ingles" |
| 102 | "Ingles" | **"inglés"** (sin mayuscula) | "traduccion en Ingles" |

### herramientas/main.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 2 | "capitulo" | **"capítulo"** | "en el capitulo \ref{cap:antecedentes}" |
| 4 | "comMetodologíaso" | **"como"** | Typo/artefacto de edicion **(LINEA COMENTADA con %, no afecta compilacion)** |
| 4 | "ademas" | **"además"** | "Nos interesa ademas" **(LINEA COMENTADA con %, no afecta compilacion)** |

### herramientas/metodologia de busqueda.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 4 | "capitulo" | **"capítulo"** | "en el capitulo \ref{cap:antecedentes}" |
| 8 | "evoluciona" | **"evaluación"** | "la posibilidad de la evoluciona de las mismas" |
| 130 | "Capitulo" | **"Capítulo"** | "en el Capitulo \ref{cap:anexo}" |
| 143 | "Capitulo" | **"Capítulo"** | "en el Capitulo \ref{cap:anexo}" |

### herramientas/objetivo de la busqueda.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 4 | "ademas" | **"además"** | "Nos interesa ademas" |

### herramientas/filtrado.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 7 | "pagina" | **"página"** | "en su pagina" |
| 13 | "pagina" | **"página"** | "en su pagina tiene un video" |
| 21 | "utilizacion" | **"utilización"** | "ni utilizacion de IA" |
| 23 | "generacion" | **"generación"** | "no la generacion" |
| 33+ | "generacion" (muchas veces) | **"generación"** | Pervasivo en todo el archivo (lineas 33, 35, 37, 91, 121, 123, 124, 170, 193, 194, 217, 240, 246, etc.) |
| 35 | "experimeto" | **"experimento"** | "es un experimeto de deteccion" |
| 35 | "deteccion" | **"detección"** | "deteccion de ambiguedad" |
| 35 | "ambiguedad" | **"ambigüedad"** | "deteccion de ambiguedad" |
| 73 | "éste" | **"este"** | "de éste análisis" (adjetivo demostrativo no lleva tilde) |
| 123 | "pagina" | **"página"** | "demo en pagina sobre generacion" |
| 128 | "implementacion" | **"implementación"** | "implementacion no disponible" |
| 134 | "ambiguiedad" | **"ambigüedad"** | Typo |
| 219 | "esta" | **"está"** | "AutoStory...esta desactualizada" |
| 221 | "utilizo" | **"utilizó"** | "se utilizo un documento" |
| 240 | "Generacion" | **"Generación"** | "Generacion de Historias" |
| 246 | "incluiyen" | **"incluyen"** | Extra 'i' |
| 253, 257, 261, 271 | "inlcuida" | **"incluida"** | Letras transpuestas |
| 255 | "encontrda" | **"encontrada"** | "no fue encontrda" |
| 255 | "menera" | **"manera"** | "la menera en la primera prueba" |
| 255 | "genero" | **"generó"** | "genero las historias" |
| 257 | "genero" | **"generó"** | "genero las historias" |
| 263 | "requisistos" | **"requisitos"** | Extra 's' |
| 271 | "aceptacion" | **"aceptación"** | "criterios de aceptacion" |
| 273 | "muesta" | **"muestra"** | "se muesta el estado" |
| 369 | "mas" | **"más"** | "las herramientas mas completas" |
| 392 | "Table" | **"Tabla"** | "En la Table \ref{table:ranking-puntos}" (palabra en ingles) |

### herramientas/seleccion de herramientas.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 2 | "una conjunto" | **"un conjunto"** | Error de concordancia de genero |
| 3, 5 | "descripto" | **"descrito"** | Participio no estandar |

### central/main.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 7 | "Esta seccion se describe" | **"En esta seccion se describen"** | Falta preposicion y concordancia |
| 9 | "derivo" | **"derivó"** | "que luego derivo en ajustes" |
| 9 | "generacion" | **"generación"** | "la generacion de historias" |
| 15 | "documentos" | **"documento"** | "al documentos de requerimientos" (debe ser singular) |
| 17 | "Ademas" | **"Además"** | Falta tilde |
| 17 | "reviso" | **"revisó"** | "se reviso el prompt" |
| 41 | "como" (x2) | **"cómo"** | Pregunta indirecta: "dudas sobre como corroborar" |
| 43 | "mas" | **"más"** | "lo mas completa posible" |
| 47, 48, 50 | "Ingles" / "ingles" (x5) | **"inglés"** | Falta tilde en todas las ocurrencias |
| 74 | "analizo" | **"analizó"** | "se analizo el articulo" |
| 74, 130, 174 | "articulo" | **"artículo"** | Falta tilde |
| 174 | "Capitulo" | **"Capítulo"** | Falta tilde |

### evaluacion/main.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 5 | "permiten" | **"permite"** | Sujeto es "El estudio" (singular) |
| 5 | "demas" | **"demás"** | "con las demas historias" |
| 12 | "incoportado" | **"incorporado"** | Letras transpuestas |
| 27 | "Como" | **"Cómo"** | "Como se evalua" (pregunta indirecta) |
| 29 | "Capitulo" | **"Capítulo"** | Falta tilde |
| 33 | "clasificacion" | **"clasificación"** | Falta tilde |
| 34 | "mas" | **"más"** | "abarcan mas de una funcionalidad" |
| 36 | "caracter" | **"carácter"** | Falta tildes |
| 36 | "sintactico" | **"sintáctico"** | Falta tilde |
| 36 | "evaluacion" | **"evaluación"** | Falta tilde |
| 37 | "pragmaticos" | **"pragmáticos"** | Falta tilde |
| 37 | "deteccion" | **"detección"** | Falta tilde |
| 37 | "duplica" | **"duplicado"** | "deteccion de duplica exacta" |
| 38 | "automatica" | **"automática"** | Falta tilde |
| 72 | "sera" | **"será"** | Falta tilde |
| 72 | "entones" | **"entonces"** | Falta 'c' |
| 73 | "pordian" | **"podrían"** | Typo y falta tilde |
| 73 | "evaluacion" | **"evaluación"** | Falta tilde |
| 83 | "aceptacion" | **"aceptación"** | Falta tilde |
| 125 | "sera" | **"será"** | Falta tilde |
| 143 | "generacion" | **"generación"** | Falta tilde |
| 147 | "explico" | **"explicó"** | Falta tilde |
| 151 | "evaluacion" | **"evaluación"** | Falta tilde |
| 153 | "evaluara" | **"evaluará"** | Falta tilde |
| 155 | "sera" | **"será"** | Falta tilde |
| 155 | "deteccion" | **"detección"** | Falta tilde |

### resultados/main.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 1 | "descripta" | **"descrita"** | Participio no estandar |
| 8 | "ambiguedades" | **"ambigüedades"** | Falta dieresis |
| 381 | "Alucinacion" | **"Alucinación"** | Falta tilde (encabezado de tabla) |
| 419 | "esta" | **"está"** | "no esta presente" (verbo) |
| 421 | "esta" | **"está"** | "no esta definido" (verbo) |
| 423 | "coherente" | **"coherentes"** | "no son coherente" (falta plural) |
| 423 | "mas" | **"más"** | "es mas una restriccion" |
| 428 | "esta" | **"está"** | "la solucion generada esta libre" |
| 806 | "fué" | **"fue"** | Monosilabo no lleva tilde |
| 868 | "explusiva" | **"exclusiva"** | Typo |
| 903 | "compartamiento" | **"comportamiento"** | Typo: vocal incorrecta |
| 903 | "mas" | **"más"** | "en mas de una historia" |
| 928 | "estan" | **"están"** | "no estan directamente" |

### main.tex

| Linea | Error | Correccion | Contexto |
|-------|-------|------------|----------|
| 69 | "Revision" | **"Revisión"** | Titulo de capitulo |
| 131 | Letra suelta "o" | **Eliminar** | Caracter suelto entre secciones |

---

## 2. Repeticion de conceptos

Se encontraron **17 casos** de repeticion, clasificados por severidad.

### Severidad ALTA (duplicacion exacta o casi exacta)

#### 2.1 Parrafo duplicado consecutivo en `seleccion de herramientas.tex`

- **Ubicacion 1:** `herramientas/seleccion de herramientas.tex`, lineas 2-3
  > "Si bien el objetivo es analizar herramientas, no resulta viable realizar un analisis en profundidad de la totalidad de las soluciones obtenidas con el filtrado descrito..."
- **Ubicacion 2:** `herramientas/seleccion de herramientas.tex`, lineas 5-6
  > "Con el fin de acotar el conjunto de herramientas a analizar y concentrar el estudio en aquellas que presentan un nivel aceptable de funcionamiento, se definio un umbral minimo de seleccion..."
- **Sugerencia:** Eliminar el primer parrafo (lineas 2-3) y mantener solo el segundo, que es mas completo e incluye el valor del umbral.

#### 2.2 Frase exacta sobre "cuatro ambitos principales" de busqueda

- **Ubicacion 1:** `herramientas/main.tex`, lineas 4-5 (dentro de bloque `\begin{comment}`)
  > "La busqueda se realizo en cuatro ambitos principales: motores de busqueda generales, repositorios de codigo abierto..."
- **Ubicacion 2:** `herramientas/metodologia de busqueda.tex`, linea 17
  > (Texto identico)
- **Sugerencia:** Eliminar la version comentada en `herramientas/main.tex`.

#### 2.3 Requerimientos funcionales duplicados

- **Ubicacion 1:** `herramientas/main.tex`, lineas 4-5 (comentado)
  > "las herramientas deben basar la generacion de historias de usuario en alguna fuente de entrada..."
- **Ubicacion 2:** `herramientas/objetivo de la busqueda.tex`, lineas 3-5
  > (Texto casi identico)
- **Sugerencia:** Eliminar la version comentada en `herramientas/main.tex`.

### Severidad MEDIA (mismo concepto reformulado)

#### 2.4 Definicion de "Historia de Usuario" en 3 lugares

- `resumen.tex` (lineas 10-11): "Las historias de usuario son una herramienta utilizada por los equipos de desarrollo..."
- `revision/conceptos/main.tex` (lineas 8-9): "Una historia de usuario es un escenario: una descripcion de un uso real de un sistema..."
- `introduccion.tex` (linea 3): "las historias de usuario se utilizan como un medio de comunicacion..."
- **Sugerencia:** El resumen puede ser autocontenido. La introduccion deberia evitar re-explicar y referenciar el capitulo de antecedentes.

#### 2.5 Proposito general del proyecto repetido en 5 sitios

- `resumen.tex`, `introduccion.tex`, `herramientas/main.tex`, `herramientas/objetivo de la busqueda.tex`, `evaluacion/main.tex`
- **Sugerencia:** Mantener la declaracion completa solo en la introduccion (y resumen). En los capitulos, usar una oracion de transicion que referencie la introduccion.

#### 2.6 Papel de los LLMs / GPT en generacion de HU (4 ubicaciones)

- `resumen.tex`, `introduccion.tex`, `revision/conceptos/main.tex`, `herramientas/metodologia de busqueda.tex`
- **Sugerencia:** Definir LLM formalmente solo en `revision/conceptos/main.tex`. Desambiguar la sigla una vez en la introduccion y referenciar.

#### 2.7 Modelo INVEST explicado en 3 capitulos

- `herramientas/analisis de herramientas.tex` (lineas 83-85)
- `evaluacion/main.tex` (lineas 18-19)
- `central/prueba piloto - ClickUp.tex` (lineas 19-23)
- **Sugerencia:** Definir INVEST una sola vez (en `revision/conceptos/main.tex` o `evaluacion/main.tex`) y referenciar desde los demas.

#### 2.8 Trazabilidad definida en 4 ubicaciones

- `herramientas/objetivo de la busqueda.tex`, `herramientas/analisis de herramientas.tex`, `evaluacion/main.tex`, `central/prueba piloto - ClickUp.tex`
- **Sugerencia:** Definir formalmente en un solo lugar y referenciar desde los demas.

#### 2.9 Descripcion de ClickUp repetida

- `herramientas/filtrado.tex` (linea 79)
- `central/prueba piloto - ClickUp.tex` (lineas 1-9 y 15-17)
- **Sugerencia:** En `filtrado.tex`, limitar a una oracion indicando que pasa el filtro y referir a la prueba piloto.

#### 2.10 Deteccion de inconsistencias/ambiguedades (4 ubicaciones)

- `herramientas/objetivo de la busqueda.tex`, `herramientas/analisis de herramientas.tex`, `central/prueba piloto - ClickUp.tex`, `evaluacion/main.tex`
- **Sugerencia:** Definir la mecanica de evaluacion en un solo lugar y referenciar.

#### 2.11 Marcos SMART vs. IEEE 830 para criterios de aceptacion

- `central/prueba piloto - ClickUp.tex` usa SMART
- `herramientas/analisis de herramientas.tex` usa rasgos IEEE 830 + SMART
- **Sugerencia:** Unificar el marco. Indicar que la prueba piloto uso SMART y el marco definitivo fue refinado.

### Severidad BAJA (superposicion menor, posiblemente intencional)

#### 2.12 Criterios de inclusion/exclusion

- `herramientas/metodologia de busqueda.tex` vs. textos comentados en otros archivos de herramientas
- **Sugerencia:** Mantener solo en `metodologia de busqueda.tex`. Si se descomentaran los otros, seria repeticion alta.

#### 2.13 Criterios de aceptacion (definicion vs. evaluacion)

- `revision/conceptos/main.tex` vs. `herramientas/analisis de herramientas.tex` vs. `central/prueba piloto - ClickUp.tex`
- **Sugerencia:** Unificar terminologia y conectar los marcos explicitamente.

#### 2.14 Documentos de prueba / dominio hospitalario

- Se introducen como nuevos en `herramientas/analisis de herramientas.tex`, `central/prueba piloto - ClickUp.tex`, `central/main.tex`
- **Sugerencia:** Definir en un solo lugar y referenciar.

#### 2.15 Prompt simple en espanol e ingles

- `central/main.tex` incluye ambas versiones completas
- **Sugerencia:** Mostrar solo la version final en ingles (que es la usada) y mencionar que fue traducida.

#### 2.16 Cobertura y Sentence-BERT explicados dos veces

- `evaluacion/main.tex` lineas 72-74 y 87-95
- **Sugerencia:** Presentar la herramienta una vez y luego los dos usos como subsecciones diferenciadas.

---

## 3. Faltas de referencia

### 3.1 Citas faltantes (afirmaciones sin \cite{})

| ID | Archivo | Linea | Texto problematico | Sugerencia |
|----|---------|-------|--------------------|------------|
| H | evaluacion/main.tex | 10 | QUSF introducido sin cita | Agregar `\cite{aqusa-paper}` (existe en .bib) |
| I | evaluacion/main.tex | 17 | Dimensiones de QUSF sin cita | Agregar `\cite{aqusa-paper}` |
| J | evaluacion/main.tex | 12, 18-19 | INVEST mencionado sin cita | Agregar `\cite{wake_invest}` |
| K | evaluacion/main.tex | 33 | "El criterio INVEST es analizado..." | Agregar `\cite{wake_invest}` |
| L | central/prueba piloto - ClickUp.tex | 19 | "perspectiva INVEST" sin cita | Agregar `\cite{wake_invest}` |
| M | central/prueba piloto - ClickUp.tex | 25 | "enfoque SMART" sin cita | Agregar `\cite{doran_smart}` (existe en .bib) |
| O | evaluacion/main.tex | 31-33 | Escala Likert sin referencia | Agregar cita a Likert (1932) o fuente metodologica |
| P | revision/conceptos/main.tex | 46-66 | Tabla de definiciones de IA sin cita en caption | Agregar `\cite{IABook}` en el `\caption` |
| Q | revision/conceptos/main.tex | 10 | Plantilla "Como un tipo de usuario..." sin cita | Agregar `\cite{cohn_user_stories_applied}` |
| R | introduccion.tex | 3-6 | Afirmaciones sobre metodologias agiles sin citas | Agregar citas de respaldo |
| S | evaluacion/main.tex | 21 | "fenomeno conocido como alucinacion" sin cita | Agregar referencia sobre alucinaciones en LLMs |
| T | central/main.tex | ~174 | FActScore mencionado sin referencia | Agregar referencia al paper original |
| U | central/main.tex | ~149, ~174 | BERTScore sin cita en texto activo | Agregar `\cite{zhang2020bertscore}` |

### 3.2 Referencias rotas / Labels problematicos

| ID | Archivo | Linea | Problema | Sugerencia |
|----|---------|-------|----------|------------|
| A | herramientas/filtrado.tex | 40 | `\cite{qualityModeller}` vs clave `qualitymodeller` | Cambiar a `\cite{qualitymodeller}` **(solo si usan BibLaTeX; BibTeX estandar es case-insensitive)** |
| B | herramientas/filtrado.tex | 48 | `\cite{geneuS}` vs clave `geneus` | Cambiar a `\cite{geneus}` **(solo si usan BibLaTeX; BibTeX estandar es case-insensitive)** |
| C | main.tex | 99, 111 | Label duplicado `fig:eval-arquitectura` (comentado) | Dar nombres distintos |
| D | herramientas/seleccion de herramientas.tex | 17, 69 | Label duplicado `tab:herramientas_hu` (uno comentado) | Eliminar duplicado |
| E | revision/conceptos/main.tex | 40 | `\label{table:defIA}` antes de `\caption` | Mover label despues del caption |
| F | herramientas/filtrado.tex | 166, 213 | Labels fuera de float | Mover dentro de longtable, despues de caption |
| G | herramientas/filtrado.tex | 315 | Label fuera de longtable | Mover dentro, despues de caption |

### 3.3 Referencias cruzadas faltantes

| ID | Archivo | Linea | Problema | Sugerencia |
|----|---------|-------|----------|------------|
| V | evaluacion/main.tex | 143 | "en la seccion" sin `\ref{}` | Agregar `\ref{...}` al label correspondiente |
| W | resultados/main.tex | 873 | "Capítulo 5" hardcodeado (tiene tilde, pero el numero esta hardcodeado) | Reemplazar con `Capítulo \ref{cap:evaluacion}` |
| X | resultados/main.tex | 892 | "en el Anexo" sin referencia precisa | Agregar `\ref{}` al anexo especifico |
| Y | resultados/main.tex | 4 | "prompt \ref{fig:prompt_simple}" | Reformular como "en la Figura \ref{fig:prompt_simple}" |

### 3.4 Entradas de bibliografia no utilizadas

Las siguientes entradas en `referencias.bib` nunca se citan en los archivos .tex activos:

| Clave | Descripcion | Accion sugerida |
|-------|-------------|-----------------|
| `CitekeyArticle` | Ejemplo placeholder (Cohen, 1963) | **Eliminar** |
| `CitekeyBook` | Ejemplo placeholder (Susskind) | **Eliminar** |
| `CitekeyInproceedings` | Ejemplo placeholder (Holleis) | **Eliminar** |
| `CitekeyManual` | Ejemplo placeholder (R Core Team) | **Eliminar** |
| `CitekeyMisc` | Ejemplo placeholder (NASA Pluto) | **Eliminar** |
| `requirementlinter2` | Duplicado de `popal` | **Eliminar** |
| `aqusa-paper` | QUSF (Lucassen) | **Citar en evaluacion/main.tex** (ver 3.1) |
| `zhang2020bertscore` | BERTScore | **Citar en central/main.tex** (ver 3.1) |

### 3.5 Otros problemas

| ID | Archivo | Linea | Problema | Sugerencia |
|----|---------|-------|----------|------------|
| Z | referencias.bib | 456 | Comentario con `//` (sintaxis invalida en BibTeX) | Cambiar a `%` |
| AA | main.tex | 131 | Letra suelta "o" entre secciones | Eliminar |

---

## Resumen general

| Categoria | Cantidad |
|-----------|----------|
| Errores ortograficos | 109 instancias |
| Repeticiones de conceptos (alta) | 3 |
| Repeticiones de conceptos (media) | 8 |
| Repeticiones de conceptos (baja) | 6 |
| Citas faltantes | 14 |
| Referencias rotas / labels problematicos | 7 |
| Referencias cruzadas faltantes | 4 |
| Entradas bib no utilizadas | 8 |
| Otros problemas | 2 |
