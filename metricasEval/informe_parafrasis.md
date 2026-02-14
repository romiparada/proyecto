# Robustez del evaluador ante paráfrasis

## Objetivo

Se busca determinar si las métricas semánticas utilizadas (SBERT y Cross-Encoder) mantienen la capacidad de alinear correctamente historias de usuario cuando estas se reformulan con sinónimos y estructuras distintas, pero conservando el mismo significado.

Para eso se tomaron los dos casos de evaluación construidos (división y fusión) y se generaron versiones parafraseadas de las historias generadas, manteniendo las esperadas sin cambios como referencia fija.

---

## Diseño del experimento

Se evaluaron cuatro variantes agrupadas en dos casos:

**Caso división (2 generadas → 1 esperada):** la generación divide una historia esperada amplia en dos más granulares. Se comparan 9 historias generadas contra 5 esperadas.

**Caso fusión (1 generada → 2 esperadas):** la generación fusiona dos historias esperadas granulares en una sola amplia. Se comparan 5 historias generadas contra 8 esperadas.

En ambos casos, las versiones parafraseadas reemplazan términos como *administrative staff* por *receptionist*, *doctor* por *physician*, *pharmacist* por *pharmacy technician*, *register* por *enter/log/add*, *appointments* por *visits*, o *stock and expiration date* por *quantity on hand and shelf life*.

---

## Resultados

### Caso división — SBERT

Los scores bajan con la paráfrasis, pero el orden relativo se preserva en todos los casos.

| Esperada | Generadas correctas | Original | Paráfrasis | Caída |
|----------|-------------------|----------|------------|-------|
| Registrar pacientes | Gen 1, Gen 2 | 0.88 / 0.85 | 0.77 / 0.70 | −0.11 / −0.15 |
| Buscar pacientes | Gen 3 | 0.999 | 0.77 | −0.23 |
| Diagnósticos y tratamientos | Gen 4, Gen 5 | 0.77 / 0.84 | 0.75 / 0.73 | −0.02 / −0.11 |
| Medicamentos e insumos | Gen 6, Gen 7 | 0.97 / 0.95 | 0.84 / 0.80 | −0.13 / −0.15 |
| Gestión de citas | Gen 8, Gen 9 | 0.77 / 0.91 | <0.60 / 0.77 | perdida / −0.14 |

### Caso división — Cross-Encoder

| Esperada | Original | Paráfrasis | Caída |
|----------|----------|------------|-------|
| Registrar pacientes | 0.79 | 0.66 | −0.13 |
| Buscar pacientes | 0.99 | 0.75 | −0.24 |
| Diagnósticos y tratamientos | 0.86 | 0.81 | −0.05 |
| Medicamentos e insumos | 0.89 | 0.74 | −0.15 |
| Gestión de citas | 0.97 | 0.66 | −0.31 |

### Caso fusión — SBERT (solo versión parafraseada)

| Esperada | Generada correcta | Similitud | Sobre 0.60 |
|----------|------------------|-----------|-----------|
| Registrar datos personales | Gen 1 | 0.75 | Sí |
| Registrar antecedentes | Gen 1 | 0.72 | Sí |
| Registrar medicamentos | Gen 2 | 0.68 | Sí |
| Registrar insumos | Gen 2 | 0.76 | Sí |
| Cancelar cita | Gen 3 | 0.70 | Sí |
| Editar cita | Gen 3 | 0.65 | Sí |
| Registrar diagnóstico | Gen 4 | 0.81 | Sí |
| Buscar pacientes | Gen 5 | 0.79 | Sí |
