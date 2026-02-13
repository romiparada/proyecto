import os
import sys

# FIX Python 3.8+ DLL loading (Windows)
if sys.platform == "win32":
    dll_path = os.path.join(sys.prefix, "Lib", "site-packages", "numpy.libs")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)

    dll_path2 = os.path.join(sys.prefix, "Lib", "site-packages", "scipy.libs")
    if os.path.exists(dll_path2):
        os.add_dll_directory(dll_path2)
import re
from sentence_transformers import util


class EvaluadorCriteriosAceptacion:

    def __init__(self, sbert_model, umbral_escenario=0.75):
        self.model = sbert_model
        self.umbral_escenario = umbral_escenario


        self.terminos_ambiguos = [
            "fast", "quick", "adequate", "appropriate",
            "correct", "proper", "user-friendly",
            "efficient", "intuitive",
            "as soon as possible", "reasonable time"
        ]

        self.patrones_condicion = [
            r"\bif\b",
            r"\bwhen\b",
            r"\bgiven\b"
        ]

        self.patrones_resultado = [
            r"\bshall\b",
            r"\bmust\b",
            r"\bdisplays?\b",
            r"\breturns?\b",
            r"\bshows?\b",
            r"\bcreates?\b",
            r"\bgenerates?\b",
            r"\bupdates?\b",
            r"\bstores?\b",
            r"\bsaves?\b",
            r"\bcalculates?\b",
            r"\bvalidates?\b",
            r"\ballows?\b",
            r"\bprevents?\b",
            r"\bnotifies?\b"
        ]

    def evaluar_completitud_escenarios(self, ca_generados, ca_esperados):
        if not ca_esperados:
            return 1.0, []

        if not ca_generados:
            return 0.0, []

        emb_gen = self.model.encode(ca_generados, convert_to_tensor=True)
        emb_exp = self.model.encode(ca_esperados, convert_to_tensor=True)

        detalle = []
        cubiertos = 0

        for i, emb_e in enumerate(emb_exp):
            sims = util.cos_sim(emb_e, emb_gen)[0]
            max_sim = sims.max().item()
            cubierto = max_sim >= self.umbral_escenario

            if cubierto:
                cubiertos += 1

            detalle.append({
                "criterio_esperado": ca_esperados[i],
                "max_similitud": float(max_sim),
                "cubierto": cubierto
            })

        score = cubiertos / len(ca_esperados)
        return score, detalle


    def evaluar_ambiguedad(self, ca_generados):
        if not ca_generados:
            return 1.0, []

        detalle = []
        ambiguos = 0

        for ca in ca_generados:
            texto = ca.lower()
            encontrados = [t for t in self.terminos_ambiguos if t in texto]

            es_ambiguo = len(encontrados) > 0
            if es_ambiguo:
                ambiguos += 1

            detalle.append({
                "criterio": ca,
                "ambiguo": es_ambiguo,
                "terminos": encontrados
            })

        score = 1 - (ambiguos / len(ca_generados))
        return score, detalle

    def evaluar(self, ca_generados, ca_esperados):
        comp_score, comp_detalle = self.evaluar_completitud_escenarios(
            ca_generados, ca_esperados
        )

        amb_score, amb_detalle = self.evaluar_ambiguedad(ca_generados)

        return {
            "completitud_escenarios": {
                "score": comp_score,
                "detalle": comp_detalle
            },
            "ambiguedad": {
                "score": amb_score,
                "detalle": amb_detalle
            }
        }
    

    def evaluar_verificabilidad(self, ca_generados):
        """
        Evalúa verificabilidad de criterios de aceptación.

        Score = promedio de criterios verificables.
        Un criterio es verificable si:
            - Tiene condición clara
            - Tiene resultado observable
            - No contiene términos ambiguos
        """

        if not ca_generados:
            return 0.0, []

        detalle = []
        verificables = 0

        for ca in ca_generados:
            texto = ca.lower()

            # condición
            tiene_condicion = any(
                re.search(p, texto) for p in self.patrones_condicion
            )

            # resultado observable
            tiene_resultado = any(
                re.search(p, texto) for p in self.patrones_resultado
            )

            # ambigüedad
            ambiguos = [t for t in self.terminos_ambiguos if t in texto]
            es_ambiguo = len(ambiguos) > 0

            # criterio verificable
            es_verificable = (
                tiene_condicion and
                tiene_resultado and
                not es_ambiguo
            )

            if es_verificable:
                clasificacion = "verificable"
            elif (tiene_condicion or tiene_resultado) and not es_ambiguo:
                clasificacion = "parcialmente_verificable"
            else:
                clasificacion = "no_verificable"

            if es_verificable:
                verificables += 1

            detalle.append({
                "criterio": ca,
                "tiene_condicion": tiene_condicion,
                "tiene_resultado_observable": tiene_resultado,
                "terminos_ambiguos": ambiguos,
                "verificable": es_verificable,
                "clasificacion": clasificacion
            })

        score = verificables / len(ca_generados)

        return score, detalle
