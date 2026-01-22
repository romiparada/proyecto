import re
from sentence_transformers import util


class EvaluadorCriteriosAceptacion:

    def __init__(self, sbert_model, umbral_escenario=0.75):
        self.model = sbert_model
        self.umbral_escenario = umbral_escenario

        self.terminos_ambiguos = [
            "fast", "quick", "adequate", "appropriate",
            "correct", "proper", "user-friendly",
            "efficient", "intuitive", "as soon as possible",
            "reasonable time"
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
