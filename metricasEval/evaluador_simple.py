from cargador_datos import (
    cargar_historias,
    guardar_txt
)
import argparse
import os


from sentence_transformers  import SentenceTransformer, CrossEncoder

model = SentenceTransformer("all-mpnet-base-v2")

encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")

def calcular_sbert(historias_esperadas, historias_generadas):
    embeddings_esperadas = model.encode(historias_esperadas)
    embeddings_generadas = model.encode(historias_generadas)
    similarities = model.similarity(embeddings_esperadas, embeddings_generadas)
    return similarities

def calcular_ranking(historias_esperada, historias_generadas):
    ranks = encoder.rank(historias_esperada, historias_generadas)
    print("Query: ", historias_esperada)
    for rank in ranks:
        print(f"{rank['score']:.2f}\t{historias_generadas[rank['corpus_id']]}")

def main():
    parser = argparse.ArgumentParser(description="Evaluacion experimental de historias de usuario")
    
    parser.add_argument("--caso", required=True)
    parser.add_argument("--salida", default="resultados")
    parser.add_argument("--casos-dir", default="casos_prueba")
    parser.add_argument("--aspectos", default="casos_prueba/aspectos_hospital.json")
    
    args = parser.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    ruta_generadas = os.path.join(args.casos_dir, args.caso, "generadas.txt")
    historias_generadas = cargar_historias(ruta_generadas)
    print(f"Historias generadas: {len(historias_generadas)}")
    
    ruta_esperadas = os.path.join(args.casos_dir, args.caso, "esperadas.txt")
    historias_esperadas = cargar_historias(ruta_esperadas)
    print(f"Historias esperadas: {len(historias_esperadas)}")

    similarities = calcular_sbert(historias_esperadas, historias_generadas)
    for i, row in enumerate(similarities):
        print(f"Similarities for expected story {i + 1}:")
        for j, value in enumerate(row):
            if value > 0.6:  # Umbral de similitud
                print(f"{j + 1}: {value.item()}")
    # guardar_txt(similarities, ruta_txt)
    
    for i, row in enumerate(historias_esperadas):
        print(f"Ranking for expected story {i + 1}:")
        calcular_ranking(historias_esperadas[i], historias_generadas)

if __name__ == "__main__":
    main()