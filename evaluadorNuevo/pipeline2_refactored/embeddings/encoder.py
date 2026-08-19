from ..semantic_encoder import SemanticEncoder


def create_encoder(model_name: str, device: str = None) -> SemanticEncoder:
    print("Cargando modelo SBERT...")
    encoder = SemanticEncoder(model_name=model_name, device=device)
    print(f"  Dispositivo : {encoder.device}")
    print(f"  Modelo      : {encoder.model_name}")
    return encoder
