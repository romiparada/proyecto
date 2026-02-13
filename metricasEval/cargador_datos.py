import os
import json
from typing import List, Dict, Tuple


def cargar_historias(ruta: str) -> List[str]:    
    with open(ruta, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
    
    historias = [linea.strip() for linea in lineas if linea.strip()]
    return historias


def cargar_aspectos(ruta: str) -> Dict[str, List[str]]:
    with open(ruta, 'r', encoding='utf-8') as f:
        aspectos = json.load(f)   
    return aspectos

def cargar_criterios(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)

def cargar_metadata(dir_caso: str) -> Dict:
    ruta_metadata = os.path.join(dir_caso, "metadata.json")
    
    if not os.path.exists(ruta_metadata):
        raise FileNotFoundError(f"metadata.json no encontrado en: {dir_caso}")
    
    with open(ruta_metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    return metadata



def cargar_conjuntos_diversidad(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    conjuntos = []
    nombres = []

    for modelo in data.get("modelos", []):
        nombres.append(modelo["nombre"])
        conjuntos.append(modelo["historias"])

    return nombres, conjuntos


def descubrir_casos(dir_casos: str) -> List[str]:
    if not os.path.isdir(dir_casos):
        return []
    
    casos = []
    for item in os.listdir(dir_casos):
        ruta_item = os.path.join(dir_casos, item)
        if os.path.isdir(ruta_item) and item.startswith("caso_"):
            casos.append(item)
    
    return sorted(casos)


def guardar_json(resultados: Dict, ruta: str):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)


def guardar_txt(contenido: str, ruta: str):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(contenido)
