# Entorno
python3 -m venv venv
source venv/bin/activate

# Instalacion

instalar librerías necesarias:
pip install -r requirements.txt

# Uso

## Alineamiento automatico
- python alineador.py
    - resultados/similitud_historias_all.json
    - resultados/similitud_criterios_all.json
    - resultados/similitud_funcionalidades_all.json
- python alineador.py --top 3
    - resultados/similitud_historias_top_3.json
    - resultados/similitud_criterios_top_3.json
    - resultados/similitud_funcionalidades_top_3.json

## Evaluacion por consola a partir del alineamiento
- python evaluador_historias.py
    - resultados/evaluador_historias.json
- python evaluador_criterios.py (se corre a partir del resultado del evaluador_historias.py)
    - resultados/evaluador_criterios.json
- python evaluador_funcionalidades.py
    - resultados/evaluador_funcionalidades.json

