# Entorno
python3 -m venv venv
source venv/bin/activate

# Instalacion

instalar librerías necesarias:
pip install -r requirements.txt

# Uso

## Alineamiento automatico historias y funcionalidades
- python alineador.py
    - resultados/similitud_historias_all.json
    - resultados/similitud_funcionalidades_all.json
    - resultados/similitud_historias_criterios_all.json
- python alineador.py --top 3
    - resultados/similitud_historias_top_3.json
    - resultados/similitud_historias_criterios_top_3.json
    - resultados/similitud_funcionalidades_top_3.json

## Evaluacion por consola a partir del alineamiento
### Historias
- python evaluador_historias.py
    - resultados/evaluador_historias.json
### Funcionalidades
- python evaluador_funcionalidades.py
- resultados/evaluador_funcionalidades.json
### Criterios
- python alineador_criterios.py (Utiliza similitud_hisotiras_criterios.json y evaluador_historias.py)
    - resultados/similitud_historias_alineadas_criterios_all.json (criterios incluyen id y desc de la historia)
    - resultados/similitud_historias_alineadas_criterios_simple_all.json
- python evaluador_criterios.py 
    - resultados/evaluador_criterios.json
