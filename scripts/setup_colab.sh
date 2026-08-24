#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/content/proyecto_estado_arte}"
VENV_DIR="${VENV_DIR:-/content/venv_estado_arte}"

echo "Proyecto: $PROJECT_DIR"
echo "Venv: $VENV_DIR"

if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "ERROR: no existe $PROJECT_DIR/requirements.txt"
  exit 1
fi

if [ ! -f "$PROJECT_DIR/constraints-colab.txt" ]; then
  echo "ERROR: no existe $PROJECT_DIR/constraints-colab.txt"
  exit 1
fi

python3 -m pip install -q --upgrade virtualenv

rm -rf "$VENV_DIR"

python3 -m virtualenv --pip bundle "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

# -c constraints-colab.txt: refuerzo explícito de las versiones
# funcionales confirmadas -- aunque requirements.txt ya está
# exacto-pineado (==), esta restricción adicional evita que una
# futura relajación de requirements.txt (ej. a rangos >=) instale
# silenciosamente una versión distinta a la validada, en particular
# chromadb (nunca 0.5.x -- el índice persistente real no podía
# abrirse con 0.5.23, KeyError('_type')).
"$PYTHON" -m pip install --no-cache-dir \
  -c "$PROJECT_DIR/constraints-colab.txt" \
  -r "$PROJECT_DIR/requirements.txt"

"$PYTHON" -m pip check

"$PYTHON" - <<'PY'
import chromadb
import transformers
import tokenizers
import sentence_transformers
import numpy
import pandas
import sklearn
import matplotlib

print("chromadb:", chromadb.__version__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("sentence-transformers:", sentence_transformers.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)

assert chromadb.__version__ == "1.5.9"
assert transformers.__version__ == "4.46.3"
assert tokenizers.__version__ == "0.20.3"

print("ENTORNO VALIDADO")
PY
