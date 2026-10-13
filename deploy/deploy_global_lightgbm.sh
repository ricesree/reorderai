#!/usr/bin/env bash
# Deploy global LightGBM model + API wiring on the Azure API VM.
# Run ON the VM after code is on disk (git pull or scp).
#
#   cd /opt/reorder-ai && sudo bash deploy/deploy_global_lightgbm.sh
#
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/reorder-ai}"
SERVICE="${SERVICE:-reorder-ai}"
MODEL_REL="models/global_lightgbm_sales_model.joblib"

cd "$APP_DIR"

echo "==> App dir: $APP_DIR"
if [[ -d .git ]]; then
  echo "==> git pull origin main"
  git pull origin main
else
  echo "==> (not a git repo — skipping pull; ensure files are already copied)"
fi

if grep -qE '^AZURE_STORAGE_CONNECTION_STRING=.+' "$APP_DIR/.env" 2>/dev/null; then
  echo "==> Blob storage configured — model loads from container blob global/$(basename "$MODEL_REL")"
  echo "    Upload it first, e.g. from your laptop:"
  echo "      az storage blob upload --connection-string \"\$AZURE_STORAGE_CONNECTION_STRING\" \\"
  echo "        -c models -n global/global_lightgbm_sales_model.joblib -f $MODEL_REL --overwrite"
elif [[ ! -f "$MODEL_REL" ]]; then
  echo "ERROR: missing $APP_DIR/$MODEL_REL"
  echo "Copy it first, e.g. from your laptop:"
  echo "  scp models/global_lightgbm_sales_model.joblib USER@VM:/opt/reorder-ai/models/"
  exit 1
else
  echo "==> Model present: $(du -h "$MODEL_REL" | awk '{print $1}')"
fi

VENV_PIP=""
if [[ -x "$APP_DIR/.venv/bin/pip" ]]; then
  VENV_PIP="$APP_DIR/.venv/bin/pip"
elif [[ -x "$APP_DIR/venv/bin/pip" ]]; then
  VENV_PIP="$APP_DIR/venv/bin/pip"
fi

if [[ -n "$VENV_PIP" ]]; then
  echo "==> pip install joblib scikit-learn lightgbm azure-storage-blob"
  "$VENV_PIP" install -q 'joblib>=1.3.0' 'scikit-learn>=1.3.0' 'lightgbm>=4.0.0' 'azure-storage-blob>=12.19.0'
else
  echo "==> (no .venv found — skip pip; install joblib manually if needed)"
fi

ENV_FILE="$APP_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  if ! grep -q '^USE_GLOBAL_LIGHTGBM=' "$ENV_FILE"; then
    echo 'USE_GLOBAL_LIGHTGBM=1' >> "$ENV_FILE"
    echo "==> appended USE_GLOBAL_LIGHTGBM=1 to .env"
  else
    sed -i 's/^USE_GLOBAL_LIGHTGBM=.*/USE_GLOBAL_LIGHTGBM=1/' "$ENV_FILE"
    echo "==> set USE_GLOBAL_LIGHTGBM=1 in .env"
  fi
else
  echo "WARN: no .env at $ENV_FILE"
fi

echo "==> smoke-load model"
python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(".env", override=True)
from v2.forecasting import model_storage
from v2.forecasting.global_lightgbm_predictor import model_ready, model_path
print("model_path", model_path(), "(blob)" if model_storage.blob_enabled() else "(local)")
print("model_ready", model_ready())
if not model_ready():
    raise SystemExit("model failed to load")
print("OK")
PY

echo "==> restart $SERVICE"
sudo systemctl restart "$SERVICE"
sleep 2
sudo systemctl is-active "$SERVICE"
curl -sf "http://127.0.0.1:8000/api/health" && echo || true
echo "==> done — detect-order sales_series.forecast now uses global_lightgbm when ready"
