#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VRROBO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAVILA_REPO="${NAVILA_REPO:-$(cd "${VRROBO_DIR}/../.." && pwd)/NaVILA-main}"

if [[ ! -d "${NAVILA_REPO}" ]]; then
    echo "NaVILA repo not found: ${NAVILA_REPO}" >&2
    echo "Set NAVILA_REPO=/path/to/NaVILA-main and rerun." >&2
    exit 1
fi

python -m pip install --upgrade pip

# Keep Isaac Lab's torch/torchvision untouched; NaVILA's pyproject pins its own torch versions.
python -m pip install -e "${NAVILA_REPO}" --no-deps
python -m pip install \
    "transformers==4.37.2" \
    "accelerate==0.27.2" \
    "peft==0.9.0" \
    "bitsandbytes==0.49.2" \
    "tokenizers>=0.15.2" \
    "sentencepiece==0.1.99" \
    "shortuuid" \
    "einops==0.6.1" \
    "einops-exts==0.0.4" \
    "timm==0.9.12" \
    "opencv-python==4.8.0.74" \
    "decord==0.6.0" \
    "huggingface_hub" \
    "deepspeed==0.9.5" \
    "loguru" \
    "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"

SITE_PKG_PATH="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp -rv "${NAVILA_REPO}/llava/train/transformers_replace/"* "${SITE_PKG_PATH}/transformers/"
cp -rv "${NAVILA_REPO}/llava/train/deepspeed_replace/"* "${SITE_PKG_PATH}/deepspeed/"

python - <<'PY'
from pathlib import Path
import site

path = Path(site.getsitepackages()[0]) / "transformers" / "integrations" / "bitsandbytes.py"
text = path.read_text()
old_signature = "def set_module_quantized_tensor_to_device(module, tensor_name, device, value=None, quantized_stats=None):"
new_signature = (
    "def set_module_quantized_tensor_to_device(\n"
    "    module, tensor_name, device, value=None, quantized_stats=None, fp16_statistics=None\n"
    "):"
)
if old_signature in text:
    text = text.replace(old_signature, new_signature, 1)
    marker = "    # Recurse if needed\n"
    text = text.replace(
        marker,
        "    if quantized_stats is None and fp16_statistics is not None:\n"
        "        quantized_stats = {\"SCB\": fp16_statistics}\n\n"
        f"{marker}",
        1,
    )
    path.write_text(text)
PY

python "${SCRIPT_DIR}/download_navila_weights.py" --model_id "${NAVILA_MODEL_ID:-a8cheng/navila-llama3-8b-8f}"
