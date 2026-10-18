"""Model configuration and cached Keras loading."""
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "model_config.json"


class ModelError(Exception):
    """A model or its configuration cannot be used safely."""


def load_config():
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"Konfigurasi model tidak dapat dibaca: {exc}") from exc

    size = config.get("image_size")
    names = config.get("class_names")
    if not (isinstance(size, list) and len(size) == 2 and all(type(v) is int and v > 0 for v in size)):
        raise ModelError("image_size harus berupa [tinggi, lebar] berisi bilangan positif.")
    if not (isinstance(names, list) and len(names) >= 2 and all(isinstance(n, str) and n.strip() for n in names)):
        raise ModelError("class_names harus berisi minimal dua nama kelas yang tidak kosong.")
    if len(set(names)) != len(names):
        raise ModelError("Nama kelas dalam class_names harus unik.")
    if config.get("color_mode") not in ("rgb", "grayscale"):
        raise ModelError("color_mode harus 'rgb' atau 'grayscale'.")
    if config.get("preprocessing") not in ("none", "rescale_0_1", "minus_one_to_one"):
        raise ModelError("preprocessing harus 'none', 'rescale_0_1', atau 'minus_one_to_one'.")
    if config.get("output_type") not in ("multiclass_softmax", "binary_sigmoid"):
        raise ModelError("output_type harus 'multiclass_softmax' atau 'binary_sigmoid'.")
    if config["output_type"] == "binary_sigmoid" and len(names) != 2:
        raise ModelError("binary_sigmoid memerlukan tepat dua class_names.")
    limit = config.get("max_upload_mb")
    if type(limit) not in (int, float) or not 0 < limit <= 10:
        raise ModelError("max_upload_mb harus lebih dari 0 dan maksimal 10 (batas server).")
    path = config.get("model_path")
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ModelError("model_path harus berupa path relatif di dalam folder models/.")
    relative = Path(path)
    if relative.parts[0] != "models" or relative.suffix.lower() not in (".keras", ".h5"):
        raise ModelError("model_path harus menunjuk file .keras atau .h5 di folder models/.")
    return config


def model_file(config):
    return ROOT / config["model_path"]


@st.cache_resource(show_spinner=False)
def _cached_model(path, modified_ns):
    try:
        from tensorflow.keras.models import load_model
    except (ImportError, OSError) as exc:
        raise ModelError(f"TensorFlow tidak dapat diimpor: {exc}") from exc
    try:
        return load_model(path, compile=False)
    except Exception as exc:
        raise ModelError(f"File model tidak dapat dimuat: {exc}") from exc


def load_model_cached(config):
    path = model_file(config)
    if not path.is_file():
        raise ModelError(f"Model Not Available: tambahkan file model ke {config['model_path']}.")
    return _cached_model(str(path), path.stat().st_mtime_ns)
