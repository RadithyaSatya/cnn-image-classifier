"""Validate model shapes and convert genuine outputs to probabilities."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.model_loader import ModelError


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    inference_ms: float


def _validate_input(model, batch):
    shape = model.input_shape
    if isinstance(shape, list):
        raise ModelError("Model memiliki beberapa input; aplikasi membutuhkan satu input gambar.")
    if len(shape) != 4 or any(expected is not None and expected != actual for expected, actual in zip(shape[1:], batch.shape[1:])):
        raise ModelError(f"Dimensi input tidak sesuai: model {shape}, gambar {batch.shape}.")


def predict(model, batch, config):
    _validate_input(model, batch)
    start = perf_counter()
    try:
        raw = model.predict(batch, verbose=0)
    except Exception as exc:
        raise ModelError(f"Proses inferensi gagal: {exc}") from exc
    elapsed_ms = (perf_counter() - start) * 1000
    if isinstance(raw, (list, tuple)):
        raise ModelError("Model memiliki beberapa output; aplikasi membutuhkan satu output.")
    values = np.asarray(raw)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ModelError(f"Bentuk output tidak sesuai: {values.shape}; diharapkan satu baris prediksi.")
    values = values[0].astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ModelError("Output model mengandung nilai tidak valid (NaN atau infinity).")
    names = config["class_names"]
    if config["output_type"] == "binary_sigmoid":
        if len(values) != 1:
            raise ModelError(f"Output binary_sigmoid harus satu nilai, ditemukan {len(values)}.")
        positive = float(values[0])
        if not 0 <= positive <= 1:
            raise ModelError("Output sigmoid harus berupa probabilitas antara 0 dan 1.")
        probabilities = np.array([1 - positive, positive])
    else:
        if len(values) != len(names):
            raise ModelError(f"Jumlah output model ({len(values)}) berbeda dari class_names ({len(names)}).")
        if np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(), 1, atol=1e-3):
            raise ModelError("Output softmax harus berupa probabilitas 0–1 dengan total mendekati 1.")
        probabilities = values
    index = int(np.argmax(probabilities))
    return Prediction(names[index], float(probabilities[index]), dict(zip(names, map(float, probabilities))), elapsed_ms)
