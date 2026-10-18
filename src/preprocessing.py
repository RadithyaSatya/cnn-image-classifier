"""Upload validation and model-specific image preprocessing."""
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageError(Exception):
    """The uploaded file is unsuitable for inference."""


def validate_image(uploaded_file, max_upload_mb):
    if uploaded_file is None:
        raise ImageError("Pilih gambar terlebih dahulu.")
    data = uploaded_file.getvalue()
    if len(data) > max_upload_mb * 1024 * 1024:
        raise ImageError(f"Ukuran file maksimal {max_upload_mb:g} MB.")
    if not data:
        raise ImageError("File gambar kosong.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ImageError("Format gambar harus JPG, JPEG, atau PNG.")
            if image.width * image.height > 40_000_000:
                raise ImageError("Dimensi gambar terlalu besar (maksimal 40 megapiksel).")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageError(f"Gambar tidak valid atau rusak: {exc}") from exc


def preprocess_image(image, config):
    mode = "RGB" if config["color_mode"] == "rgb" else "L"
    height, width = config["image_size"]
    resized = image.convert(mode).resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if mode == "L":
        array = array[..., np.newaxis]
    if config["preprocessing"] == "rescale_0_1":
        array /= 255.0
    elif config["preprocessing"] == "minus_one_to_one":
        array = array / 127.5 - 1.0
    return np.expand_dims(array, axis=0)
