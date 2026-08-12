"""DICOM decoding. This is the part of the pipeline most likely to change
(different modality, different normalization) -- it's isolated here and
only used through the `decode_fn` hook in `storage.py`, so swapping it
doesn't require touching orchestration or storage code."""

from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def load_dicom_as_array(dicom_path: Path) -> np.ndarray:
    dicom = pydicom.dcmread(dicom_path)
    array = apply_voi_lut(dicom.pixel_array, dicom)
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = np.amax(array) - array
    array = array.astype(np.float32)
    dst = np.empty_like(array, dtype=np.uint8)
    array = cv2.normalize(array, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return array


def load_and_resize_image(raw_path: Path, img_size: int) -> tuple:
    """Returns (resized_uint8_array, orig_h, orig_w). Original dimensions
    are returned so annotation boxes (in original pixel space) can be
    rescaled to match."""
    array = load_dicom_as_array(raw_path)
    orig_h, orig_w = array.shape
    resized = cv2.resize(array, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8, copy=False), orig_h, orig_w
