from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


# load annotations
def load_annotations(csv_path: str | Path) -> pd.DataFrame:
    """Load training annotations from csv file"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} annotation rows across {df.image_id.nunique()} images")
    print(df.head())
    print(f"Number of nans in column:")
    print(df.isna().sum())
    return df

def load_dicom_as_array(dicom_path: str | Path) -> np.ndarray:
    """Load a DICOM file and return a normalized uint8 grayscale image."""
    dicom = pydicom.dcmread(dicom_path)
    array = apply_voi_lut(dicom.pixel_array, dicom)

    # Some CXR DICOMs are stored inverted (MONOCHROME1)
    if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        array = np.amax(array) - array

    array = array.astype(np.float32)
    array = cv2.normalize(array, np.empty_like(array, dtype=np.uint8), 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return array
