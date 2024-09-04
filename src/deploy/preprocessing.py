import numpy as np

from pipeline.transforms import UniformSelectionTransform

transformation = None
MAP_VALUE_MAPPER = {
    1.0: 0.66,
    1.1: 0.63,
    1.2: 0.57,
    1.3: 0.57,
    1.4: 0.54,
    1.5: 0.50,
    1.6: 0.48,
    1.7: 0.44,
    1.8: 0.42,
    1.9: 0.39,
    2.0: 0.36,
    2.1: 0.33,
    2.2: 0.31,
    2.3: 0.30,
    2.4: 0.28,
    2.5: 0.25,
    2.6: 0.25,
    2.7: 0.23,
    2.8: 0.21,
    2.9: 0.21,
    3.0: 0.20,
    3.1: 0.18,
    3.2: 0.18,
    3.3: 0.17,
    3.4: 0.15,
    3.5: 0.16,
    3.6: 0.14,
    3.7: 0.12,
    3.8: 0.14,
    3.9: 0.15,
    4.0: 0.17,
}


def get_transformation() -> UniformSelectionTransform:
    global transformation
    if transformation is None:
        transformation = UniformSelectionTransform(dict(max_blob_size=2000, method="max"))
    return transformation


def preprocess(blob: np.ndarray) -> np.ndarray:
    """
    
    """
    transformation = get_transformation()
    blob = transformation.preprocess(blob)
    return blob


def scale_cryoem_blob(blob: np.ndarray, resolution: float) -> np.ndarray:
    """

    """
    return blob * (MAP_VALUE_MAPPER[round(resolution, 1)] / blob[blob > 0].min())
