import numpy as np

from pipeline.transforms import UniformSelectionTransform

transformation = None


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
