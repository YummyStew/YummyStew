import numpy as np
from faiss import pairwise_distances as faiss_pairwise_distance

from stew.base import IndividualSequence

__all__ = ['get_distance_metric',
           'squared_euclidean', 'euclidean_distance', 'scaled_squared_euclidean', 'scaled_euclidean_distance']


def get_distance_metric(value):
    if isinstance(value, str):
        if value == 'euclidean':
            value = euclidean_distance
        else:
            raise ValueError(f"Unknown `dist_metric` type: {value}")
    return value


def squared_euclidean(query: IndividualSequence, container: IndividualSequence, rescale: bool = False) -> np.ndarray:
    # Note: faiss support float32 only, and return squared euclidean distance
    # Append Batch Dimension
    qf = np.stack([ind.features for ind in query])
    cf = np.stack([ind.features for ind in container])

    if rescale:
        return faiss_pairwise_distance(qf, cf) / np.sqrt(qf.shape[-1])

    return faiss_pairwise_distance(qf, cf)


def euclidean_distance(query: IndividualSequence, container: IndividualSequence) -> np.ndarray:
    squared_distance = squared_euclidean(query, container)
    squared_distance = np.clip(squared_distance, a_min=0., a_max=np.inf)  # Mitigate numerical instability.
    return np.sqrt(squared_distance)


def scaled_squared_euclidean(query: IndividualSequence, container: IndividualSequence) -> np.ndarray:
    # Note: faiss support float32 only, and return squared euclidean distance
    # Append Batch Dimension
    return squared_euclidean(query, container, rescale=True)


def scaled_euclidean_distance(query: IndividualSequence, container: IndividualSequence) -> np.ndarray:
    squared_distance = scaled_squared_euclidean(query, container)
    squared_distance = np.clip(squared_distance, a_min=0., a_max=np.inf)  # Mitigate numerical instability.
    return np.sqrt(squared_distance)

