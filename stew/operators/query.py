from copy import deepcopy
from typing import Callable, Union
import random

import numpy as np

from .base import QueryOperator
from .metrics import get_distance_metric
from stew.base import ContainerLike, IndividualLike, IndividualSequence
from stew.file_utils import QueryOutput

__all__ = ['RandomQuery', 'NearestNeighbor']


class RandomQuery(QueryOperator):
    def __init__(self, batch_size: int = 1, recursive: bool = True, also_from_depot: bool = True, copy: bool = True):
        self.batch_size = batch_size
        self.recursive = recursive
        self.also_from_depot = also_from_depot
        self.copy = copy

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        value = int(value)
        if value <= 0:
            raise ValueError("batch_size should be greater than 0!")
        self._batch_size = value

    def forward(self, container, queries=None):
        return RandomQuery.random_query(container, self.batch_size, self.recursive, self.also_from_depot, self.copy)

    def extra_repr(self):
        repr_str = ""
        if self.batch_size > 1:
            repr_str += f"batch_size={self.batch_size}, "
        if self.recursive:
            repr_str += f"recursive={self.recursive}, "
        if self.also_from_depot:
            repr_str += f"also_from_depot={self.also_from_depot}, "
        if self.copy:
            repr_str += "copy=True"

        return repr_str.rstrip(", ")

    @staticmethod
    def random_query(container: ContainerLike,
                     batch_size: int = 1,
                     recursive: bool = True,
                     also_from_depot: bool = True,
                     copy: bool = True) -> QueryOutput:
        result = []
        for _ in range(batch_size):
            ind = RandomQuery._single_query(container, recursive, also_from_depot)
            if copy:
                ind = deepcopy(ind)
            result.append(ind)
        return QueryOutput(output_individuals=result)

    @staticmethod
    def _single_query(container, recursive, also_from_depot):
        selected_individual = RandomQuery._query_step(container, also_from_depot)
        while recursive and isinstance(selected_individual, ContainerLike):
            container = selected_individual
            selected_individual = RandomQuery._query_step(container, also_from_depot)
        return selected_individual

    @staticmethod
    def _query_step(container, also_from_depot):
        collection = container.depot if also_from_depot else container.active_items
        non_empty = []
        for elem in collection:
            if isinstance(elem, ContainerLike):
                if len(elem.depot) == 0 or (not also_from_depot and len(elem.active_items) == 0):
                    continue
            non_empty.append(elem)
        selected = random.choices(non_empty)[0]
        return selected


# TODO Test
class NearestNeighbor(QueryOperator):
    def __init__(self,
                 nn_size: int,
                 elementwise_distance: Union[str, Callable] = 'euclidean',
                 copy: bool = True):
        self.nn_size = nn_size
        self.distance_metric = elementwise_distance
        self.copy = copy

    @property
    def distance_metric(self) -> Callable:
        return self._distance_metric

    @property
    def nn_size(self) -> int:
        return self._nn_size

    @distance_metric.setter
    def distance_metric(self, dist_metric: Union[str, Callable]):
        dist_metric = get_distance_metric(dist_metric)
        self._distance_metric = dist_metric

    @nn_size.setter
    def nn_size(self, value):
        if value < 1:
            raise ValueError("``nn_size`` should be greater than 1!")
        if not isinstance(value, int):
            raise ValueError("``nn_size`` should be int!")

        self._nn_size = value

    def forward(self, container, query) -> QueryOutput:
        if not isinstance(query, ContainerLike) and isinstance(query, IndividualLike):
            query = [query]
        return NearestNeighbor.nearest_neighbor(container, query, self.nn_size, self.distance_metric, self.copy)

    @staticmethod
    def nearest_neighbor(container: IndividualSequence,
                         queries: IndividualSequence,
                         nn_size: int = 1,
                         elementwise_distance: Union[str, Callable] = 'euclidean',
                         copy: bool = True) -> QueryOutput:
        elementwise_distance = get_distance_metric(elementwise_distance)
        distance_matrix = elementwise_distance(queries, container)
        distance_matrix = np.atleast_2d(distance_matrix)

        nn_size = min(nn_size, len(container) - 1)
        np.fill_diagonal(distance_matrix, np.inf)

        # Select k-Nearest neighbor index, calculate and sort their distance.
        nn_idx = np.argpartition(distance_matrix, kth=nn_size)[..., :nn_size]
        nn_distance = np.take_along_axis(distance_matrix, nn_idx, -1)
        order_idx = np.argsort(nn_distance, axis=-1)
        nn_idx = np.take_along_axis(nn_idx, order_idx, -1)
        nn_distance = np.take_along_axis(nn_distance, order_idx, -1)

        # Fetch the valid mask, ignore distance with np.inf or np.nan
        valid_ind_mask = np.isfinite(nn_distance)
        flatten_valid_idx = nn_idx[valid_ind_mask]

        # Dirty solution for quick index  # TODO would it become bottleneck?
        if copy:
            individual_array = [deepcopy(ind) for ind in container]
        else:
            individual_array = list(container)
        individual_array.append(None)
        individual_array = np.array(individual_array)

        # Split the individual array based on the count for each individual.
        # Remove the last index, for reasons see doc of np.split
        split_idx = np.cumsum(np.sum(valid_ind_mask, axis=-1))[:-1]
        k_nearest = np.split(individual_array[flatten_valid_idx], split_idx)

        return QueryOutput(
            output_individuals=k_nearest,
            target_container=container,
            distance_matrix=nn_distance,
            container_index=nn_idx,
            valid_mask=valid_ind_mask
        )


random_query = RandomQuery.random_query
nearest_neighbor = NearestNeighbor.nearest_neighbor
