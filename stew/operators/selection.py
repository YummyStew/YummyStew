import logging
from typing import (Sequence, Union)

import numpy as np

from .base import QueryOperator, SelectionOperator
from .query import NearestNeighbor
from stew.base import FitnessLike
from stew.file_utils import SelectionOutput

__all__ = ['NoveltySearchLocalCompetition', 'TruncationSelection']

logger = logging.getLogger(__name__)


def dominates(self_obj_values: FitnessLike,
              other_obj_values: FitnessLike,
              dominate_margin: Union[int, float, np.ndarray] = 0.,
              worse_margin: Union[int, float, np.ndarray] = 0.) -> np.ndarray:
    self_arr = np.atleast_2d(self_obj_values)
    other_arr = np.atleast_2d(other_obj_values)
    being_dominated = np.any(self_arr + worse_margin < other_arr, axis=-1)
    is_dominate = np.any(self_arr > other_arr + dominate_margin, axis=-1)
    return is_dominate and not being_dominated


def strictly_dominates(self_obj_values: FitnessLike, other_obj_values: FitnessLike) -> np.ndarray:
    """Return true if each objective of ``self_obj_values`` is not strictly worse than
    the corresponding objective of ``other_obj_values`` and at least one objective is
    strictly better.
    """
    return dominates(self_obj_values, other_obj_values, 0., 0.)


class NoveltySearchLocalCompetition(SelectionOperator):
    def __init__(self,
                 novelty_threshold: float,
                 query_operator: QueryOperator = NearestNeighbor(nn_size=5, copy=False)):
        self.novelty_threshold = novelty_threshold
        if query_operator is None:
            query_operator = NearestNeighbor(nn_size=5, copy=False)
        self.query_operator = query_operator

    def extra_repr(self):
        repr_str = f"novelty_threshold={self.novelty_threshold}, " \
                   f"query_operator={self.query_operator}"
        return repr_str

    def forward(self, container, candidates):
        novelty_threshold = self.novelty_threshold
        query_operator = self.query_operator

        # Calculate pairwise distance
        candidate_indexes = np.arange(len(candidates))
        pending_inds = list(candidates) + list(container.depot)  # Candidates first!!!
        discard_individuals = None
        if len(pending_inds) == 1:
            # There should be only one candidate, and the container is empty.
            assert len(candidates) == 1, f"unknown error happened within {self.__class__.__name__}, " \
                                         f"please check the implementation to resolve this issue!"
            selected_candidates = candidates
        else:
            query_output = query_operator(pending_inds, candidates)
            novelty = np.nanmean(query_output.distance_matrix, axis=-1)

            # All candidates passed novelty_threshold are consider as selected initially.
            passed_mask = (novelty > novelty_threshold)
            if not np.all(passed_mask):
                # If at least one candidate failed novelty check, run local competition.
                # Compare each failed candidate against its nearest competitor.
                # Note: the nearest competitor can either be another candidate, or individual from container.
                # If the competitor is a candidate, it will be always discarded.

                failed_mask = ~passed_mask
                challenger_idx = candidate_indexes[failed_mask]
                competitor_idx = query_output.container_index[failed_mask, 0]

                is_improved = strictly_dominates(
                    np.stack([candidates[idx].objective_value for idx in challenger_idx]),
                    np.stack([pending_inds[idx].objective_value for idx in competitor_idx])
                )

                improved_idx = challenger_idx[is_improved]
                failed_idx = np.unique(competitor_idx[is_improved])
                failed_novel_indexes = np.intersect1d(failed_idx, candidate_indexes, assume_unique=True)
                passed_mask[improved_idx] = True
                passed_mask[failed_novel_indexes] = False

                # Retrieve discard individuals that is active in container
                discard_idx = np.setdiff1d(failed_idx, candidate_indexes, assume_unique=True)
                discard_inds = []
                for idx in discard_idx:
                    ind = pending_inds[idx]
                    if ind in container.active_items:
                        discard_inds.append(ind)
                if discard_inds:
                    discard_individuals = discard_inds

            selected_candidates = [candidates[idx] for idx in candidate_indexes[passed_mask]]

        return SelectionOutput(
            target_container=container,
            selected_individuals=selected_candidates,
            discard_individuals=discard_individuals
        )


class TruncationSelection(SelectionOperator):
    def __init__(self,
                 elite_size: Union[float, int] = 'auto',
                 order: Union[int, Sequence[int]] = None,
                 verbose=True):
        # order is an integer or a list of integer, specify the comparison order of the fitness dimension
        if elite_size == 'auto':
            elite_size = None

        if isinstance(elite_size, float):
            if 0.0 < elite_size < 1.0:
                raise ValueError(f"If elite_size is a float it should between 0.0 and 1.0, "
                                 f"but got {elite_size} instead.")
        elif isinstance(elite_size, int):
            if elite_size <= 0:
                raise ValueError(f"If elite_size is int it should be greater than 0, "
                                 f"but got {elite_size} instead.")
        elif elite_size is not None:
            raise ValueError(f"Unknown elite_size type `{elite_size}`!")

        self.elite_size = elite_size
        self.order = order
        self.verbose = verbose

    def extra_repr(self):
        repr_str = f"elite_size={self.elite_size}"
        if self.order is not None:
            repr_str += f", order={self.order}"
        if self.verbose:
            repr_str += ", verbose=True"
        return repr_str

    def forward(self, container, candidates):
        elite_size = self.elite_size
        order = self.order

        if elite_size is None:
            # Auto determine by container capacity
            elite_size = np.floor(container.capacity)
        elif isinstance(elite_size, float):
            elite_size = np.round(container.capacity * elite_size)

        if np.isinf(elite_size) and self.verbose:
            logger.warning("Encounter infinite elite_size in truncation selection, "
                           "all individuals will be directly inserted without truncation.")

        if elite_size >= container.size + len(candidates):
            # Add all individuals
            selected_candidates = candidates
            discard_individuals = None
        else:
            pending_candidates = list(candidates) + list(container.active_items)
            # The container individuals should be extended after input candidates.
            # So we can distinguish the source of individuals by index.
            candidate_idx = np.arange(len(candidates))
            container_idx = np.arange(len(candidates), len(pending_candidates))

            # Sort by objective values
            obj_values = np.stack([ind.objective_value for ind in pending_candidates])
            obj_values = np.atleast_2d(obj_values)

            if order is not None:
                # Permute the last dimension (fitness dimension)
                prioritized_dim = np.atleast_1d(order)
                old_order = np.arange(obj_values.shape[-1])
                new_order = np.concatenate([prioritized_dim, np.setdiff1d(old_order, prioritized_dim)])
                obj_values[..., old_order] = obj_values[..., new_order]

            # Numpy partition is ascending order (smallest first)
            selected_idx = np.argpartition(obj_values, -elite_size, axis=0)
            selected_idx = selected_idx[-elite_size:]  # Truncation
            if selected_idx.size > elite_size:
                # Handle Multi-objective fitness case
                # Note: We select the individual by the fitness ranking in each dimension
                # TODO Add Note: order of dimension will affect the select result.
                unique_idx = np.unique(selected_idx)
                sorted_idx = np.argsort(obj_values[unique_idx], axis=0)
                sorted_idx = np.flip(sorted_idx[-elite_size:], axis=0)  # Reverse order to best first
                second_selected = set()
                for idx in sorted_idx:
                    second_selected.add(idx)
                    if len(second_selected) >= elite_size:
                        break
                selected_idx = np.fromiter(second_selected, dtype=sorted_idx.dtype, count=elite_size)

            selected_idx = selected_idx.flatten()

            # noinspection PyTypeChecker
            selected_candidates = [candidates[idx]
                                   for idx in np.intersect1d(candidate_idx, selected_idx, assume_unique=True)]
            discard_individuals = [pending_candidates[idx]
                                   for idx in np.setdiff1d(container_idx, selected_idx, assume_unique=True)]

        return SelectionOutput(
            target_container=container,
            selected_individuals=selected_candidates,
            discard_individuals=discard_individuals
        )
