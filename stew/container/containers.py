#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.

__all__ = ["Container", "Population"]

from typing import (Any, Iterable, Iterator, Optional, Sequence, Tuple, Union)

import numpy as np

from stew.base import (ContainerLike, Copyable, CreatableFromConfig, IndividualLike,
                       registry, Summarisable, Module, FitnessLike, FeaturesLike)
from stew.phenotypes import IndividualProps


def tuplify(item: Union[Any, Sequence[Any]]) -> Tuple[Any, ...]:
    return tuple(item) if isinstance(item, Sequence) else tuple([item])


# TODO verify that containers are thread-safe
# TODO Use customized Exception
# TODO Link the construct callback by collection type, the wrapped collection class should include:
#   - __len__
#   - __getitem__
#   - __contains__
#   - __iter__
#   - add_to_collection
#   - discard_from_collection
#   - clear (to remove all individuals)
@registry.register
class Container(Module, Sequence, ContainerLike, Summarisable, Copyable, CreatableFromConfig):
    """ A simple container that can storage individuals and enforce the bound check

    Parameters  # TODO
    ----------
    iterable: Iterable[IndividualLike] or None

    storage_type: Backend (MutableSet or MutableSequence)

    depot_type: bool or Backend (MutableSet or MutableSequence)

    """

    def __init__(self,
                 iterable: Optional[Iterable] = None,
                 capacity: Optional[float] = None,
                 keep_discarded: bool = False,
                 name: Optional[str] = None):
        self._reset_properties()

        # self.recentness = []
        self.name = name if name is not None else f"{self.__class__.__name__}-{id(self)}"
        self.capacity = np.inf if capacity is None else capacity
        self._keep_discarded = keep_discarded
        if self._keep_discarded:
            self._active_items = list()
            self._depot = list()
        else:
            self._depot = self._active_items = list()

        if iterable is not None:
            self.add(iterable)

    @property
    def keep_discarded(self):
        return self._keep_discarded

    @property
    def active_items(self):
        return self._active_items

    @property
    def depot(self):
        return self._depot

    @property
    def num_added(self):
        return self._num_added

    @property
    def num_discarded(self):
        return self._num_discarded

    @property
    def num_rejected(self):
        return self._num_rejected

    @property
    def num_operations(self):
        return self.num_added + self.num_discarded

    @property
    def size(self):
        # For one-dimension container, self.size is equivalent to len(self), the number of items in container.
        return self.__len__()

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        if value <= 0:
            raise ValueError("Container capacity should be greater than 0!")
        self._capacity = value

    @property
    def free(self):
        return self.capacity - self.size

    def reset(self):
        self.clear(clear_depot=True)
        self._reset_properties()

    def __len__(self) -> int:
        return len(self.active_items)

    def __getitem__(self, i: Union[int, slice]):
        return self.active_items[i]

    def __contains__(self, x: object) -> bool:
        return x in self.active_items

    def __iter__(self) -> Iterator[IndividualLike]:
        return iter(self.active_items)

    def extra_repr(self):
        repr_str = f"active_items={self.__len__()}"
        if np.isfinite(self.capacity):
            repr_str += f", capacity={self.capacity}"
        if self.keep_discarded:
            repr_str += f", depot_size={len(self.depot)}"

        return repr_str

    def add(self, individuals, **kwargs) -> Optional[Sequence[int]]:
        """Add ``individuals`` to the container, and returns its index if successful, None elsewhere. """
        if isinstance(individuals, IndividualLike):
            individuals = [individuals]

        return self._add_internal(individuals, False)

    def _add_internal(self, individuals: Sequence[IndividualLike], only_to_depot: bool) -> Optional[Sequence[int]]:
        self._capacity_check(individuals, only_to_depot=only_to_depot)

        index = None
        if not only_to_depot:
            # Add to storage first
            index = self._add_to_collection(self.active_items, individuals)
            self._num_added += len(index)
            self._num_rejected += len(individuals) - len(index)

        # Also add to depot if we want to keep all individuals.
        if self.keep_discarded:
            self._add_to_collection(self.depot, individuals)

        return index

    def _capacity_check(self, individuals: Sequence[IndividualLike], only_to_depot: bool):
        # Verify if we do not exceed capacity
        if self.free < len(individuals) and not only_to_depot:
            self._num_rejected += len(individuals)
            raise IndexError(f"No free slot available in this container.")
        if only_to_depot and not self.keep_discarded:
            self._num_rejected += len(individuals)
            raise ValueError(f"`only_to_depot` can only be set to True if self.keep_discarded=True.")

    @staticmethod
    def _add_to_collection(collection, individuals: Sequence[IndividualLike]) -> Sequence[int]:
        collection: list
        start_idx = len(collection)
        collection.extend(individuals)
        indexes = tuple(range(start_idx, len(collection)))
        if len(indexes) != len(individuals):
            raise RuntimeError(f"Failed to insert all individuals to the container list.")
        return indexes

    def discard(self, individuals, also_from_depot=False, **kwargs):
        """Remove ``individuals`` from the container.
         If ``also_from_depot`` is True, discard them also from the depot, if exists."""
        if isinstance(individuals, IndividualLike):
            individuals = [individuals]
        return self._discard_internal(individuals, also_from_depot=also_from_depot)

    def _discard_internal(self, individuals: Sequence[IndividualLike], also_from_depot: bool) -> Sequence[IndividualLike]:
        discarded = self._discard_from_collection(self.active_items, individuals)
        self._num_discarded += len(discarded)
        if also_from_depot and self.keep_discarded:
            discarded = self._discard_from_collection(self.depot, individuals)
        return discarded

    @staticmethod
    def _discard_from_collection(collection, individuals: Sequence[IndividualLike]) -> Sequence[IndividualLike]:
        collection: list
        discarded = []
        for ind in individuals:
            try:
                idx = collection.index(ind)
                discarded.append(collection.pop(idx))
            except ValueError:
                continue
        return tuple(discarded)

    def clear(self, clear_depot: bool = True) -> None:
        """Clear all active individuals in the container.
        if clear_depot=True, the individuals in the depot will also be removed."""
        # TODO Link clear method with collection type
        self._clear_collection(self.active_items)
        if clear_depot:
            self._clear_collection(self.depot)

    @staticmethod
    def _clear_collection(collection):
        collection: list
        collection.clear()

    def _reset_properties(self) -> None:
        self._num_added = 0
        self._num_discarded = 0
        self._num_rejected = 0


@registry.register
class Population(Container, IndividualProps):
    def __init__(self,
                 iterable: Optional[Iterable] = None,
                 capacity: Optional[float] = None,
                 keep_discarded: bool = False,
                 name: Optional[str] = None,
                 features: Optional[FeaturesLike] = np.array([]),
                 fitness: Optional[FitnessLike] = np.nan,
                 weight: Optional[FitnessLike] = 1.,
                 elapsed: Optional[float] = np.nan):
        super(Population, self).__init__(iterable, capacity, keep_discarded, name)
        self.weight = weight
        self.features = features
        self.fitness = fitness
        self.elapsed = elapsed

    def reset(self):
        super(Population, self).reset()

    def extra_repr(self):
        repr_str = super(Population, self).extra_repr()
        if len(self.features) != 0:
            repr_str += f", features={self.features}"
        repr_str += f", fitness={self.fitness}, weight={self.weight}"
        return repr_str

    def __hash__(self):
        # FIXME ??
        # Unlike Individual, we use memory address to identify the Population
        return hash(id(self))
