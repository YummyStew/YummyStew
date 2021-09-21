#    This file is modified from a part of qdpy.
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

"""Some base classes, stubs and types."""

import pickle
from abc import ABC
from typing import Iterable, Optional

import numpy as np

from stew.base import FeaturesLike, FitnessLike, IndividualLike, Module

__all__ = ['IndividualProps', 'Individual']


def simple_hash(obj):
    try:
        p = hash(tuple(obj))
    except TypeError:
        # Fallback to pickle binary hash
        p = hash(pickle.dumps(obj, -1))
    return p


class IndividualProps(IndividualLike, ABC):
    _features: FeaturesLike
    _fitness: FitnessLike
    _weight: FitnessLike
    elapsed: Optional[float]

    def reset(self):
        # Note: Only float type support np.nan behavior
        self.fitness = np.full_like(self.fitness, np.nan, dtype=float)
        self.features = np.full_like(self.features, np.nan, dtype=float)
        self.elapsed = np.nan

    @property
    def features(self) -> FeaturesLike:
        return self._features

    @property
    def fitness(self) -> FitnessLike:
        return self._fitness

    @property
    def weight(self) -> FitnessLike:
        return self._weight

    @features.setter
    def features(self, value):
        self._features = np.atleast_1d(value)

    @fitness.setter
    def fitness(self, value):
        value = np.atleast_1d(value)
        try:
            broadcast_shape = np.broadcast_shapes(value.shape, self.weight.shape)
            if broadcast_shape != value.shape:
                raise ValueError
        except ValueError:
            raise ValueError(f"shape mismatch: fitness and weight cannot be broadcast to a single shape.")
        self._fitness = value

    @weight.setter
    def weight(self, value):
        if not np.all(np.isfinite(value)):
            raise ValueError("`weight` should be a finite number!")
        self._weight = np.atleast_1d(value)

    @property
    def objective_value(self):
        return self.fitness * self.weight


class Individual(Module, list, IndividualProps):
    """Qdpy Individual class. Note that containers and algorithms all use internally
     either the QDPYIndividualLike Protocol or the IndividualWrapper class,
      so you can easily declare an alternative class to Individual."""

    def __init__(self,
                 iterable: Optional[Iterable] = None,
                 name: Optional[str] = None,
                 features: Optional[FeaturesLike] = np.array([]),
                 fitness: Optional[FitnessLike] = np.nan,
                 weight: Optional[FitnessLike] = 1.,
                 elapsed: Optional[float] = np.nan) -> None:
        super(Individual, self).__init__()
        self.name = name if name else ""
        self.weight = weight
        self.features = features
        # Larger fitness means better individual
        # By default we assume maximize all fitness objectives
        self.fitness = fitness
        self.elapsed = elapsed

        if iterable is not None:
            self.extend(iterable)

    def extra_repr(self):
        repr_str = f"data={list.__repr__(self)}"
        if len(self.features) != 0:
            repr_str += f", features={self.features}"
        repr_str += f", fitness={self.fitness}" \
                    f", weight={self.weight}"

        return repr_str

    def clear(self):
        super(Individual, self).clear()
        self.reset()

    def __hash__(self):
        return simple_hash(self)

    def __add__(self, other: Iterable):
        result = Individual(self)
        result.extend(other)
        result.reset()
        return result

    def __eq__(self, other):
        return self.__class__ == other.__class__ and tuple(self) == tuple(other)

    def __ne__(self, other):
        return not self.__eq__(other)
