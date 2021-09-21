from typing import Optional, Sequence, Set, Union

import numpy as np

from .base import SpawnOperator
from stew.phenotypes import Individual

__all__ = ['SpawnRandomSequence']


class SpawnRandomSequence(SpawnOperator):
    def __init__(self,
                 char_set: Union[Set, Sequence],
                 low: int,
                 high: int = None,
                 replace: bool = True,
                 choice_prob: np.ndarray = None):
        self.char_set = char_set
        self.low = low
        self.high = high
        self.replace = replace
        self.choice_prob = choice_prob

        self._bound_check(self.low, self.high)

    @property
    def char_set(self):
        """
        The available character set for substitution.
        """
        return self._char_set

    @property
    def range(self):
        self._bound_check(self.low, self.high)

        if self.high is None:
            return 1, self.low + 1

        return self.low, self.high

    @property
    def choice_prob(self) -> Optional[np.ndarray]:
        """
        An optional numpy 1d-array that determine the weight of character.
        """
        return self._choice_prob

    @char_set.setter
    def char_set(self, value):
        self._char_set = list(set(value))

    @choice_prob.setter
    def choice_prob(self, value):
        # Simple test for value validity
        np.random.choice(self.char_set, p=value)
        self._choice_prob = value

    def forward(self, container=None, seed=None):
        low, high = self.range
        length = np.random.randint(low, high)
        sequence = np.random.choice(self.char_set, length, self.replace, self.choice_prob)
        return Individual(sequence)

    def extra_repr(self):
        repr_str = f"char_set={self.char_set}"
        low, high = self.range
        repr_str += f", low={low}, high={high}"
        if self.replace:
            repr_str += ", replace=True"
        if self.choice_prob:
            repr_str += f", prob={self.choice_prob}"
        return repr_str

    @staticmethod
    def _bound_check(low, high=None):
        if low <= 0:
            raise ValueError("param `low` should be greater than 0.")
        if high and low >= high:
            raise ValueError("param `high` should be greater than `low`.")

# TODO BufferSpawnOperator
# class BufferGenerator(Operator):
#     def __init__(self, batch_generator: Callable):
#         self.batch_generator = batch_generator
#         self.buffer = []
#
#     def __call__(self, seed, container):
#         if len(self.buffer) == 0:
#             self.generate_ind_batch(seed, container)
#             if len(self.buffer) == 0:
#                 raise RuntimeError("Failed to generate new individuals!")
#         return self.buffer.pop()
#
#     def generate_ind_batch(self, seed, container):
#         new_batch = self.batch_generator(seed, container)
#         if isinstance(new_batch, Individual) and not isinstance(new_batch, ContainerLike):
#             # The new_batch is an individual, append it to the buffer.
#             self.buffer.append(new_batch)
#         else:
#             self.buffer.extend(new_batch)
#
#     def clear(self):
#         self.buffer.clear()
