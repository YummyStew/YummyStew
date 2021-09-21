from abc import ABC
from typing import Callable, Iterable, Optional, Sequence, Set, Union

import numpy as np

from .base import VariationOperator
from .query import RandomQuery
from stew.base import ContainerLike, IndividualLike, Operator
from stew.file_utils import BaseOperatorOutput, QueryOutput, SplitStrategy

__all__ = ['WordInplaceSubstitution', 'InsertionMutation', 'DeletionMutation', 'CrossOver']


class WordInplaceSubstitution(VariationOperator):
    def __init__(self,
                 char_set: Union[Set, Sequence],
                 mutate_prob: Union[float, np.ndarray] = 0.2,
                 choice_prob: Optional[np.ndarray] = None,
                 inplace: bool = False):
        self.char_set = char_set
        self.mutate_prob = mutate_prob
        self.choice_prob = choice_prob
        self.inplace = inplace

    @property
    def char_set(self):
        """
        The available character set for substitution.
        """
        return self._char_set

    @property
    def mutate_prob(self) -> float:
        """
        The mutate probability for each char.
        """
        return self._mutate_prob

    @property
    def choice_prob(self) -> Optional[np.ndarray]:
        """
        An optional numpy 1d-array that determine the weight of character.
        """
        return self._choice_prob

    @property
    def step_callback(self) -> Callable:
        return lambda ind, cont: \
            WordInplaceSubstitution.word_substitution_(ind, self.char_set, self.mutate_prob, self.choice_prob)

    @char_set.setter
    def char_set(self, value):
        self._char_set = list(set(value))

    @mutate_prob.setter
    def mutate_prob(self, value):
        if not 0 < value <= 1.:
            raise ValueError(f"`mutate_prob` should in range of 0. < mut_prob <= 1, "
                             f"but got {value} instead")
        self._mutate_prob = value

    @choice_prob.setter
    def choice_prob(self, value):
        # Simple test for value validity
        np.random.choice(self.char_set, p=value)
        self._choice_prob = value

    @staticmethod
    def word_substitution_(individual,
                           char_set: Union[Set, Sequence],
                           mutate_prob: Union[float, np.ndarray] = 0.2,
                           choice_prob: Optional[np.ndarray] = None):
        # Ensure at least one mutation happens
        always_mutate = np.random.randint(len(individual))
        for idx, ch in enumerate(individual):
            if idx == always_mutate or np.random.rand() < mutate_prob:
                ch_idx = char_set.index(ch)
                subset = [char_set[i] for i in range(len(char_set)) if i != ch_idx]
                choice_prob_ = choice_prob
                if choice_prob_:
                    choice_prob_ = list(choice_prob_)
                    del choice_prob_[ch_idx]
                    # Normalize to sum 1 choice_prob
                    choice_prob_ = np.asarray(choice_prob_) / np.sum(choice_prob_)
                individual[idx] = np.random.choice(subset, p=choice_prob_)
        return individual

    def extra_repr(self) -> str:
        repr_str = f"mutate_prob={self.mutate_prob}"
        if self.choice_prob:
            repr_str += f", choice_prob={self.choice_prob}"
        if self.inplace:
            repr_str += ", inplace=True"
        return repr_str


def _random_split(start, end, low, high, split_strategy: SplitStrategy) -> range:
    """
    Return a random index range with index between [start, end) and len(range) between [low, high)
    NOTE:
        Returned range might be empty, please check len(range) before use its index.
        If the rolled fragment length is greater than index range,
        the resulted range will be clipped to the index range.
    """
    if start >= end:
        # Empty range
        return range(start, start)
    if low >= high - 1:
        # Empty range with start = stop = split_idx
        idx = np.random.randint(start, end)
        return range(idx, idx)

    frag_len = np.random.randint(low, high)
    if split_strategy == SplitStrategy.LEFT:
        result = range(start, min(end, start + frag_len))
    elif split_strategy == SplitStrategy.MID:
        cent_start = np.random.randint(start, max(start + 1, end - frag_len))
        result = range(cent_start, min(end, cent_start + frag_len))
    elif split_strategy == SplitStrategy.RIGHT:
        result = range(max(start, end - frag_len - 1), end)
    else:
        raise ValueError("Unknown split strategy!")
    return result


def _crossover_param_check(spawn_or_query_operator, result_min_length, result_max_length):
    if spawn_or_query_operator is None:
        raise ValueError("spawn_or_query_operator must be provided!")
    if result_max_length is None or result_max_length <= 0:
        raise ValueError(f"``result_max_length`` must be a positive integer, "
                         f"but got ``{result_max_length}`` instead")
    if result_min_length >= result_max_length:
        raise ValueError(f"``result_max_length`` should be equal to or greater than ``result_min_length``")


class BaseCrossOver(VariationOperator, ABC):
    def __init__(self,
                 result_min_length: int = 0,
                 result_max_length: int = np.inf,
                 spawn_or_query_operator: Operator = RandomQuery(),
                 split_strategies: Sequence[Union[str, SplitStrategy]] = SplitStrategy,
                 inplace: bool = False):
        self.spawn_or_query_operator = spawn_or_query_operator
        self.result_min_length = result_min_length
        self.result_max_length = result_max_length
        self.split_strategies = split_strategies
        self.inplace = inplace

    @property
    def result_min_length(self) -> Union[int, float]:
        return self._result_min_length

    @property
    def result_max_length(self) -> Union[int, float]:
        return self._result_max_length

    @property
    def split_strategies(self) -> Sequence[SplitStrategy]:
        return self._split_strategies

    @result_min_length.setter
    def result_min_length(self, value):
        if value < 0:
            raise ValueError(f"``result_min_length`` must be a non-negative integer, "
                             f"but got ``{value}`` instead")
        self._result_min_length = value

    @result_max_length.setter
    def result_max_length(self, value):
        if value <= 0:
            raise ValueError(f"``result_max_length`` must be a positive integer, "
                             f"but got ``{value}`` instead")
        self._result_max_length = value

    @split_strategies.setter
    def split_strategies(self, value):
        if isinstance(value, str) or not isinstance(value, Iterable):
            value = [value]
        self._split_strategies = tuple(SplitStrategy(v) for v in value)

    # noinspection DuplicatedCode
    def extra_repr(self):
        repr_str = f"spawn_or_query_operator={self.spawn_or_query_operator}, "
        if self.result_min_length > 0:
            repr_str += f"result_min_length={self.result_min_length}"
        if np.isfinite(self.result_max_length):
            repr_str += f", result_max_length={self._result_max_length},"
        if set(self.split_strategies) != set(SplitStrategy):
            repr_str += f", split_strategies={list(enum.value for enum in self.split_strategies)}"
        if self.inplace:
            repr_str += ", inplace=True"

        return repr_str


class CrossOver(BaseCrossOver):
    @property
    def step_callback(self) -> Callable:
        return lambda ind, cont: CrossOver.crossover_(ind, cont, self.result_min_length, self.result_max_length,
                                                      self.spawn_or_query_operator, self.split_strategies)

    def valid_check(self, individual) -> bool:
        return self.result_min_length <= individual <= self.result_max_length

    @staticmethod
    def crossover_(individual,
                   container: Optional[ContainerLike] = None,
                   result_min_length: int = 0,
                   result_max_length: int = np.inf,
                   spawn_or_query_operator: Operator = RandomQuery(),
                   split_strategies: Sequence[SplitStrategy] = SplitStrategy):
        _crossover_param_check(spawn_or_query_operator, result_min_length, result_max_length)

        output = spawn_or_query_operator(container, [individual])
        if isinstance(output, (BaseOperatorOutput, QueryOutput)):
            cross_target = output.output_individuals[0]
        elif isinstance(output, IndividualLike):
            cross_target = output
        else:
            raise RuntimeError("Unknown output type!")

        source_len = len(individual)
        target_len = len(cross_target)
        # Remove fragment from source sequence
        if source_len == 1:
            # Discard the only one element
            min_given, max_given = 1, 1
        else:
            # Ensure at least one element of result is from source individual
            max_given = min(source_len + target_len - result_min_length, source_len - 1)
            min_given = max(source_len - result_max_length, 0)
        delete_range = _random_split(0, source_len, min_given, max_given + 1, np.random.choice(split_strategies))

        max_recv = min(result_max_length + len(delete_range) - source_len, target_len)
        min_recv = max(result_min_length + len(delete_range) - source_len, 1)
        insert_range = _random_split(0, target_len, min_recv, max_recv + 1, np.random.choice(split_strategies))

        if len(insert_range) != 0:
            target_slice = slice(insert_range.start, insert_range.stop)
            combined = individual[:delete_range.start] + cross_target[target_slice] + individual[delete_range.stop:]
            individual.clear()
            individual.extend(combined)

        return individual


class InsertionMutation(BaseCrossOver):
    def __init__(self,
                 insert_min_length: int = 1,
                 result_max_length: int = np.inf,
                 spawn_or_query_operator: Operator = RandomQuery(),
                 split_strategies: Sequence[Union[str, SplitStrategy]] = SplitStrategy,
                 inplace: bool = False):
        super(InsertionMutation, self).__init__(0, result_max_length, spawn_or_query_operator, split_strategies, inplace)  # noqa
        self.insert_min_length = insert_min_length

    @property
    def insert_min_length(self):
        return self._insert_min_length

    @property
    def step_callback(self) -> Callable:
        return lambda ind, cont: InsertionMutation.insertion_mutation_(
            ind, cont,
            self.insert_min_length,
            self.result_max_length,
            self.spawn_or_query_operator,
            self.split_strategies
        )

    @insert_min_length.setter
    def insert_min_length(self, value):
        if value < 1:
            raise ValueError(f"``insert_min_length`` should be at least 1, but got ``{value}`` instead")
        if value >= self.result_max_length:
            raise ValueError(f"``insert_min_length`` should be lesser than ``result_max_length``!")
        self._insert_min_length = value

    def valid_check(self, individual) -> bool:
        return len(individual) < self.result_max_length

    @staticmethod
    def insertion_mutation_(individual,
                            container=None,
                            insert_min_length: int = 1,
                            result_max_length: int = np.inf,
                            spawn_or_query_operator: Operator = RandomQuery(),
                            split_strategies: Sequence[SplitStrategy] = SplitStrategy):
        _crossover_param_check(spawn_or_query_operator, 0, result_max_length)
        if insert_min_length < 1:
            raise ValueError(f"``insert_min_length`` should be at least 1, "
                             f"but got ``{insert_min_length}`` instead")

        output = spawn_or_query_operator(container, [individual])
        if isinstance(output, (BaseOperatorOutput, QueryOutput)):
            insert_source = output.output_individuals[0]
        elif isinstance(output, IndividualLike):
            insert_source = output
        else:
            raise RuntimeError("Unknown output type!")

        split_idx = _random_split(0, len(individual), 0, 1, SplitStrategy.MID).start
        max_recv = min(result_max_length - len(individual), len(insert_source))
        min_recv = min(insert_min_length, max_recv)
        insert_range = _random_split(0, len(insert_source), min_recv, max_recv + 1, np.random.choice(split_strategies))

        if len(insert_range) > 0:
            fragment_slice = slice(insert_range.start, insert_range.stop)
            combined = individual[:split_idx] + insert_source[fragment_slice] + individual[split_idx:]
            individual.clear()
            individual.extend(combined)

        return individual

    # noinspection DuplicatedCode
    def extra_repr(self):
        repr_str = f"spawn_or_query_operator={self.spawn_or_query_operator}, " \
                   f"insert_min_length={self.insert_min_length}"

        if np.isfinite(self.result_max_length):
            repr_str += f", result_max_length={self._result_max_length},"
        if set(self.split_strategies) != set(SplitStrategy):
            repr_str += f", split_strategies={list(enum.value for enum in self.split_strategies)}"
        if self.inplace:
            repr_str += ", inplace=True"

        return repr_str


class DeletionMutation(BaseCrossOver):
    def __init__(self,
                 low: int,
                 high: int = None,
                 result_min_length: int = 0,
                 split_strategies: Sequence[Union[str, SplitStrategy]] = SplitStrategy,
                 inplace: bool = False):
        super(DeletionMutation, self).__init__(result_min_length=result_min_length,
                                               spawn_or_query_operator=RandomQuery(),
                                               split_strategies=split_strategies,
                                               inplace=inplace)
        self.low = low
        self.high = high

        self._bound_check(self.low, self.high)
        del self.spawn_or_query_operator

    @property
    def range(self):
        self._bound_check(self.low, self.high)

        if self.high is None:
            return 1, self.low + 1

        return self.low, self.high

    @property
    def step_callback(self) -> Callable:
        low, high = self.range
        return lambda ind, cont: DeletionMutation.deletion_mutation_(
            ind, low, high,
            self.result_min_length, self.split_strategies
        )

    def valid_check(self, individual) -> bool:
        return len(individual) > self.result_min_length

    @staticmethod
    def _bound_check(low, high=None):
        if low <= 0:
            raise ValueError("param `low` should be greater than 0.")
        if high and low >= high:
            raise ValueError("param `high` should be greater than `low`.")

    @staticmethod
    def deletion_mutation_(individual,
                           low: int,
                           high: int = None,
                           result_min_length: int = 0,
                           split_strategies: Sequence[SplitStrategy] = SplitStrategy):
        DeletionMutation._bound_check(low, high)
        if high is None:
            low, high = 1, low

        high = min(high, len(individual) - result_min_length + 1)
        delete_range = _random_split(0, len(individual), low, high, np.random.choice(split_strategies))

        if len(delete_range) > 0:
            # Note: type(removed_result) is list, NOT Individual!
            removed_result = individual[:delete_range.start] + individual[delete_range.stop:]
            individual.clear()
            individual.extend(removed_result)

        return individual

    def extra_repr(self):
        repr_str = f"low={self.low}"
        if self.high:
            repr_str += f", high={self.high}"
        if self.result_min_length > 0:
            repr_str += f"result_min_length={self.result_min_length}"
        if set(self.split_strategies) != set(SplitStrategy):
            repr_str += f", split_strategies={list(enum.value for enum in self.split_strategies)}"
        if self.inplace:
            repr_str += ", inplace=True"

        return repr_str
