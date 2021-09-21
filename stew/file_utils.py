from collections import OrderedDict
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Sequence

import numpy as np

from stew.base import ContainerLike, IndividualSequence

__all__ = ['OperatorOutput', 'BaseOperatorOutput', 'SelectionOutput', 'QueryOutput', 'ExplicitEnum', 'SplitStrategy']


@dataclass()
class OperatorOutput(OrderedDict):
    """
    Inspired from HuggingFace's ModelOutput class.
    Base class for all operator outputs as dataclass. Has a ``__getitem__`` that allows indexing by integer or slice
    (like a tuple) or strings (like a dictionary) that will ignore the ``None`` attributes.
    Otherwise behaves like a regular python dictionary.

    .. warning::
        You can't unpack a :obj:`OperatorOutput` directly. Use the :meth:`~stew.file_utils.OperatorOutput.to_tuple`
        method to convert it to a tuple before.
    """

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self) -> tuple[Any, ...]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@dataclass()
class BaseOperatorOutput(OperatorOutput):
    output_individuals: Sequence = None
    target_container: IndividualSequence = None


@dataclass()
class SelectionOutput(OperatorOutput):
    target_container: ContainerLike = None
    selected_individuals: Sequence = None
    discard_individuals: Sequence = None


@dataclass()
class QueryOutput(OperatorOutput):
    output_individuals: Sequence = None
    target_container: IndividualSequence = None
    distance_matrix: np.ndarray = None
    container_index: np.ndarray = None
    valid_mask: np.ndarray = None


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class SplitStrategy(ExplicitEnum):
    """
    # TODO Fill this doc when the specific argument is clarified.
    Possible values for the argument in VariationOperator.
    Useful for tab-completion in an IDE.
    """
    LEFT = "left"
    MID = "mid"
    RIGHT = "right"
