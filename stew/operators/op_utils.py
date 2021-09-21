from typing import Sequence
import random

import numpy as np

from .base import SpawnOperator, VariationOperator
from .query import RandomQuery
from ..base import ContainerLike, IndividualLike, Operator
from ..file_utils import BaseOperatorOutput, QueryOutput

__all__ = ['RandomOperatorPicker', 'ChainedOperator']


# TODO Replace operator storage with torch style module
class RandomOperatorPicker(Operator):

    def __init__(self, operator, *operators: Operator, choice_prob: np.ndarray = None):
        self.operators = (operator,) + tuple(operators)
        self.choice_prob = choice_prob

    def forward(self, *args, **kwargs):
        # TODO check if the operator is valid?
        selected_operator = random.choices(self.operators, self.choice_prob)[0]
        return selected_operator(*args, **kwargs)

    def extra_repr(self):
        repr_lines = [str(op) + ",\n" for op in self.operators]
        if self.choice_prob is not None:
            repr_lines.append(str(list(self.choice_prob)))
        if len(repr_lines) > 0:
            repr_lines[-1] = repr_lines[-1].rstrip(",\n")

        return "".join(repr_lines)


class ChainedOperator(Operator):
    __support_operator__ = (SpawnOperator, VariationOperator, RandomQuery, RandomOperatorPicker)

    def __init__(self, operator: Operator, *operators: Operator):
        operators = [operator] + list(operators)
        results = []
        for op in operators:
            if isinstance(op, ChainedOperator):
                results.extend(op.operators)
                continue

            if not isinstance(op, self.__support_operator__):
                raise ValueError(f"``ChainOperator`` only support these operators: "
                                 f"{list(op.__class__.__name__ for op in self.__support_operator__)}")
            results.append(op)

        self.operators = results

    def forward(self, container, individual, *args, **kwargs):
        output = None
        # TODO how to deal with other operator output?
        for operator in self.operators:
            output = operator(container, individual)
            if isinstance(output, IndividualLike):
                individual = output
            elif isinstance(output, (BaseOperatorOutput, QueryOutput)):
                individual = output.output_individuals
                container = output.target_container if output.target_container else container
            elif isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], ContainerLike):
                container, individual = output
            elif isinstance(output, Sequence) and len(output) > 0 and isinstance(output[0], IndividualLike):
                # Assume the whole sequence is individual
                individual = output
            else:
                RuntimeError("Unknown output type!")

        return output

    def extra_repr(self):
        repr_lines = [str(op) + ",\n" for op in self.operators]
        if len(repr_lines) > 0:
            repr_lines[-1] = repr_lines[-1].rstrip(",\n")

        return "".join(repr_lines)
