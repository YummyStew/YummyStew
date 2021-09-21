import copy
import itertools
from abc import ABC, abstractmethod
from typing import Callable, Optional, overload, Sequence, Union

from stew.base import ContainerLike, IndividualLike, IndividualSequence, Operator
from stew.file_utils import BaseOperatorOutput, QueryOutput, SelectionOutput

__all__ = ['SpawnOperator', 'VariationOperator', 'QueryOperator', 'SelectionOperator']


class SpawnOperator(Operator, ABC):
    @overload
    def forward(self,
                container: IndividualSequence,
                seed: IndividualLike) -> Union[IndividualLike, BaseOperatorOutput]: ...

    @overload
    def forward(self,
                container: IndividualSequence,
                seed: IndividualSequence) -> Union[IndividualLike, BaseOperatorOutput]: ...

    @abstractmethod
    def forward(self, container, seed) -> Union[IndividualLike, BaseOperatorOutput]: ...


class VariationOperator(Operator, ABC):
    inplace: bool

    @property
    @abstractmethod
    def step_callback(self) -> Callable:
        """
        A callable object or function that required to be implement in concrete operator.
        Require signatures as __call__(individual, container)
        """

    @overload
    def forward(self,
                container: Optional[ContainerLike],
                individual: IndividualLike) -> BaseOperatorOutput:
        ...

    @overload
    def forward(self,
                container: Optional[ContainerLike],
                individual_batch: Sequence[IndividualLike]) -> BaseOperatorOutput:
        ...

    def forward(self, cont_or_ind_batch, *args):
        if len(args) == 0:
            # treat the container as individual batch
            individual_batch = cont_or_ind_batch,
            container = None
        else:
            container = cont_or_ind_batch
            individual_batch = args[0]

        if not isinstance(individual_batch, ContainerLike) and isinstance(individual_batch, IndividualLike):
            individual_batch = [individual_batch]
        if not self.inplace:
            individual_batch = copy.deepcopy(individual_batch)

        # TODO Parallelism
        result = map(self.step_callback, individual_batch, itertools.repeat(container))
        output_individuals = list(result)

        return BaseOperatorOutput(
            output_individuals=output_individuals,
            target_container=container
        )


class QueryOperator(Operator, ABC):
    copy: bool

    @overload
    def forward(self, container: IndividualSequence, query: IndividualLike) -> QueryOutput: ...

    @overload
    def forward(self, container: IndividualSequence, queries: IndividualSequence) -> QueryOutput: ...

    @abstractmethod
    def forward(self, container, queries) -> QueryOutput: ...


class SelectionOperator(Operator, ABC):
    @abstractmethod
    def forward(self, container: ContainerLike, candidates: IndividualSequence) -> SelectionOutput: ...
