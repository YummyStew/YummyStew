import itertools
from typing import Optional, Sequence, Union

import numpy as np

from stew.base import AlgorithmLike, ContainerLike, FitnessLike, IndividualLike, Operator
from stew.file_utils import OperatorOutput, SelectionOutput
from stew.operators import ChainedOperator, RandomOperatorPicker

__all__ = ['BaseEvolutionAlgorithm', 'Evolution']


def _assign_info(individuals, fitness, features, elapsed):
    ind_infos = [fitness, features, elapsed]
    if any(info is not None for info in ind_infos):
        for idx, ind_info in enumerate(ind_infos):
            if ind_info is None:
                ind_infos[idx] = itertools.repeat(None, len(individuals))
            else:
                ind_infos[idx] = np.atleast_1d(ind_info)

        for ind, fit, feat, eval_time in zip(individuals, *ind_infos):
            if fit is not None:
                ind.fitness = fit
            if feat is not None:
                ind.features = feat
            if eval_time is not None:
                ind.elapsed = eval_time


def _to_composite_operator(operators, composite_class):
    if not isinstance(operators, Sequence):
        return operators
    return composite_class(*operators)


class BaseEvolutionAlgorithm(AlgorithmLike):
    def __init__(self,
                 container: ContainerLike,
                 candidate_generator: Operator,
                 candidate_selector: Operator,
                 num_objective: Optional[int] = None,
                 objective_weight: Union[float, FitnessLike] = 1.0):
        self._container = container
        self._candidate_generator = candidate_generator
        self._candidate_selector = candidate_selector

        if num_objective is not None and num_objective != np.size(objective_weight):
            msg = "``num_objective`` argument will be overridden by ``objective_weight`` due to the size inconsistency!"
            raise RuntimeWarning(msg)  # noqa
        if np.ndim(objective_weight) == 0:
            if num_objective is None:
                num_objective = 1
            objective_weight = np.full(num_objective, objective_weight)
        elif np.ndim(objective_weight) > 1:
            actual_dim = np.ndim(objective_weight)
            raise RuntimeError(f"``objective_weight`` dimension should at most 1, got {actual_dim}!")

        self.objective_weight = objective_weight

    @property
    def container(self):
        return self._container

    @property
    def candidate_generator(self):
        return self._candidate_generator

    @property
    def candidate_selector(self):
        return self._candidate_selector

    @property
    def objective_weight(self):
        return self._weight

    @property
    def num_objectives(self):
        return self.objective_weight.size

    @objective_weight.setter
    def objective_weight(self, value):
        if not np.all(np.isfinite(value)):
            raise ValueError("`objective_weight` should be a finite number!")
        self._weight = np.atleast_1d(value)

    def ask(self):
        output = self.candidate_generator(self.container, None)
        if isinstance(output, IndividualLike) and not isinstance(output, ContainerLike):
            output = [output]
        elif isinstance(output, OperatorOutput):
            if hasattr(output, 'output_individuals'):
                output = output.output_individuals
            else:
                raise ValueError("Unknown OperatorOutput type from candidate_generator!")

        for ind in output:
            # Initialize the individual weight
            ind.weight = self.objective_weight

        return output

    def tell(self, individuals, fitness=None, features=None, elapsed=None):
        if isinstance(individuals, IndividualLike) and not isinstance(individuals, ContainerLike):
            individuals = [individuals]

        _assign_info(individuals, fitness, features, elapsed)
        selection_output = self.candidate_selector(self.container, individuals)
        # TODO process batched SelectionOutput - for GridLike container
        if not isinstance(selection_output, SelectionOutput):
            raise RuntimeError("Unknown output type from candidate_selector!")

        # TODO operation info? e.g. success inserted, failed inserted...
        container = selection_output.target_container
        if selection_output.discard_individuals:
            container.discard(selection_output.discard_individuals)
        if selection_output.selected_individuals:
            container.add(selection_output.selected_individuals)


class Evolution(BaseEvolutionAlgorithm):
    def __init__(self,
                 container: ContainerLike,
                 spawn_or_query_operator: Union[Operator, Sequence[Operator]],
                 selection_operator: Union[Operator, Sequence[Operator]],
                 mutation_operator: Union[Operator, Sequence[Operator]] = None,
                 crossover_operator: Union[Operator, Sequence[Operator]] = None,
                 num_objective: Optional[int] = None,
                 objective_weight: Union[float, FitnessLike] = 1.0):
        generator_ops = [_to_composite_operator(spawn_or_query_operator, RandomOperatorPicker)]
        if mutation_operator:
            generator_ops.append(_to_composite_operator(mutation_operator, RandomOperatorPicker))
        if crossover_operator:
            generator_ops.append(_to_composite_operator(crossover_operator, RandomOperatorPicker))

        candidate_generator = ChainedOperator(*generator_ops) if len(generator_ops) > 1 else generator_ops[0]
        candidate_selector = _to_composite_operator(selection_operator, ChainedOperator)
        super(Evolution, self).__init__(
            container,
            candidate_generator,
            candidate_selector,
            num_objective=num_objective,
            objective_weight=objective_weight
        )
