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

"""Some base classes, stubs and types."""

import copy
import inspect
import pickle
import textwrap
import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, MutableMapping, Optional, overload, Sequence, TypeVar, Union

import numpy as np


__all__ = ['FitnessLike', 'FeaturesLike', 'registry', 'Summarisable', 'Saveable', 'Copyable', 'CreatableFromConfig',
           'Module', 'Operator', 'IndividualLike', 'ContainerLike', 'IndividualSequence', 'AlgorithmLike']

# Interface & Stubs

T = TypeVar("T")
FitnessLike = np.ndarray
FeaturesLike = np.ndarray


# BASE CLASSES #
# Copy from qdpy library
class Summarisable(object):
    """Describes a class that can be summarised by using the `self.summary` method.
    The summarised information is provided by the `self.__get_summary_state__` method. """

    def __get_summary_state__(self) -> Mapping[str, Any]:
        """Return a dictionary containing the relevant entries to build a summary of the class.
        By default, it includes all public attributes of the class. Must be overridden by subclasses."""
        # Find public attributes
        entries = {}
        for k, v in inspect.getmembers(self):
            if not k.startswith('_') and not inspect.ismethod(v):
                try:
                    entries[k] = v
                except Exception:
                    pass
        return entries

    def summary(self, max_depth: Optional[int] = None, max_entry_length: Optional[int] = 250) -> str:
        """Return a summary description of the class.
        The summarised information is provided by the `self.__get_summary_state__` method.

        Parameters
        ----------
        :param max_depth: Optional[int]
            The maximal recursion depth allowed. Used to summarise attributes of `self` that are also Summarisable.
            If the maximal recursion depth is reached, the attribute is only described with a reduced representation (`repr(attribute)`).
            If `max_depth` is set to None, there are no recursion limit.
        :param max_entry_length: Optional[int]
            If `max_entry_length` is not None, the description of a non-Summarisable entry exceeding `max_entry_length`
            is cropped to this limit.
        """
        res: str = f"Summary {self.__class__.__name__}:\n"
        subs_max_depth = max_depth - 1 if max_depth is not None else None
        summary_state = self.__get_summary_state__()
        for i, k in enumerate(summary_state.keys()):
            v = summary_state[k]
            res += f"  {k}:"
            if isinstance(v, Summarisable):
                if max_depth is None or max_depth > 0:
                    res += textwrap.indent(v.summary(subs_max_depth), '  ')
                else:
                    res += f" {repr(v)}"
            else:
                str_v = f" {v}"
                str_v = str_v.replace("\n", " ")
                if max_entry_length is not None and len(str_v) > max_entry_length:
                    str_v = str_v[:max_entry_length - 4] + " ..."
                res += str_v
            if i != len(summary_state) - 1:
                res += "\n"
        return res

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class Saveable(object):
    """Describes a class with internal information that can be saved into an output file.
    The list of attributes from `self` that are saved in the output file are provided by the method `self.__get_saved_state__`.
    """

    def __get_saved_state__(self) -> Mapping[str, Any]:
        """Return a dictionary containing the relevant information to save.
        By default, it includes all public attributes of the class. Must be overridden by subclasses."""
        # Find public attributes
        entries = {k: v for k, v in inspect.getmembers(self) if not k.startswith('_') and not inspect.ismethod(v)}
        return entries

    def save(self, output_path: str, output_type: str = "pickle") -> None:
        """Save the into an output file.

        Parameters
        ----------
        :param output_path: str
            Path of the output file.
        :param output_type: str
            Type of the output file. Currently, only supports "pickle".
        """
        if output_type != "pickle":
            raise ValueError(f"Invalid `output_type` value. Currently, only supports 'pickle'.")
        saved_state = self.__get_saved_state__()
        with open(output_path, "wb") as f:
            pickle.dump(saved_state, f)


class Copyable(object):
    """Describes a class capable to be copied and deepcopied."""

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))  # FIXME??
        return result


class CreatableFromConfig(object):
    """Describe a class capable to be created from a configuration dictionary."""

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> Any:
        """Create a class using the information from `config` to call the class constructor.

        Parameters
        ----------
        :param config: Mapping[str, Any]
            The configuration mapping used to create the class. For each entry, the key corresponds
            to the name of a parameter of the constructor of the class.
        :param kwargs: Any
            Additional information used to create the class. The configuration entries from kwargs
            take precedence over the entry in `config`.
        """
        final_kwargs = {**config, **kwargs}
        return cls(**final_kwargs)  # type: ignore


# TODO Replace storage with torch style module
class Module(object):
    def _get_name(self):
        return self.__class__.__name__

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        lines = extra_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def extra_repr(self):
        return ""


class Operator(Module):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs): ...

    def valid_check(self, individual) -> bool:
        """
        Check whether the individual is valid for the operator input.
        """
        return True  # Default


class IndividualLike(ABC):
    name: str
    elapsed: float

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def fitness(self) -> FitnessLike: ...

    @property
    @abstractmethod
    def features(self) -> FeaturesLike: ...

    @property
    @abstractmethod
    def weight(self) -> FitnessLike: ...

    @property
    @abstractmethod
    def objective_value(self) -> FitnessLike: ...

    def __setitem__(self, key, values) -> None: ...


class ContainerLike(ABC):
    name: Optional[str]

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...

    @overload
    def add(self, individual: IndividualLike, **kwargs) -> Optional[Union[int, Sequence]]: ...

    @overload
    def add(self, individuals: Sequence[IndividualLike], **kwargs) -> Optional[Union[int, Sequence]]: ...

    @abstractmethod
    def add(self, individual, **kwargs) -> Optional[Union[int, Sequence]]: ...

    @overload
    def discard(self,
                individual: IndividualLike,
                also_from_depot: bool = False, **kwargs) -> Sequence[IndividualLike]: ...

    @overload
    def discard(self,
                individuals: Sequence[IndividualLike],
                also_from_depot: bool = False, **kwargs) -> Sequence[IndividualLike]: ...

    @abstractmethod
    def discard(self, individual, also_from_depot: bool = False, **kwargs) -> Sequence[IndividualLike]: ...

    @abstractmethod
    def clear(self, also_from_depot: bool = False): ...

    @property
    @abstractmethod
    def active_items(self) -> Sequence[IndividualLike]:
        """
        Return the sequences of individuals that is in active status.
        An individual is active if it has been added to the container and has not been discarded yet.
        If self.depot is None, this property is equivalent to the Container itself.
        """
        pass

    @property
    @abstractmethod
    def depot(self) -> Sequence[IndividualLike]:
        """
        Return all individuals that have been added to this container regardless of their active status.
        If the container is initialized with keep_discarded_individual=False, this property is equivalent to self.
        """
        pass

    @property
    @abstractmethod
    def num_discarded(self) -> Union[int, np.ndarray]:
        """Return the number of individuals discarded by the container since its creation. """
        pass

    @property
    @abstractmethod
    def num_added(self) -> Union[int, np.ndarray]:
        """Return the number of individuals added into the container since its creation. """
        pass

    @property
    @abstractmethod
    def num_rejected(self) -> Union[int, np.ndarray]:
        """Return the number of individuals rejected by the container since its creation. """

    @property
    @abstractmethod
    def num_operations(self) -> Union[int, np.ndarray]:
        """Return the number of adds and discards since the creation of this container. """
        pass

    @property
    @abstractmethod
    def size(self) -> Union[int, np.ndarray]:
        """Return the size of the container (i.e. number of items, spots, bins, etc)."""
        pass

    @property
    @abstractmethod
    def capacity(self) -> Union[int, float, np.ndarray]:
        """Return the capacity of the container (i.e. maximal number of items/spots/bins/etc). Can be math.inf."""
        pass

    @property
    @abstractmethod
    def free(self) -> Union[int, float, np.ndarray]:
        """Return the number of free spots in the container. Can be math.inf."""
        pass


IndividualSequence = Union[Sequence[IndividualLike], ContainerLike]


class AlgorithmLike(ABC):

    @property
    @abstractmethod
    def container(self) -> ContainerLike: ...

    @property
    @abstractmethod
    def candidate_generator(self) -> Operator: ...

    @property
    @abstractmethod
    def candidate_selector(self) -> Operator: ...

    @property
    @abstractmethod
    def objective_weight(self) -> FitnessLike: ...

    @property
    @abstractmethod
    def num_objectives(self) -> int: ...

    @abstractmethod
    def ask(self) -> IndividualSequence: ...

    @overload
    def tell(self,
             individuals: IndividualSequence,
             fitness: Optional[Sequence[FitnessLike]] = None,
             features: Optional[Sequence[FeaturesLike]] = None,
             elapsed: Optional[Sequence[float]] = None): ...

    @overload
    def tell(self,
             individual: IndividualLike,
             fitness: Optional[Sequence[FitnessLike]] = None,
             features: Optional[Sequence[FeaturesLike]] = None,
             elapsed: Optional[Sequence[float]] = None): ...

    @abstractmethod
    def tell(self, individuals, fitness=None, features=None, elapsed=None): ...


# Inspired from Nevergrad (MIT License) Registry class
# (https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/common/decorators.py)
class Registry(dict):
    """Registers function or classes as a dict."""

    def __init__(self) -> None:
        super().__init__()
        self._information: MutableMapping[str, Mapping] = {}

    def register(self, obj: Any, info: Optional[Mapping[Any, Any]] = None) -> Any:
        """Decorator method for registering functions/classes
        The `info` variable can be filled up using the register_with_info
        decorator instead of this one.
        """
        name = obj.__name__
        if name in self:
            # raise RuntimeError(f'Encountered a name collision "{name}".')
            warnings.warn(f'Encountered a name collision "{name}".')
            return self[name]
        self[name] = obj
        if info is not None:
            self._information[name] = info
        return obj

    def get_info(self, name: str) -> Mapping[str, Any]:
        if name not in self:
            raise ValueError(f'`{name}` is not registered.')
        return self._information.setdefault(name, {})


registry = Registry()


class Factory(dict):  # FIXME
    """Build objects from configuration."""

    def __init__(self, obj_registry=None) -> None:
        super().__init__()
        if obj_registry is None:
            obj_registry = registry
        self.registry = obj_registry

    def _get_name(self, obj: Any, key: Optional[str], config: Mapping[str, Any]) -> Optional[str]:
        """Check if `obj` possess a name, and return it."""
        name: str = ""
        if hasattr(obj, "name"):
            return obj.name
        elif key is not None:
            return key
        elif "name" in config:
            return config["name"]
        else:
            return None

    def _build_internal(self, config: MutableMapping[str, Any], default_params: Mapping[str, Any] = None,
                        **kwargs: Any) -> Any:
        # Recursively explore config and create objects for each entry possessing a "type" configuration entry
        built_objs = []
        default_config = {}
        if default_params is not None:
            default_config.update(**default_params)

        for k, v in config.items():
            if isinstance(v, Iterable) and "type" in v:
                sub_obj = self._build_internal(v, {**default_config, "name": k})
                sub_name = self._get_name(sub_obj, k, v)
                if sub_name is not None and len(sub_name) > 0 and not sub_name in self:
                    self[sub_name] = sub_obj
                    built_objs.append(sub_obj)
                config[k] = sub_obj
            elif isinstance(v, str) and v in self:
                config[k] = self[v]
            elif isinstance(v, Mapping):
                v_ = copy.copy(v)
                config[k] = []
                for k2, v2 in v_.items():
                    if isinstance(k2, str) and isinstance(v2, int) and k2 in self:
                        for _ in range(v2):
                            new_k2 = copy.deepcopy(self[k2])
                            new_k2.reinit()
                            config[k].append(new_k2)
            elif isinstance(v, Iterable):
                for i, val in enumerate(v):
                    if isinstance(val, str) and val in self:
                        new_val = copy.deepcopy(self[val])
                        new_val.reinit()
                        config[k][i] = new_val
            if not isinstance(v, Iterable) or (isinstance(v, Iterable) and "type" not in v):
                default_config[k] = config[k]

        # If the configuration describes an object, build it
        if "type" in config:
            # Retrieve the class from registry
            type_name: str = config["type"]
            assert type_name in self.registry, \
                f"The class `{type_name}` is not declared in the registry." \
                f" To get the list of registered classes, use: 'print(registry.keys())'."
            cls: Any = self.registry[type_name]
            assert issubclass(cls, CreatableFromConfig), \
                f"The class `{cls}` must inherit from `CreatableFromConfig` to be built."
            # Build object
            obj: Any = cls.from_config(config, **default_params, **kwargs)
            # If 'obj' possess a name, add it to the factory
            name: Optional[str] = self._get_name(obj, None, config)
            if name is not None and len(name) > 0 and not name in self:
                self[name] = obj
            return obj

        else:  # Return all built objects
            return built_objs

    def build(self, config: Mapping[str, Any], default_params: Mapping[str, Any] = None, **kwargs: Any) -> Any:
        """Create and return an object from `self.registry` based on
        configuration entries from `self` and `config`. The class of the created
        object must inherit from the Class `CreatableFromConfig`.
        The object is created by iteratively executing the `from_config` methods.
        The class of the object must be specified in configuration entry 'type'.

        If a configuration entry contains a sub-entry 'type', it also created by the factory.
        If it also contains a sub-entry 'name', it is added to `self` with key 'name', and accessible
        through `self[name]`.

        Parameters
        ----------
        :param config: Mapping[str, Any]
            The mapping containing configuration entries of the built object.
        :param default_params:
            Additional configuration entries to send to the `from_config` class method when creating the main object and all sub-objects.
        :param kwargs: Any
            Additional configuration entries to send to the `from_config` class method when creating the main object.
        """
        if default_params is None:
            default_params = {}
        if "name" in config and config["name"] in self:
            return self[config["name"]]
        final_config: MutableMapping[str, Any] = dict(copy.deepcopy(config))
        return self._build_internal(final_config, default_params, **kwargs)
