import sys
from typing import Any, Iterable, Iterator, List, MutableSet, Optional, Sequence

from stew.base import registry, T


@registry.register
class OrderedSet(MutableSet[T], Sequence[T]):
    """A MutableSet variant that conserves entries order, and can be accessed like a Sequence.
    This implementation is not optimised, but does not requires the type ``T`` of items to be hashable.
    It also does not implement indexing by slices.

    Parameters
    ----------
    iterable: Optional[Iterable[T]]
        items to add to the OrderedSet
    """

    _items: List[T]  # Internal storage

    def __init__(self, iterable: Optional[Iterable] = None) -> None:
        self._items = []
        if iterable is not None:
            self.update(iterable)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, key):
        # if isinstance(key, slice):
        #     raise NotImplementedError
        return self._items[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._items)

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def __delitem__(self, idx) -> None:
        del self._items[idx]

    def count(self, key: T) -> int:
        return 1 if key in self else 0

    def index(self, key: T, start: int = 0, stop: int = sys.maxsize) -> int:
        try:
            return self._items.index(key, start, stop)
        except ValueError:
            raise KeyError(f"{key} is not in the OrderedSet")

    def add(self, key: T) -> None:
        """Add ``key`` to this OrderedSet, if it is not already present. """
        if key not in self._items:
            self._items.append(key)

    def discard(self, key: T) -> None:
        """Discard ``key`` in this OrderedSet. Does not raise an exception if absent."""
        try:
            self._items.remove(key)
        except ValueError:
            return

    def update(self, iterable: Iterable) -> None:
        """Add the items in ``iterable``, if they are not already present in the OrderedSet.  """
        try:
            for item in iterable:
                self.add(item)
        except TypeError:
            raise ValueError(f"Argument needs to be an iterable, got {type(iterable)}")
