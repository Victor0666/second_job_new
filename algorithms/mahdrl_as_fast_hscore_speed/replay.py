"""Deque order and eviction, with constant-time indexing for random.sample."""
from collections.abc import Sequence
from itertools import chain, islice
from operator import index


class IndexedReplay(Sequence):
    def __init__(self, maxlen):
        self.maxlen = index(maxlen)
        if self.maxlen < 1:
            raise ValueError("Replay capacity must be positive")
        self._items, self._start = [], 0

    def __len__(self):
        return len(self._items)

    def __getitem__(self, position):
        position = index(position)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("Replay index out of range")
        return self._items[(self._start + position) % len(self)]

    def __iter__(self):
        return chain(islice(self._items, self._start, None),
                     islice(self._items, 0, self._start))

    def append(self, transition):
        if len(self) < self.maxlen:
            self._items.append(transition)
        else:
            self._items[self._start] = transition
            self._start = (self._start + 1) % self.maxlen
