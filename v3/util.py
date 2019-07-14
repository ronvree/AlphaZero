import collections
import numpy as np


class Distribution(collections.MutableMapping):

    def __init__(self, d: dict):
        self.dist = dict(d)

    def __setitem__(self, k, v) -> None:
        self.dist[k] = v

    def __delitem__(self, k) -> None:
        del self.dist[k]

    def __getitem__(self, k):
        return self.dist[k]

    def __len__(self) -> int:
        return len(self.dist)

    def __iter__(self) -> iter:
        return iter(self.dist)

    def sample(self):  # TODO - test
        # Sample a random number in [0, 1)
        r = np.random.random() * sum(self.dist.values())  # Note: Mult with dist sum as it might not sum to 1 exactly
        # Let r point to some item in the distribution
        for a, p in self.dist.items():
            if r <= p:
                return a
            r -= p
        raise Exception('Something went wrong when sampling!')

    def sample_max(self):
        return max(self.dist.keys(), key=self.dist.get)
