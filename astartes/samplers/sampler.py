"""Abstract Sampling class"""
from abc import ABC, abstractmethod

from ..exceptions import *


class Sampler(abc):
    @abstractmethod
    def __init__(self, configs):
        pass

    @abstractmethod
    def populate(self, X, y=None):
        """
        Load data in this instance.

        """
        self.X = X
        self.y = y
        self.samples = []

    @abstractmethod
    def get_sample(self):
        """
        Get one sample.

        """
        if not self.is_populated:
            raise NotInitializedError('Populate sampler instance with '
                                      'data to get samples')
        if len(self.samples) == len(X):
            raise ValueError('Dataset exhausted')

        self.samples.append(self.get_next_samle_id())
        return self.samples[-1]

    @abstractmethod
    def get_next_sample_id(self):
        """
        Get the id of the next sample.

        """
        pass

    @abstractmethod
    def get_batch_samples(self, n_samples):
        """
        Get a batch of samples

        """
        return self.X[self.get_batch_sample_idx(n_samples)]

    @abstractmethod
    def get_batch_sample_idx(self, n_samples):
        """
        Get idx of the next batch of samples.

        """
        return [self.get_next_sample_id() for _ in range(n_samples)]
