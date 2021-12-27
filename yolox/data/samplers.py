#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import itertools
from typing import Optional

import paddle
from paddle.io import BatchSampler, Sampler


class YoloBatchSampler(BatchSampler):
    """
    This batch sampler will generate mini-batches of (mosaic, index) tuples from another sampler.
    It works just like the :class:`paddle.utils.data.sampler.BatchSampler`,
    but it will turn on/off the mosaic aug.
    """

    def __init__(self, *args, mosaic=True, **kwargs):
        def_args = set(['dataset', 'sampler', 'shuffle', 'batch_size', 'drop_last'])
        nkwargs = {k: v for k, v in kwargs.items() if k in def_args}
        if len(args) < 3:
            nkwargs['shuffle'] = False
        else:
            args = list(args)
            args[2] = False
            args = tuple(args)
        super().__init__(*args, **nkwargs)
        self.mosaic = mosaic

    def __iter__(self):
        for batch in super().__iter__():
            yield [(self.mosaic, idx) for idx in batch]


class InfiniteSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(
        self,
        size: int,
        shuffle: bool = True,
        seed: Optional[int] = 0,
        rank=0,
        world_size=1,
    ):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        self._shuffle = shuffle
        self._seed = int(seed)
        self._rank = rank
        self._world_size = world_size

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self._world_size
        )

    def _infinite_indices(self):
        paddle.seed(self._seed)
        while True:
            if self._shuffle:
                yield from paddle.randperm(self._size)
            else:
                yield from paddle.arange(self._size)

    def __len__(self):
        return self._size // self._world_size
