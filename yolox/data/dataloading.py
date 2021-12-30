#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random
import uuid
from multiprocessing import cpu_count

import numpy as np
import paddle
from paddle.io import DataLoader as ppDataLoader

from yolox.data.samplers import YoloBatchSampler


def get_yolox_datadir():
    """
    get dataset dir of YOLOX. If environment variable named `YOLOX_DATADIR` is set,
    this function will return value of the environment variable. Otherwise, use data
    """
    yolox_datadir = os.getenv("PP_DATADIR", None)
    if yolox_datadir is None:
        import yolox

        yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
        yolox_datadir = os.path.join(yolox_path, "datasets")
    return yolox_datadir


class DataLoader(ppDataLoader):
    """
    Lightnet dataloader that enables on the fly resizing of the images.
    See :class:`torch.utils.data.DataLoader` for more information on the arguments.
    Check more on the following website:
    https://gitlab.com/EAVISE/lightnet/-/blob/master/lightnet/data/_dataloading.py
    """
    # (dataset, feed_list=None, places=None, return_list=False, batch_sampler=None,
    # batch_size=1, shuffle=False, drop_last=False, collate_fn=None, num_workers=0,
    # use_buffer_reader=True, use_shared_memory=True, timeout=0, worker_init_fn=None)
    def __init__(self, *args, **kwargs):
        def_args = set(['feed_list',
                        'places',
                        'return_list',
                        'batch_sampler',
                        'batch_size',
                        'shuffle',
                        'drop_last',
                        'collate_fn',
                        'num_workers',
                        'use_buffer_reader',
                        'use_shared_memory',
                        'timeout',
                        'worker_init_fn'])
        nkwargs = {k: v for k, v in kwargs.items() if k in def_args}
        if nkwargs.get('use_shared_memory') is False:
            nkwargs['use_shared_memory'] = True
        # if nkwargs.get('num_workers', 0) == 0:
        #     nkwargs['num_workers'] = cpu_count()
        super().__init__(*args, **nkwargs)
        # super().__init__(*args, **kwargs)
        self.__initialized = False
        shuffle = False
        batch_sampler = None
        sampler = None
        if "shuffle" in kwargs:
            shuffle = kwargs["shuffle"]
        if "sampler" in kwargs:
            sampler = kwargs["sampler"]
        if "batch_sampler" in kwargs:
            batch_sampler = kwargs["batch_sampler"]

        # Use custom BatchSampler
        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = paddle.io.RandomSampler(self.dataset)
                    # sampler = torch.utils.data.DistributedSampler(self.dataset)
                else:
                    sampler = paddle.io.SequenceSampler(self.dataset)
            batch_sampler = YoloBatchSampler(
                None,
                sampler,
                False,
                self.batch_size,
                self.drop_last,
                input_dimension=self.dataset.input_dim,
            )
            # batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations =

        self.batch_sampler = batch_sampler

        self.__initialized = True

    def close_mosaic(self):
        self.batch_sampler.mosaic = False


def list_collate(batch):
    """
    Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of
    items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = paddle.io.default_collate_fn(items[i])

    return items


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2 ** 32
    random.seed(seed)
    paddle.seed(seed)
    np.random.seed(seed)
