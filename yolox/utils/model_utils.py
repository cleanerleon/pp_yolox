#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from copy import deepcopy

import paddle
import paddle.nn as nn
import numpy as np

def get_model_info(model, tsize):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    # 获取参数情况
    for p in model.parameters():
        mulValue = np.prod(p.shape)  # 使用numpy prod接口计算数组所有元素之积
        total_params += mulValue  # 总参数量
        if p.stop_gradient:
            non_trainable_params += mulValue  # 可训练参数量
        else:
            trainable_params += mulValue  # 非可训练参数量

    info = "Params: {:.2f}M".format(total_params)
    print(info)

    # stride = 64
    # img = paddle.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    # flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    # params /= 1e6
    # flops /= 1e9
    # flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    # info = "Params: {:.2f}M, Gflops: {:.2f}".format(params, flops)
    return info


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = paddle.diag(bn.weight.div(paddle.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(paddle.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        paddle.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        paddle.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(paddle.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def fuse_model(model):
    from yolox.models.network_blocks import BaseConv

    for m in model.modules():
        if type(m) is BaseConv and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


def replace_module(module, replaced_module_type, new_module_type, replace_func=None):
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model
