# -*- coding:utf-8 -*-

import argparse

from loguru import logger

from yolox.core.trainer import Trainer
from yolox.exp.build import get_exp


def make_parser():
    parser = argparse.ArgumentParser("PP-YOLOX train parser")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    return parser


@logger.catch
def main(exp, args):
    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    # if not args.experiment_name:
    args.experiment_name = exp.exp_name

    main(exp, args)
