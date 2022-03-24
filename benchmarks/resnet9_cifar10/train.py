# Copyright 2021 MosaicML. All Rights Reserved.

"""Entrypoint that runs the Composer trainer on a provided YAML hparams file.

Adds a --datadir flag to conveniently set a common
data directory for both train and validation datasets.

Example that trains MNIST with label smoothing::

    >>> python examples/run_composer_trainer.py
    -f composer/yamls/models/classify_mnist_cpu.yaml
    --algorithms label_smoothing --alpha 0.1
    --datadir ~/datasets
"""
import os
import sys
import textwrap
import yaml

from composer.trainer import TrainerHparams
from composer.utils import run_directory

import composer

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams



@dataclass
class CIFARResNet9Hparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for :class:`.CIFAR10_ResNet9`.

    Args:
        num_classes (int, optional): The number of classes. Needed for classification tasks. Default: ``10``.
    """
    num_classes: int = hp.optional("The number of classes.  Needed for classification tasks", default=10)

    def initialize_object(self):
        from model import CIFAR10_ResNet9
        return CIFAR10_ResNet9(num_classes=self.num_classes)



def main() -> None:
    if len(sys.argv) == 1:
        sys.argv = [sys.argv[0], "--help"]

    composer.trainer.trainer_hparams.model_registry['resnet9_cifar10'] = CIFARResNet9Hparams

    hparams = TrainerHparams.create(cli_args=True)  # reads cli args from sys.argv
    trainer = hparams.initialize_object()
    if hparams.save_folder is not None:
        # If saving a checkpoint, dump the hparams to the checkpoint folder
        hparams_path = os.path.join(run_directory.get_run_directory(), hparams.save_folder, "hparams.yaml")
        try:
            with open(hparams_path, "x") as f:
                # Storing the config (ex. hparams) in a separate file so they can be modified before resuming
                f.write(hparams.to_yaml())
        except FileExistsError as e:
            with open(hparams_path, "r") as f:
                # comparing the parsed hparams to ignore whitespace and formatting differences
                if hparams.to_dict() != yaml.safe_load(f):
                    raise RuntimeError(
                        textwrap.dedent(f"""\
                        The hparams in the existing checkpoint folder {hparams_path}
                        differ from those being used in the current training run.
                        Please specify a new checkpoint folder.""")) from e
    print("calling trainer.fit()")
    trainer.fit()


if __name__ == "__main__":
    main()
