# Copyright 2021 MosaicML. All Rights Reserved.

"""`Selective Backprop <https://arxiv.org/abs/1910.00762>`_ prunes minibatches according to the difficulty of the
individual training examples, and only computes weight gradients over the pruned subset, reducing iteration time and
speeding up training.

The algorithm runs on :attr:`~composer.core.event.Event.INIT` and :attr:`~composer.core.event.Event.AFTER_DATLOADER`.
On Event.INIT, it gets the loss function before the model is wrapped. On Event.AFTER_DATALOADER, it applies selective
backprop if the time is between ``self.start`` and ``self.end``.

See the :doc:`Method Card </method_cards/selective_backprop>` for more details.
"""

from composer.algorithms.selective_backprop.selective_backprop import SelectiveBackprop as SelectiveBackprop
from composer.algorithms.selective_backprop.selective_backprop import select_using_loss as select_using_loss
from composer.algorithms.selective_backprop.selective_backprop import \
    should_selective_backprop as should_selective_backprop

__all__ = ['SelectiveBackprop', 'select_using_loss', 'should_selective_backprop']
