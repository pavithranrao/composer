# Copyright 2021 MosaicML. All Rights Reserved.

"""Performance profiling tools.

The profiler gathers performance metrics during a training run that can be used to diagnose bottlenecks and
facilitate model development.

The metrics gathered include:

* Duration of each :class:`.Event` during training
* Time taken by the data loader to return a batch
* Host metrics such as CPU, system memory, disk and network utilization over time
* Execution order, latency and attributes of PyTorch operators and GPU kernels (see :doc:`torch:profiler`)

See the :doc:`Profiling Guide </trainer/performance_tutorials/profiling>` for additional information.
"""
from composer.profiler.json_trace_handler import JSONTraceHandler
from composer.profiler.marker import Marker
from composer.profiler.profiler import Profiler
from composer.profiler.profiler_action import ProfilerAction
from composer.profiler.profiler_hparams import (CyclicProfilerScheduleHparams, JSONTraceHparams, ProfileScheduleHparams,
                                                TraceHandlerHparams)
from composer.profiler.profiler_schedule import cyclic_schedule
from composer.profiler.system_profiler import SystemProfiler
from composer.profiler.torch_profiler import TorchProfiler
from composer.profiler.trace_handler import TraceHandler

# All needs to be defined properly for sphinx autosummary
__all__ = [
    "Marker",
    "Profiler",
    "ProfilerAction",
    "TraceHandler",
    "cyclic_schedule",
    "JSONTraceHandler",
    "CyclicProfilerScheduleHparams",
    "ProfileScheduleHparams",
    "JSONTraceHparams",
    "TraceHandlerHparams",
    "SystemProfiler",
    "TorchProfiler",
]
