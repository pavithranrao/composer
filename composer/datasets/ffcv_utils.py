import json
import logging
import textwrap
from typing import Optional, Sequence

import numpy as np
import torch

import numpy as np

from composer.core.types import Dataset
from composer.datasets.webdataset_utils import init_webdataset_meta
from composer.utils import dist

log = logging.getLogger(__name__)

__all__ = ["write_ffcv_dataset", "ffcv_init_sampler", "ffcv_monkey_patches"]

try:
    import ffcv  # type: ignore
    ffcv_installed = True
except ImportError:
    ffcv_installed = False


def _require_ffcv():
    if not ffcv_installed:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without ffcv support.
            To use ffcv with Composer, please install ffcv in your environment."""))


class RandomOrder(ffcv.traversal_order.base.TraversalOrder):

    def __init__(self, loader: 'Loader'):
        super().__init__(loader)

        if self.distributed:
            self.sampler = torch.utils.data.DistributedSampler(
                self.indices,
                shuffle=True,
                seed=self.seed,
                drop_last=False,
                num_replicas=dist.get_world_size(),
                rank=dist.get_global_rank(),
            )

    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            generator = np.random.default_rng(self.seed + epoch if self.seed is not None else None)
            return generator.permutation(self.indices)

        self.sampler.set_epoch(epoch)

        return self.indices[np.array(list(self.sampler))]


class SequentialOrder(ffcv.traversal_order.base.TraversalOrder):

    def __init__(self, loader: 'Loader'):
        super().__init__(loader)

        if self.distributed:
            self.sampler = torch.utils.data.DistributedSampler(
                self.indices,
                shuffle=False,
                seed=self.seed,
                drop_last=False,
                num_replicas=dist.get_world_size(),
                rank=dist.get_global_rank(),
            )

    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            return self.indices

        self.sampler.set_epoch(epoch)

        return self.indices[np.array(list(self.sampler))]


def ffcv_init_sampler():
    _require_ffcv()
    ffcv.loader.loader.ORDER_MAP[ffcv.loader.loader.OrderOption.RANDOM] = RandomOrder
    ffcv.loader.loader.ORDER_MAP[ffcv.loader.loader.OrderOption.SEQUENTIAL] = SequentialOrder

def ffcv_monkey_patches():
    _require_ffcv()

    def new_len(self):
        if not hasattr(self, "init_traversal_order"):
            self.init_traversal_order = self.next_traversal_order()
        if self.drop_last:
            return len(self.init_traversal_order) // self.batch_size
        else:
            return int(np.ceil(len(self.init_traversal_order) / self.batch_size))

    ffcv.loader.loader.Loader.__len__ = new_len

def write_ffcv_dataset(dataset: Optional[Dataset] = None,
                       remote: Optional[str] = None,
                       write_path: str = "/tmp/dataset.ffcv",
                       max_resolution: Optional[int] = None,
                       num_workers: int = 16,
                       write_mode: str = 'raw',
                       compress_probability: float = 0.50,
                       jpeg_quality: float = 90,
                       chunk_size: int = 100):
    """Converts PyTorch ``dataset`` or webdataset at ``remote`` into FFCV format at filepath ``write_path``.

    Args:
        dataset (Iterable[Sample]): A PyTorch dataset. Default: ``None``.
        remote (str): A remote path for webdataset. Default: ``None``.
        write_path (str): Write results to this file. Default: ``"/tmp/dataset.ffcv"``.
        max_resolution (int): Limit resolution if provided. Default: ``None``.
        num_workers (int): Numbers of workers to use. Default: ``16``.
        write_mode (str): Write mode for the dataset. Default: ``'raw'``.
        compress_probability (float): Probability with which image is JPEG-compressed. Default: ``0.5``.
        jpeg_quality (float): Quality to use for jpeg compression. Default: ``90``.
        chunk_size (int): Size of chunks processed by each worker during conversion. Default: ``100``.
    """

    _require_ffcv()
    if dataset is None and remote is None:
        raise ValueError("At least one of dataset or remote should not be None.")

    log.info(f"Writing dataset in FFCV <file>.ffcv format to {write_path}.")
    writer = ffcv.writer.DatasetWriter(write_path, {
        'image':
            ffcv.fields.RGBImageField(write_mode=write_mode,
                                      max_resolution=max_resolution,
                                      compress_probability=compress_probability,
                                      jpeg_quality=jpeg_quality),
        'label':
            ffcv.fields.IntField()
    },
                                       num_workers=num_workers)
    if dataset:
        writer.from_indexed_dataset(dataset, chunksize=chunk_size)
    elif remote is not None:
        pipeline = lambda dataset: dataset.decode('pil').to_tuple('jpg', 'cls')

        text = init_webdataset_meta(remote)

        metadata = json.loads(text)
        num_shards = metadata['n_shards']
        if metadata['n_leftover'] > 0:
            num_shards = num_shards + 1

        if remote.startswith('s3://'):
            urls = [f'pipe: aws s3 cp {remote}/{idx:05d}.tar -' for idx in range(num_shards)]
        else:
            urls = [f'{remote}/{idx:05d}.tar' for idx in range(num_shards)]

        lengths = np.repeat(metadata['samples_per_shard'], metadata['n_shards'])
        lengths = np.insert(lengths, 0, 0)
        # we don't have n_leftover in the lengths array so we don't need to
        # remove it from offsets.
        offsets = np.cumsum(lengths)
        todos = zip(urls, offsets)
        total_len = metadata['samples_per_shard'] * metadata['n_shards'] + metadata['n_leftover']
        # We call this internal API instead of writer.from_webdataset because with writer.from_webdataset
        # FFCV downloads the whole dataset twice.
        writer._write_common(total_len, todos, ffcv.writer.worker_job_webdataset, (pipeline,))

        #writer.from_webdataset(urls, pipeline)
