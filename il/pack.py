"""Fast hand-over of a collated training batch between processes.

A FirstLight IL batch keeps one observation record per turn, so a batch of 8 game sides is ~10^5
small tensors. Pickled as tensors, every one goes through torch's reducer: 6 s to pickle in the
worker and 5 s to unpickle in the training process for each ~600 MB batch (measured), all of it
time the GPU waits. pack() copies every tensor into one contiguous buffer and pickles only the
record structure, with each tensor replaced by its index; unpack() rebuilds the tensors as views
of that buffer. Same values, dtypes and shapes; one buffer instead of 10^5 storages.
"""
from __future__ import annotations

import io
import pickle

import torch

_ALIGN = 16


class _Packer(pickle.Pickler):
    def __init__(self, file) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.tensors: list[torch.Tensor] = []
        self._index: dict[int, int] = {}

    def persistent_id(self, value):
        if isinstance(value, torch.Tensor):
            key = id(value)
            index = self._index.get(key)
            if index is None:
                index = self._index[key] = len(self.tensors)
                self.tensors.append(value)
            return index
        return None


class _Unpacker(pickle.Unpickler):
    def __init__(self, file, tensors: list[torch.Tensor]) -> None:
        super().__init__(file)
        self._tensors = tensors

    def persistent_load(self, index):
        return self._tensors[index]


def pack(value) -> tuple[bytes, list[tuple[int, int, torch.dtype, tuple[int, ...]]], bytearray]:
    """-> (structure, tensor layout, data). CPU tensors only; a tensor that appears several times
    (FirstLight repeats the last observation to pad a sequence) is stored once."""
    stream = io.BytesIO()
    packer = _Packer(stream)
    packer.dump(value)
    layout, offset = [], 0
    for tensor in packer.tensors:
        size = tensor.numel() * tensor.element_size()
        layout.append((offset, size, tensor.dtype, tuple(tensor.shape)))
        offset += -(-size // _ALIGN) * _ALIGN
    data = bytearray(offset)
    if offset:
        flat = torch.frombuffer(data, dtype=torch.uint8)
        for tensor, (start, size, _dtype, _shape) in zip(packer.tensors, layout):
            if size:
                flat[start:start + size].copy_(tensor.detach().contiguous().reshape(-1).view(torch.uint8))
    return stream.getvalue(), layout, data


def unpack(packed):
    structure, layout, data = packed
    flat = torch.frombuffer(data, dtype=torch.uint8) if len(data) else torch.empty(0, dtype=torch.uint8)
    tensors = [flat[start:start + size].view(dtype).reshape(shape) for start, size, dtype, shape in layout]
    return _Unpacker(io.BytesIO(structure), tensors).load()
