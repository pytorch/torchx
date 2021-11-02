#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import select
import threading
from types import TracebackType
from typing import (
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
    Iterator,
    Type,
    BinaryIO,
)


class Tee(BinaryIO):
    """
    Tee is a IO writer that allows writing to multiple writers. This uses pipe
    and file descriptors so it works with subprocess.

    .. note:: Tee creates a background thread so must be closed.

    The order of writes is not guaranteed when writing to both fileno() and
    directly via write().
    """

    def __init__(self, *streams: io.FileIO) -> None:
        assert len(streams) > 0, "must have at least one stream"
        for s in streams:
            assert s.writable(), f"{s} must be writable"
            assert not s.closed, f"{s} must not be closed"
            assert s.fileno() >= 0, f"{s} must have fileno"
            assert s.mode == "wb", f"{s} must be in wb mode"

        self.streams_lock = threading.Lock()  # protects streams
        self.streams: Tuple[io.FileIO] = streams
        r, w = os.pipe()
        self._fileno: int = w
        self.r: io.FileIO = io.open(r, "rb", buffering=0)
        self._closed = False

        self.thread = threading.Thread(
            target=self._start_thread,
        )
        self.thread.daemon = True
        self.thread.start()

    def _start_thread(self) -> None:
        BUFSIZE = 64000
        while True:
            r, w, e = select.select([self.r], [], [], 0.1)
            if self.r not in r:
                if self._closed:
                    break
                continue

            data = self.r.read(BUFSIZE)
            self.write(data)

    @property
    def mode(self) -> str:
        return "wb"

    @property
    def name(self) -> "Union[os.PathLike[bytes], os.PathLike[str], bytes, int, str]":
        return self.streams[0].name

    def close(self) -> None:
        self._closed = True
        self.thread.join()
        with self.streams_lock:
            for s in self.streams:
                s.close()
            self.r.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def fileno(self) -> int:
        return self._fileno

    def flush(self) -> None:
        with self.streams_lock:
            for s in self.streams:
                s.flush()

    def isatty(self) -> bool:
        return False

    def read(self, n: Optional[int] = -1) -> bytes:
        raise NotImplementedError("_Tee doesn't support read")

    def readable(self) -> bool:
        return False

    def readline(self, limit: int = -1) -> bytes:
        raise NotImplementedError("_Tee doesn't support readline")

    def readlines(self, hint: int = -1) -> List[bytes]:
        raise NotImplementedError("_Tee doesn't support readlines")

    def seek(self, offset: int, whence: int = 0) -> int:
        raise NotImplementedError("_Tee doesn't support seek")

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        raise NotImplementedError("_Tee doesn't support tell")

    def truncate(self, size: Optional[int] = None) -> int:
        raise NotImplementedError("_Tee doesn't support truncate")

    def writable(self) -> bool:
        return True

    def write(self, s: bytes) -> int:
        with self.streams_lock:
            n = 0
            for stream in self.streams:
                n = stream.write(s)
            return n

    def writelines(self, lines: Iterable[bytes]) -> None:
        with self.streams_lock:
            for s in self.streams:
                s.writelines(lines)

    def __enter__(self) -> BinaryIO:
        return self

    def __exit__(
        self,
        t: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        pass

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        raise NotImplementedError("_Tee doesn't support __next__")


if TYPE_CHECKING:
    # Enforce that Tee is a valid BinaryIO type
    _TEE: BinaryIO = Tee()
