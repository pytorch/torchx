#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import os.path
import shutil
import tempfile
import unittest

from torchx.schedulers.streams import Tee


class TeeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="torchx_runner_config_test")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_combined(self) -> None:
        a_path = os.path.join(self.test_dir, "a")
        b_path = os.path.join(self.test_dir, "b")
        ab_path = os.path.join(self.test_dir, "ab")

        ab = io.open(ab_path, "wb", buffering=0)
        a = Tee(io.open(a_path, "wb", buffering=0), ab)
        b = Tee(io.open(b_path, "wb", buffering=0), ab)

        a.write(b"1")
        b.write(b"2")
        a.write(b"3")
        b.write(b"4")

        a.close()
        b.close()

        with open(a_path, "rb") as f:
            self.assertEqual(f.read(), b"13")
        with open(b_path, "rb") as f:
            self.assertEqual(f.read(), b"24")
        with open(ab_path, "rb") as f:
            self.assertEqual(f.read(), b"1234")

    def test_basic(self) -> None:
        a_path = os.path.join(self.test_dir, "a")
        b_path = os.path.join(self.test_dir, "b")
        a = Tee(
            io.open(a_path, "wb", buffering=0),
            io.open(b_path, "wb", buffering=0),
        )
        self.assertEqual(a.mode, "wb")
        self.assertEqual(a.name, a_path)
        self.assertFalse(a.closed)
        self.assertGreaterEqual(a.fileno(), 0)

        self.assertFalse(a.isatty())
        self.assertFalse(a.readable())
        self.assertFalse(a.seekable())
        self.assertTrue(a.writable())

        with self.assertRaises(NotImplementedError):
            a.read()
        with self.assertRaises(NotImplementedError):
            a.readline()
        with self.assertRaises(NotImplementedError):
            a.readlines()
        with self.assertRaises(NotImplementedError):
            a.seek(0)
        with self.assertRaises(NotImplementedError):
            a.tell()
        with self.assertRaises(NotImplementedError):
            a.truncate()
        with self.assertRaises(NotImplementedError):
            list(a)

        with a as f:
            self.assertEqual(f, a)

        a.write(b"1\n")
        a.writelines([b"2\n", b"3\n"])
        a.flush()

        a.close()
        self.assertTrue(a.closed)

        with open(a_path, "rb") as f:
            self.assertEqual(f.read(), b"1\n2\n3\n")
        with open(b_path, "rb") as f:
            self.assertEqual(f.read(), b"1\n2\n3\n")

    def test_fileno(self) -> None:
        a_path = os.path.join(self.test_dir, "a")
        ab_path = os.path.join(self.test_dir, "ab")

        ab = io.open(ab_path, "wb", buffering=0)
        a = Tee(io.open(a_path, "wb", buffering=0), ab)

        w: io.FileIO = io.open(a.fileno(), "wb", buffering=0)
        w.write(b"1")
        w.write(b"3")

        a.close()

        with open(a_path, "rb") as f:
            self.assertEqual(f.read(), b"13")
