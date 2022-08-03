# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import date, timedelta


def get_past_epoch_time(days_past: int) -> int:
    past_day = date.today() - timedelta(days_past)
    return int(past_day.strftime("%s"))
