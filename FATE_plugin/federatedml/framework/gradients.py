#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import abc
import operator
from arch.api.utils import log_utils

import numpy as np

from arch.api.utils.splitable import segment_transfer_enabled
from federatedml.secureprotol.encrypt import Encrypt
LOGGER = log_utils.getLogger()

FRAGMENT_16M = 0x1000000
FRAGMENT_24M = 0x1100000


class TransferableGradients(metaclass=segment_transfer_enabled(max_part_size=FRAGMENT_24M)):
    def __init__(self, gradients, cls, *args, **kwargs):
        self._gradients = gradients
        self._cls = cls
        if args:
            self._args = args
        if kwargs:
            self._kwargs = kwargs

    @property
    def unboxed(self):
        return self._gradients

    @property
    def gradients(self):
        if not hasattr(self, "_args") and not hasattr(self, "_kwargs"):
            return self._cls(self._gradients)
        else:
            args = self._args if hasattr(self, "_args") else ()
            kwargs = self._kwargs if hasattr(self, "_kwargs") else {}
            return self._cls(self._gradients, *args, **kwargs)


class Gradients(object):

    def __init__(self, g):
        self._gradients = g

    def for_remote(self):
        return TransferableGradients(self._gradients, self.__class__)

    @property
    def unboxed(self):
        return self._gradients
