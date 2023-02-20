# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
from numpy.lib.stride_tricks import as_strided

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.ports.ports import OutPort
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.magma.runtime.message_infrastructure import (
    ChannelQueueSize,
    GetDDSChannel,
    DDSTransportType,
    DDSBackendType,
)


class RosDVSInput(AbstractProcess):
    """
    outputting a frame from DVS camera fetched from DDSChannel
    """

    def __init__(
        self,
        true_height: int,
        true_width: int,
        down_sample_factor: int = 1,
        num_steps=1,
    ) -> None:
        super().__init__(
            true_height=true_height,
            true_width=true_width,
            down_sample_factor=down_sample_factor,
            num_steps=num_steps
        )

        down_sampled_height = true_height // down_sample_factor
        down_sampled_width = true_width // down_sample_factor

        out_shape = (down_sampled_width, down_sampled_height)
        self.event_frame_out = OutPort(shape=out_shape)


@implements(proc=RosDVSInput, protocol=LoihiProtocol)
@requires(CPU)
class RosDVSInputPM(PyLoihiProcessModel):
    event_frame_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)

    def __init__(self, proc_params):
        super().__init__(proc_params)
        self._true_height = proc_params["true_height"]
        self._true_width = proc_params["true_width"]
        self._true_shape = (self._true_width, self._true_height)

        self._down_sample_factor = proc_params["down_sample_factor"]
        self._down_sampled_height = (
            self._true_height // self._down_sample_factor
        )
        self._down_sampled_width = self._true_width // self._down_sample_factor
        self._down_sampled_shape = (
            self._down_sampled_width,
            self._down_sampled_height,
        )
        self._frame_shape = self._down_sampled_shape[::-1]

        self._num_steps = proc_params["num_steps"]
        self._cur_steps = 0

        # DDSChannel relavent
        name = 'rt/prophesee/PropheseeCamera_optical_frame/cd_events_buffer'

        self.dds_channel = GetDDSChannel(
            name,
            DDSTransportType.DDSUDPv4,
            DDSBackendType.FASTDDSBackend,
            ChannelQueueSize,
        )

        self.recv_port = self.dds_channel.dst_port
        self.recv_port.start()

    def run_spk(self):
        self._cur_steps += 1

        res = self.recv_port.recv()
        stamp = int.from_bytes(bytearray(np.flipud(res[0:8]).tolist()),
                            byteorder='big', signed=False)
        width = int.from_bytes(bytearray(np.flipud(res[8:12:]).tolist()),
                            byteorder='big', signed=False)
        height = int.from_bytes(bytearray(np.flipud(res[12:16:]).tolist()),
                                byteorder='big', signed=False)
        # print("stamp nsec = ", stamp)
        # print("width = ", width)
        # print("height = ", height)
        img_data = res[16:]
        img_data = img_data.reshape(height, width)

        img_data = \
                self._pool_2d(img_data, kernel_size=self._down_sample_factor,
                              stride=self._down_sample_factor, padding=0,
                              pool_mode='max')
        

        # * later codes assume (width, height) shaped image
        img_data = np.transpose(img_data, axes=(1, 0))
        self.event_frame_out.send(img_data)

    def post_guard(self) -> bool:
        return self._cur_steps == self._num_steps

    def run_post_mgmt(self) -> None:
        self.recv_port.join()

    def _pool_2d(self, matrix: np.ndarray, kernel_size: int, stride: int,
                 padding: int = 0, pool_mode: str = 'max'):
        # Padding
        padded_matrix = np.pad(matrix, padding, mode='constant')

        # Window view of A
        output_shape = ((padded_matrix.shape[0] - kernel_size) // stride + 1,
                        (padded_matrix.shape[1] - kernel_size) // stride + 1)
        shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
        strides_w = (stride * padded_matrix.strides[0],
                     stride * padded_matrix.strides[1],
                     padded_matrix.strides[0],
                     padded_matrix.strides[1])
        matrix_w = as_strided(padded_matrix, shape_w, strides_w)

        # Return the result of pooling
        if pool_mode == 'max':
            return matrix_w.max(axis=(2, 3))
        elif pool_mode == 'avg':
            return matrix_w.mean(axis=(2, 3))
