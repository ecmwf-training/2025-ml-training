# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC, abstractmethod
import logging
from typing import Any

import torch

from anemoi.inference.checkpoint import Checkpoint

LOG = logging.getLogger(__name__)


class Perturbation(ABC):
    """Base Perturbation"""
    def __init__(
        self,
        checkpoint: str,
        patch_metadata: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        """Initialize the Perturbation.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint.
        """
        self._checkpoint = Checkpoint(checkpoint, patch_metadata=patch_metadata)

    @property
    def variable_to_output_tensor_index(self) -> dict[str, int]:
        return self._checkpoint._metadata.variable_to_output_tensor_index

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            1,
            1,
            self._checkpoint._metadata.number_of_grid_points,
            len(self._checkpoint._metadata.variable_to_output_tensor_index),
        )

    @property
    def coords(self) -> torch.Tensor:
        lats = torch.from_numpy(self._checkpoint._metadata._supporting_arrays["latitudes"])
        lons = torch.from_numpy(self._checkpoint._metadata._supporting_arrays["longitudes"])
        return torch.stack([lats, lons], dim=-1)

    @abstractmethod
    def create(self, *args, **kwargs) -> torch.Tensor:
        pass
