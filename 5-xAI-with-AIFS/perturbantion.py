# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from abc import ABC
from abc import abstractmethod
from types import MappingProxyType as frozendict
from typing import Any

import torch
from anemoi.inference.checkpoint import Checkpoint

LOG = logging.getLogger(__name__)


def _make_indices_mapping(indices_from: list, indices_to: list) -> frozendict:
    assert len(indices_from) == len(indices_to), (indices_from, indices_to)
    return frozendict({i: j for i, j in zip(indices_from, indices_to)})


class ExpandedCheckpoint(Checkpoint):
    """Expanded Checkpoint with additional properties."""

    @property
    def number_of_grid_points(self) -> int:
        return self._metadata.number_of_grid_points

    @property
    def variable_to_output_tensor_index(self) -> Any:
        """Get the variable to input tensor index."""
        mapping = _make_indices_mapping(
            self._metadata._indices.data.output.full,
            self._metadata._indices.model.output.full,
        )

        return frozendict({v: mapping[i] for i, v in enumerate(self._metadata.variables) if i in mapping})

    @property
    def input_tensor_index_to_variable(self) -> Any:
        """Get the output tensor index to variable."""
        mapping = _make_indices_mapping(
            self._metadata._indices.model.input.full,
            self._metadata._indices.data.input.full,
        )
        return frozendict({k: self._metadata.variables[v] for k, v in mapping.items()})


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
        self._checkpoint = ExpandedCheckpoint(checkpoint, patch_metadata=patch_metadata)

    @property
    def variable_to_output_tensor_index(self) -> dict[str, int]:
        """Return the mapping between variable name and output tensor index."""
        return self._checkpoint.variable_to_output_tensor_index

    @property
    def input_tensor_index_to_variable(self) -> dict[str, int]:
        """Return the mapping between variable name and input tensor index."""
        return self._checkpoint.input_tensor_index_to_variable

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            1,
            1,
            self._checkpoint.number_of_grid_points,
            len(self.variable_to_output_tensor_index),
        )

    @property
    def coords(self) -> torch.Tensor:
        lats = torch.from_numpy(self._checkpoint._metadata._supporting_arrays["latitudes"])
        lons = torch.from_numpy(self._checkpoint._metadata._supporting_arrays["longitudes"])
        return torch.stack([lats, lons], dim=-1)

    @abstractmethod
    def create(self, *args, **kwargs) -> torch.Tensor:
        pass
