from typing import Sequence

import numpy as np

from wavesim.engine import (
    Array,
    empty_like,
    block_enumerate,
    edges,
    BlockArray,
)


class DomainController:
    """Domain controller that implements multi-domain correction through the medium operator."""

    def __init__(self, domains, periodic: Sequence[bool], edge_widths: np.ndarray, block_boundaries: list[list[int]]):
        """

        Args:
            periodic:
            edge_widths:
        """
        self.domains = domains
        self.periodic = np.asarray(periodic)
        self.ndim = len(self.domains[0].shape)
        self.edge_widths = edge_widths
        self.block_boundaries = block_boundaries

        # Allocate memory for incoming and outgoing edge corrections.
        # for each edge of each subdomain, define four buffers:
        # edges_out: for storing the computed edge correction
        # transfer_in: for receiving the edge correction from the neighbors.
        # wrapping_in: for receiving the edge correction from the wrapping correction.
        #               These are _references_ to elements of edges_out
        # transfer_out: for sending the edge correction to the neighbor.
        #               These are _references_ to elements of transfer_in
        # store the edges such that:
        # * zip(edges_out, wrapping_in) gives the correct combination for wrapping correction.
        # * zip(edges_out, transfer_out) gives the correct combinations for transfer
        #
        for idx, block in block_enumerate(self.domains):
            edge_templates = edges(block.B_scat, widths=self.edge_widths, empty_as_none=True)
            for (d, side), template in np.ndenumerate(edge_templates):
                if template is None:
                    continue  # no edge corrections needed since boundary_width is zero

                block.edges_out[d, side] = empty_like(template)
                block.wrapping_in[d, 1 - side] = block.edges_out[d, side]  # todo: remove (always equal to edges_out)

                # find the neighboring subdomain if it exists
                neighbor_idx = np.asarray(idx)
                neighbor_idx[d] += 1 if side == 1 else -1
                if self.periodic[d]:
                    neighbor_idx[d] %= self.domains.shape[d]
                elif neighbor_idx[d] < 0 or neighbor_idx[d] >= self.domains.shape[d]:
                    continue  # at the edge: no neighbor in this direction

                block.transfer_in[d, side] = empty_like(template)

                # tell the neighbor where to store the transfer correction
                neighbor = self.domains[tuple(neighbor_idx)]
                neighbor.transfer_out[d, 1 - side] = block.transfer_in[d, side]

    def medium(self, x: Array, out: BlockArray):
        """Apply the medium operator, allow for multi-domain correction."""
        self._apply_all_domains("medium", x, out)

    def propagator(self, x: Array, out: BlockArray):
        """Apply the propagator, allow for multi-domain correction."""
        self._apply_all_domains("propagator", x, out)

    def _apply_all_domains(self, coroutine_name, x: Array, out: BlockArray):
        # handle data that is not split yet (or sparse)
        if self.domains.size > 1 and not isinstance(x, BlockArray):
            x = BlockArray(x, boundaries=self.block_boundaries, factories=out.factories)

        # start the coroutines for each domain
        coroutines = np.empty_like(self.domains)
        for idx, dom, x_, out_ in block_enumerate(self.domains, x, out):
            coroutines[idx] = getattr(dom, coroutine_name)(x_, out=out_)

        while True:
            if any([next(c) for c in coroutines.flat]):  # need to continue if any block has more work to do
                self.synchronize()
            else:
                break

    def inverse_propagator(self, x: Array, out: Array):
        """Apply the inverse propagator, allow for multi-domain correction."""
        self._apply_all_domains("inverse_propagator", x, out)

    def synchronize(self):
        pass  # This method is a placeholder for synchronization logic, if needed.
