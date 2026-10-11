#!/usr/bin/env python3
"""Reader for MACIS block-sparse impurity RDM files (rho_blocks.bin).

File format written by ImpRdmBlocks::write_blocks_bin (all little-endian):

  magic        char[8]   "RDMBLOCK"
  version      uint32    1
  basis_size   uint64    full impurity Fock dimension D = 4**n_imp
  n_imp        int32
  n_blocks     uint32    (n_imp+1)**2
  drop_thresh  double
  per block (n_blocks times, ordered by (n_up, n_dn)):
    n_up           int32
    n_dn           int32
    dim            uint64
    trace          double        block trace, always present
    stored         uint8         1 if the matrix follows
    basis_indices  uint32[dim]   canonical basis index of each row/column
    matrix         float64[dim, dim]  row-major, only if stored == 1

The RDM is block-diagonal in (N_up_imp, N_dn_imp); a dropped block is zero in
the dense RDM up to the drop threshold. The block traces give P(N_up, N_dn).
"""

import numpy as np

MAGIC = b"RDMBLOCK"


def read_rho_blocks(path):
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not a rho_blocks file")
        version = int(np.fromfile(f, "<u4", 1)[0])
        if version != 1:
            raise ValueError(f"{path}: unsupported version {version}")
        basis_size = int(np.fromfile(f, "<u8", 1)[0])
        n_imp = int(np.fromfile(f, "<i4", 1)[0])
        n_blocks = int(np.fromfile(f, "<u4", 1)[0])
        drop_thresh = float(np.fromfile(f, "<f8", 1)[0])

        blocks = []
        for _ in range(n_blocks):
            n_up = int(np.fromfile(f, "<i4", 1)[0])
            n_dn = int(np.fromfile(f, "<i4", 1)[0])
            dim = int(np.fromfile(f, "<u8", 1)[0])
            trace = float(np.fromfile(f, "<f8", 1)[0])
            stored = int(np.fromfile(f, "u1", 1)[0])
            indices = np.fromfile(f, "<u4", dim)
            if indices.size != dim:
                raise ValueError(f"{path}: truncated block index list")
            matrix = None
            if stored:
                matrix = np.fromfile(f, "<f8", dim * dim)
                if matrix.size != dim * dim:
                    raise ValueError(f"{path}: truncated block matrix")
                matrix = matrix.reshape(dim, dim)
            blocks.append(
                dict(
                    n_up=n_up,
                    n_dn=n_dn,
                    dim=dim,
                    trace=trace,
                    stored=bool(stored),
                    basis_indices=indices,
                    matrix=matrix,
                )
            )
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after last block")

    return dict(
        basis_size=basis_size,
        n_imp=n_imp,
        drop_thresh=drop_thresh,
        blocks=blocks,
    )


def blocks_to_dense(data):
    dense = np.zeros((data["basis_size"], data["basis_size"]))
    for blk in data["blocks"]:
        if blk["matrix"] is None:
            continue
        idx = blk["basis_indices"].astype(np.intp)
        dense[np.ix_(idx, idx)] = blk["matrix"]
    return dense


if __name__ == "__main__":
    import sys

    data = read_rho_blocks(sys.argv[1])
    dense = blocks_to_dense(data)
    print(f"basis_size = {data['basis_size']}, n_imp = {data['n_imp']}")
    for blk in data["blocks"]:
        print(
            f"N_up={blk['n_up']} N_dn={blk['n_dn']} dim={blk['dim']:5d} "
            f"trace={blk['trace']:.12e} stored={int(blk['stored'])}"
        )
    print(f"total trace = {np.trace(dense):.12f}")
