# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""cr — reference implementation of the Computation Receipt (CR) format, v0.1.

The spec is the authority (../spec/CR-v0.1.md); this package exists so a third party
can check the published conformance vectors and verify receipts without reading our
other code. Deliberately dependency-light: stdlib + numpy.
"""
from .receipt import (  # noqa: F401
    ACCEPT,
    MALFORMED,
    REJECT,
    UNVERIFIABLE,
    Receipt,
    ReceiptError,
    build_chunk_receipt,
    build_closing_receipt,
    build_receipt,
    build_sampled_receipt,
    check_wellformed,
    conformance_vectors,
    sample_indices_of,
    verify,
    verify_chain,
    verify_sampled,
)
