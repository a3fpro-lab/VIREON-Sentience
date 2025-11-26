"""
VIREON_SHELL_BLACKBOX record schema and parser.

Each record is exactly 256 bytes, with the layout:

[  0..3]  uint32   real_time_ms         // R: external clock
[  4..7]  uint32   subjective_step      // P: internal TRP step counter
[  8..11] float32  trp_dilation         // T = R × P modulation factor
[ 12..15] uint32   entropy_score        // compressed entropy / KL score
[ 16..23] uint64   identity_hash        // agent / config fingerprint
[ 24..27] uint32   event_class          // enum (OK, WARN, FAULT, ANOMALY, ...)
[ 28..31] uint32   event_severity       // scaled 0–100 or similar
[ 32..255] bytes   payload              // compressed sensors / state snapshot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, BinaryIO, Optional
import struct

RECORD_SIZE = 256

# Little-endian:
#   uint32, uint32, float32, uint32, uint64, uint32, uint32  -> 4+4+4+4+8+4+4 = 32 bytes
_HEADER_STRUCT = struct.Struct("<II f I Q II")


@dataclass
class BlackboxRecord:
    real_time_ms: int
    subjective_step: int
    trp_dilation: float
    entropy_score: int
    identity_hash: int
    event_class: int
    event_severity: int
    payload: bytes  # length = 224

    @property
    def T_effective(self) -> float:
        """Convenience: TRP-style effective time T = R × P scaling proxy."""
        return float(self.real_time_ms) * self.trp_dilation


def parse_record(raw: bytes) -> BlackboxRecord:
    """
    Parse a single 256-byte raw record into a BlackboxRecord.

    Raises ValueError if length is wrong.
    """
    if len(raw) != RECORD_SIZE:
        raise ValueError(f"Expected {RECORD_SIZE} bytes, got {len(raw)}")

    header = raw[: _HEADER_STRUCT.size]
    payload = raw[_HEADER_STRUCT.size :]

    (real_time_ms,
     subjective_step,
     trp_dilation,
     entropy_score,
     identity_hash,
     event_class,
     event_severity) = _HEADER_STRUCT.unpack(header)

    return BlackboxRecord(
        real_time_ms=real_time_ms,
        subjective_step=subjective_step,
        trp_dilation=trp_dilation,
        entropy_score=entropy_score,
        identity_hash=identity_hash,
        event_class=event_class,
        event_severity=event_severity,
        payload=payload,
    )


def iter_records_from_bytes(blob: bytes) -> Iterator[BlackboxRecord]:
    """
    Iterate over consecutive 256-byte records in a bytes object.
    Stops when there is not enough data for a full record.
    """
    n = len(blob)
    for offset in range(0, n - RECORD_SIZE + 1, RECORD_SIZE):
        chunk = blob[offset : offset + RECORD_SIZE]
        yield parse_record(chunk)


def iter_records_from_file(
    f: BinaryIO,
    limit: Optional[int] = None,
) -> Iterator[BlackboxRecord]:
    """
    Stream records from an open binary file (e.g., a QSPI dump).

    Args:
        f: file-like object opened in 'rb' mode.
        limit: optional maximum number of records to read.

    Yields:
        BlackboxRecord instances until EOF or limit is reached.
    """
    count = 0
    while True:
        raw = f.read(RECORD_SIZE)
        if not raw or len(raw) < RECORD_SIZE:
            break

        yield parse_record(raw)
        count += 1

        if limit is not None and count >= limit:
            break
