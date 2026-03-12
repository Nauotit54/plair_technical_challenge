#!/usr/bin/env python3
"""
Basic storage implementation for the storage challenge.

Reads length-prefixed pickled packets from stdin and appends them
(re-pickled) to a single output file.

Usage:
    python data_generator.py | python basic_storage.py output.pkl
"""
import argparse
import pickle
import struct
import sys
import time
from datetime import datetime, timezone

import numpy as np


def read_exact(stream, n: int) -> bytes:
    """Read exactly *n* bytes from *stream*, or return empty on EOF."""
    data = b""
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return b""
        data += chunk
    return data


def get_packet_from_stream() -> list[dict[str, np.array], int]:
    """Read one packet from data_generator, or return empty on EOF.

    Return a tuple (decoded data , raw data length)
    """
    stdin = sys.stdin.buffer

    # Read the 4-byte length prefix
    header = read_exact(stdin, 4)
    if not header:
        return b"", 0

    (length,) = struct.unpack(">I", header)

    # Read the payload
    payload = read_exact(stdin, length)
    if not payload:
        raise ValueError("Empty payload")

    # Deserialize
    return pickle.loads(payload), len(payload) + 4


def write_to_storage(data: bytes, output_path: str):
    """Add data to storage.

    This is the naive implementation using pickle, you must do better than this!
    """
    with open(output_path, "ab") as fout:
        pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)
        fout.flush()


def cmd_write(args):
    """Write loop"""

    output_path = args.storage_file
    total_bytes_received = 0
    packets_written = 0

    while True:
        packet, raw_data_length = get_packet_from_stream()
        if packet == b"":
            break
        write_to_storage(packet, output_path)
        packets_written += 1
        total_bytes_received += raw_data_length

    print(
        f"Wrote to storage {packets_written} packets "
        f"({total_bytes_received} bytes).",
        file=sys.stderr,
    )


def cmd_read(args):
    """Read back data from storage"""
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    stop = datetime.fromisoformat(args.stop).replace(tzinfo=timezone.utc)
    start_ts = start.timestamp()
    stop_ts = stop.timestamp()
    print(f"Reading data between {start} and {stop}")

    t00 = time.monotonic()
    ts = []
    scat = []
    spec = []
    with open(args.storage_file, "rb") as fin:
        while True:
            try:
                data = pickle.load(fin)
                ts += [data["timestamps"]]
                scat += [data["scattering"]]
                spec += [data["spectral"]]
            except Exception:
                break

    ts = np.hstack(ts)
    print(
        f"Found storage data from {datetime.fromtimestamp(ts[0], tz=timezone.utc)} to {datetime.fromtimestamp(ts[-1], tz=timezone.utc)}"
    )
    mask = np.where((ts >= start_ts) & (ts <= stop_ts))
    data = {
        "timestamps": ts,
        "scattering": np.vstack(scat)[mask],
        "spectral": np.vstack(spec)[mask],
    }
    print(f"Found {len(mask[0])} particles.")
    print(
        f"Read bandwidth {len(mask[0])/1024/(time.monotonic() - t00):.2f} kParticles/s."
    )
    return data


def main():

    parser = argparse.ArgumentParser(
        description="Example of data storage system",
    )
    parser.add_argument(
        "--storage-file",
        type=str,
        default=None,
        help="Pickle storage file location",
    )

    sub = parser.add_subparsers(dest="command")

    # -- write --
    p_w = sub.add_parser("write", help="Ingest packets from stdin")

    # -- read --
    p_r = sub.add_parser("read", help="Query by time range")
    p_r.add_argument(
        "--start",
        required=True,
        help="Start timestamp (ISO 8601 format, inclusive)",
    )
    p_r.add_argument(
        "--stop",
        required=True,
        help="Stop timestamp (ISO 8601 format, inclusive)",
    )

    args = parser.parse_args()

    if args.command == "write":
        cmd_write(args)
    elif args.command == "read":
        cmd_read(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
