#!/usr/bin/env python3
"""
Random data generator for the storage challenge.

Generates packets of numpy array data and streams them to stdout.

Each packet is a pickled dictionary containing:
  - "timestamps":  np.ndarray shape (N,),       dtype int64
                   (epoch seconds, monotonically increasing)
  - "scattering":  np.ndarray shape (N, 64, 16), dtype int32
  - "spectral":    np.ndarray shape (N, 32, 16), dtype int32

N varies randomly between 1 and 1000 for each packet.

Usage:
    # Default: as fast as possible, unlimited
    python data_generator.py

    # Rate-limited to 50 packets/s, stop after 100 MB sent
    python data_generator.py --pps 50 --max-mb 100

    # Pipe into storage
    python data_generator.py --pps 100 --max-mb 50 | \\
        python parquet_storage.py ./data
"""

import argparse
import pickle
import struct
import sys
import time
from datetime import datetime, timezone

import numpy as np


def generate_packet() -> dict:
    """Generate a single data packet with random N."""
    n = np.random.randint(1, 1001)

    timestamps = np.ones(n, dtype="float64") * datetime.now(tz=timezone.utc).timestamp()
    return {
        "timestamps": timestamps,
        "scattering": np.random.randint(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            size=(n, 64, 16),
            dtype=np.int32,
        ),
        "spectral": np.random.randint(
            low=np.iinfo(np.int32).min,
            high=np.iinfo(np.int32).max,
            size=(n, 32, 16),
            dtype=np.int32,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stream random data packets to stdout",
    )
    parser.add_argument(
        "--pps",
        type=float,
        default=0,
        help=("Packets per second (0 = unlimited). " "Typical: 50-100."),
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=0,
        help=("Stop after this many MB have been sent " "(0 = unlimited)."),
    )
    args = parser.parse_args()

    stdout = sys.stdout.buffer
    total_bytes = 0
    max_bytes = int(args.max_mb * 1024 * 1024) if args.max_mb > 0 else 0
    pps = args.pps
    packet_count = 0
    particle_count = 0
    t00 = time.monotonic()

    try:
        while True:
            t0 = time.monotonic()

            packet = generate_packet()
            payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
            header = struct.pack(">I", len(payload))
            stdout.write(header)
            stdout.write(payload)
            stdout.flush()

            total_bytes += len(header) + len(payload)
            packet_count += 1
            particle_count += len(packet["timestamps"])

            if packet_count == 1:
                print(
                    f"First Data timestamps {datetime.fromtimestamp(packet['timestamps'][0], tz=timezone.utc).isoformat()}",
                    file=sys.stderr,
                )

            # Check max data size
            if max_bytes > 0 and total_bytes >= max_bytes:
                print(
                    f"Reached {total_bytes}"
                    f" bytes after {packet_count} packets ({particle_count} particles). "
                    f"Stopping.",
                    file=sys.stderr,
                )
                print(
                    f"Write bandwidth: {particle_count/1024/(time.monotonic() - t00):.2f} kParticles/s",
                    file=sys.stderr,
                )
                print(
                    f"Last Data timestamp {datetime.fromtimestamp(packet['timestamps'][-1], tz=timezone.utc).isoformat()}",
                    file=sys.stderr,
                )

                break

            # Rate limiting
            if pps > 0:
                elapsed = time.monotonic() - t0
                sleep_time = (1.0 / pps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except (BrokenPipeError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
