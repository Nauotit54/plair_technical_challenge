#!/usr/bin/env python3
"""
HDF5 storage implementation of the storage challenge.

This file reuse the basic_storage.py CLI and can be used exactly the same way to write and read data.

Why did I first chose HDF5 storage ?
- Easy to use with Numpy data structures
- Can store a large amount of data
- Can use compression (I did not used it here as it was slowing down the writing)
- Queryable data and fast reading

What I discovered during the implementation:
- Writing data speed is dropping as soon as we are above 2Gb --> this is a big issue for Tb applications
- I tried different solutions to fix that, I measured the execution time in the append_packets method and
  it looks like sometimes the last part of the method is extremly slow (about 10 secondes instead of 0.5 - 0.7).
  I did not really found why, what I found on the web is that is could be due to chuncking, but I suspect an
  issue with file writing.
- I also try writing each 100 000 particles in different files, it helped maintaing a consistent write bandwidth
  but the speed was not good, and the reading was more complex and slower.

I stopped at this point to match the deadline, but with more time, I would have explored other solutions, 
trying using a database for example.
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np
import h5py

from basic_storage import get_packet_from_stream

BUFFER_MAX_PARTICLES = 100000


def init_storage(path):
    """
    Initialize the h5py storage by creating three datasets to store timestamps, scattering and spectral.
    Did not use a compressor to be as fast as possible.
    """
    storage_file = h5py.File(path, "w", libver="latest")

    storage_file.create_dataset(
        name="timestamps",
        shape=(0,),
        maxshape=(None,),
        dtype="float64",
        chunks=(65536, ),
    )

    storage_file.create_dataset(
        name="scattering",
        shape=(0, 64, 16),
        maxshape=(None, 64, 16),
        dtype="int32",
        chunks=(1024, 64, 16, ),
    )

    storage_file.create_dataset(
        name="spectral",
        shape=(0, 32, 16),
        maxshape=(None, 32, 16),
        dtype="int32",
        chunks=(2048, 32, 16, ),
    )

    return storage_file


def append_packets(storage_file, timestamps, scattering, spectral):
    """
    Append timestamps, scattering and spectral data inside hdf5 datasets
    """
    timestamps_dataset = storage_file["timestamps"]
    scattering_dataset = storage_file["scattering"]
    spectral_dataset = storage_file["spectral"]

    old_length = timestamps_dataset.shape[0]
    new_length = len(timestamps)
    total_length = old_length + new_length

    timestamps_dataset.resize((total_length,))
    scattering_dataset.resize((total_length, 64, 16))
    spectral_dataset.resize((total_length, 32, 16))

    # This part of the process is sometimes unexpectedly slow...
    # I have not been able to find why
    timestamps_dataset[old_length:total_length] = timestamps
    scattering_dataset[old_length:total_length] = scattering
    spectral_dataset[old_length:total_length] = spectral


def cmd_write(args):
    """Write loop"""

    output_path = args.storage_file
    total_bytes_received = 0
    packets_written = 0

    storage_file = init_storage(output_path)

    timestamps_buffer = []
    scattering_buffer = []
    spectral_buffer = []

    buffered_particles = 0

    while True:
        packet, raw_data_length = get_packet_from_stream()
        if packet == b"":
            break

        timestamps_buffer.append(packet["timestamps"])
        scattering_buffer.append(packet["scattering"])
        spectral_buffer.append(packet["spectral"])

        buffered_particles += len(packet["timestamps"])

        packets_written += 1
        total_bytes_received += raw_data_length

        # Emptying the buffer to avoid overflows
        if buffered_particles >= BUFFER_MAX_PARTICLES:

            append_packets(
                storage_file,
                np.concatenate(timestamps_buffer),
                np.concatenate(scattering_buffer),
                np.concatenate(spectral_buffer)
            )

            timestamps_buffer.clear()
            scattering_buffer.clear()
            spectral_buffer.clear()

            buffered_particles = 0

    if timestamps_buffer:

        append_packets(
            storage_file,
            np.concatenate(timestamps_buffer),
            np.concatenate(scattering_buffer),
            np.concatenate(spectral_buffer),
        )

    storage_file.close()

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

    start_time = time.monotonic()

    with h5py.File(args.storage_file, "r") as storage_file:

        timestamps = storage_file["timestamps"][:]

        print(
            f"Found storage data from "
            f"{datetime.fromtimestamp(timestamps[0], tz=timezone.utc)} "
            f"to {datetime.fromtimestamp(timestamps[-1], tz=timezone.utc)}"
        )

        # Find start and stop timestamps
        i0 = np.searchsorted(timestamps, start_ts)
        i1 = np.searchsorted(timestamps, stop_ts)

        packets = {
            "timestamps": timestamps[i0:i1],
            "scattering": storage_file["scattering"][i0:i1],
            "spectral": storage_file["spectral"][i0:i1],
        }

    print(f"Found {len(packets['timestamps'])} particles.")
    print(
        f"Read bandwidth {len(packets['timestamps'])/1024/(time.monotonic() - start_time):.2f} kParticles/s."
    )

    return packets


def main():

    parser = argparse.ArgumentParser(
        description="HDF5 data storage system",
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
