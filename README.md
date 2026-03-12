---
title: "Plair Technical Challenge"
author: Emmanuel Pauchard
date: Feb 17, 2026
geometry: margin=2cm
output: pdf_document
---
# Data Storage Challenge

## Context

Plair manufactures a viable particle counter (see [our
website](https://www.plair.ch/)). During calibration of our instrument, we
generate a lot of data from the machine sensors (mainly spectrometers and
scattering light sensors) which can amount to several gigabytes of data per
minute. We need to store this data to disk for post-processing and machine
learning.

## Objective

Design and implement a **data storage system** in Python that ingests a
continuous stream of numerical data from `stdin` and persists it to disk in a
robust, queryable format.

You are provided with:

| File | Description |
|---|---|
| `data_generator.py` | Streams random data packets to `stdout`. **Do not modify.** |
| `basic_storage.py` | Naïve reference implementation (pickle + append). |

Your task is to **replace `basic_storage.py`** with a production-quality storage
backend that satisfies the requirements below.

Your code should be a python module or package with instructions on how to
execute in combination with `data_generator.py`. You can reuse the same command
line interface as the example (last section).

Once ready, send me your code ideally by sharing a link to a github (or other)
project, or in a tar archive. I will then test your software on my laptop.

---

## Data Format

Each packet contains **N particles** (N varies randomly between 1 and 1 000).
The packet is a Python `dict` with three NumPy arrays:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `timestamps` | `(N,)` | `float64` | One timestamp per particle (epoch s) |
| `scattering` | `(N, 64, 16)` | `int32` | One 64×16 measurement per particle |
| `spectral` | `(N, 32, 16)` | `int32` | One 32×16 measurement per particle |

**Data model**: Each particle is one row in your storage. A particle has:

- 1 timestamp
- 1 scattering array (64×16)
- 1 spectral array (32×16)


Packets arrive as length-prefixed pickled blobs on `stdin` (4-byte big-endian
length header followed by the pickle payload). See `data_generator.py` for
details.

### Note:

To keep the generation side simple, all particles in a packet use the same
timestamp.

---

## Requirements

### Must-have

1. **Storage capacity**: the system must support up to **1 TB** of stored data,
   **50k particles written per second** and **200k particles read** per second.

1. **Embedded device**: the storage will run on an embedded device. Do not worry
   about CPU / RAM (consider our device is equivalent to a decent laptop (2Ghz+
   multi-core and 16Gb+ RAM) and runs a recent Linux distribution), but try to
   anticipate constraints from this context.

1. **Read access optimised by time range**: provide a read API or CLI that
   retrieves all rows whose `timestamps` fall within a given `[start, stop]`
   range efficiently.

1. **Don't reinvent the wheel**. Use as many off-the-shelf Python libraries as
   you need, while keeping in mind the industrial context.

1. **Python only**: the solution must be implemented in Python.

1. **LLM**: It is ok to use AI assistance. Use it wisely, or not at all: The
   goal for me is to evaluate if **your** style and skills, not ChatGPT's, will
   be a good match for our team. By the way, this challenge *has* been generated
   using a LLM: I wanted to keep it neutral to see your coding style. Be
   natural!

---

## How to Run the example

```bash
# tested with Python 3.12 and numpy 2.4.2

# Install dependencies (in a virtual environment)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy

# Generate data and pipe into your storage implementation
python data_generator.py --pps 100 --max-mb 200 \
    | python your_storage.py write <output_path>

# Generator options:
#   --pps N       Packets per second (0 = unlimited). Typical: 50-100.
#   --max-mb N    Stop after N MB sent (0 = unlimited).

# The reference (naïve) implementation: for comparison only
python data_generator.py --pps 150 --max-mb 500 \
    | python basic_storage.py --storage-file output.pkl write
    
First Data timestamps 2026-02-17T15:32:36.080619+00:00
Reached 527396279 bytes after 165 packets (85720 particles). Stopping.
Write bandwidth: 60.58 kParticles/s
Last Data timestamp 2026-02-17T15:32:37.442515+00:00
Wrote to storage 165 packets (527396279 bytes).


# Read data from storage (to adapt to the timestamps output in previous command)
python  basic_storage.py --storage-file output.pkl read \
   --start "2026-02-17T15:32:37" \
   --stop "2026-02-17T15:32:37.1"

Reading data between 2026-02-17 15:32:37+00:00 and 2026-02-17 15:32:37.100000+00:00
Found storage data from 2026-02-17 15:04:28.783457+00:00 to 2026-02-17 15:32:37.442515+00:00
Found 7280 particles.
Read bandwidth 2.91 kParticles/s.
    
```

---

## Guidelines

| Guideline | Detail |
|---|---|
| **Time budget** | 2–4 hours maximum. This is a technical screen, not a marathon. |
| **Scope** | Focus on the storage layer. No need for a GUI, REST API, or database server. A CLI or simple Python API is sufficient. |
| **Documentation** | Include a short write-up (in code comments or a separate file) explaining your design choices, trade-offs, and what you would improve given more time. |
| **Focus on storage, not on parsing input data** | To parse input you can reuse `get_packet_from_stream` from the example.|

---

Do not hurt yourself with this task. The goal is to have a quick view on your
skills and also a basis for discussion for the next interview. My focus will be
more on architecture and software good practices than performances. I will even
accept uncomplete submissions if you document your reasoning well enough.

Good luck!
