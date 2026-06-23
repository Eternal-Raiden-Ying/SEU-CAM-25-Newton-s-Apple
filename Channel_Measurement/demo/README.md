# Demo

Two-step encode/decode demo using the OFDM+LDPC pipeline.

## Quick Start

```bash
# 1. Encode a text file into a signal
cd Channel_Measurement/demo
python encode_demo.py ../data/demo.txt

# 2. Decode the signal back to text
python decode_demo.py
```

Decoded output appears in `../output/ldpc/demo_decoded.txt`.

## CPU-only

Both scripts default to CPU LDPC decoding (no GPU required).
Run time is ~10 seconds total for a ~1KB text file.

## Custom Input

```bash
python encode_demo.py /path/to/your/file.txt
python decode_demo.py /path/to/your/signal.npy
```

Decode picks up the most recent signal in `assets/signals/` if no path is given.
