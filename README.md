# SEUCAM — OFDM Acoustic Communication System

Software-defined acoustic modem: transmit files through sound waves using OFDM modulation and LDPC error-correcting codes, implemented in Python.

**Pipeline:** file → LDPC encode → QPSK map → OFDM modulate → audio (speaker) → air/water channel → microphone → synchronize → OFDM demodulate → LDPC decode → recovered file

## Repository Structure

```
SEUCAM/
├── Channel_Measurement/   ← Main project (active development)
│   ├── module/            Core Python package
│   │   ├── cfg/           Typed dataclass configuration
│   │   ├── emitter/       Transmit waveform builder
│   │   ├── receiver/      Receive & decode pipeline
│   │   └── utils/         Modulation, coding, channel estimation, …
│   ├── demo/              Self-contained demo (CPU-only, no arguments)
│   ├── assets/            Pre-computed pilots & demo signal
│   └── output/            Decoded output files
│
├── legacy/                ← Archived reference implementations
│   └── ...                Older monolithic scripts (N=4096, etc.)
│
│
└── Week1 Challenge/       ← Might help, data missed, only source code here
```

## Quick Start

```bash
# Activate virtual environment
e.g. .venv\Scripts\activate

# Run the demo (no GPU required)
cd Channel_Measurement
python demo/decode_demo.py    # signal → text
python demo/encode_demo.py    # text → signal
```

The demo ships with a pre-encoded signal (`assets/signals/demo_signal.npy`) — decode it, then re-encode to verify.

## Environment Setup

| Requirement | Version |
|-------------|---------|
| Python | 3.12+ |
| numpy | 2.x |
| scipy | 1.x |
| matplotlib | 3.x |
| sounddevice | 0.5+ |
| soundfile | 0.13+ |

GPU acceleration (optional, for faster LDPC decoding):

| Package | Purpose | Version |
|---------|---------|---------|
| torch + CUDA | GPU tensor ops | 2.5.1+cu121 |
| dgl | Graph-based LDPC sum-product decoding | 2.5 |

for dgl, we provide a precompiled python package for python3.12 and x86 system cause the official site no longer provides precompiled Windows binaries; building from source yourself is fairly complex, the .whl file is available here: https://pan.baidu.com/s/1NEdpXVYd05QGgtMdnnwGBw code: n6y2 


for any other question(especially code bug), issue in the github repo or contact me via 3136992610@qq.com

