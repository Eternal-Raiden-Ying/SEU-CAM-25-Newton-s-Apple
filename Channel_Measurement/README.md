# Channel_Measurement

Main project: OFDM acoustic communication with LDPC coding.

## Directory Map

```
Channel_Measurement/
├── emitter.py              CLI entry — encode file → transmit waveform
├── receiver.py             CLI entry — recorded WAV → decoded file
├── record_signal.py        Audio recording utility
│
├── demo/                   Self-contained demo (no GPU, no arguments)
│   ├── decode_demo.py      signal → text
│   ├── encode_demo.py      text → signal
│   └── README.md
│
├── module/                 Core package
│   ├── cfg/config.py       Typed dataclasses (OFDM, Chirp, LDPC, …)
│   ├── emitter/emitter.py  Build complete transmit waveform
│   ├── receiver/           Receiver pipeline (stable + OOP dev variant)
│   └── utils/              Utility functions (see utils/README.md)
│
├── assets/                 Pre-computed binary data
│   ├── pilots/             Frequency-domain pilot references
│   └── signals/            Demo signal (demo_signal.npy)
│
├── data/                   Input data (ignored by git)
└── output/                 Decoded outputs (ignored by git)
    └── ldpc/               LDPC decoding results
```

## Demo

Two scripts, no arguments, CPU-only:

```bash
python demo/decode_demo.py    # assets/signals/demo_signal.npy → output/ldpc/demo_decoded.txt
python demo/encode_demo.py    # output/ldpc/demo_decoded.txt → assets/signals/demo_reencoded.npy
```

The demo signal is ~3.1 seconds at 48 kHz, encoding 1020 bytes with:
- 0.5 s chirp preamble (10–24000 Hz linear sweep)
- 8 pilot OFDM symbols for channel estimation
- LDPC 802.11n, rate 1/2, Z=81, max_iter=5 (CPU)
- No scrambler, no comb pilots

## Configuration

All parameters live in `module/cfg/config.py` as composable dataclasses:

```python
from module.cfg.config import EmitterConfig, ReceiverConfig

# Emitter
cfg = EmitterConfig(pilot_mode="standard")
cfg.ldpc.device = "cpu"
cfg.ldpc.max_iter = 5

# Receiver
cfg = ReceiverConfig()
cfg.ldpc.device = "gpu"     # or "cpu"
cfg.ldpc.batch = 512        # parallel codeword decoding
```

Key dataclasses: `OFDMConfig`, `ChirpConfig`, `LDPCConfig`, `ScramblerConfig`, `EmitterConfig`, `ReceiverConfig`.

## Pipeline

```
[Emitter]
  file → bits → [header 64b] → [scrambler?] → LDPC encode
  → QPSK → OFDM modulate → [chirp | pilots | data | chirp]
  → .wav / .npy

[Receiver]
  .wav → chirp sync → CP strip → FFT → channel estimate (pilots)
  → ZF equalize → SFO/CFO drift track → PLL → QPSK LLR
  → LDPC decode → [descrambler?] → strip header → bytes → file
```

## Dependencies

See [root README](../README.md#environment-setup). Core: `numpy scipy matplotlib sounddevice`. GPU: `torch dgl`.
