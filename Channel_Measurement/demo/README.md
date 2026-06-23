# Demo

No arguments, no GPU required. Two steps:

```bash
python decode_demo.py    # signal -> text
python encode_demo.py    # text -> signal
```

Decode reads `assets/signals/demo_signal.npy` and writes to `output/ldpc/demo_decoded.txt`.
Encode reads that txt and writes back to `assets/signals/demo_reencoded.npy`.
