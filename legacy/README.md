# Legacy Code Archive

Archived from the old codebase. These files are kept for reference only.

## Contents

| File | Description | Why Archived |
|------|-------------|--------------|
| `receiver_ldpc_N4096.py` | N=4096 OFDM LDPC receiver | Retains N=4096 mode for compatibility with old recordings |
| `receiver_ldpc_part-valid_comb.py` | Advanced comb-pilot receiver with bad-block source analysis | Most mature old receiver; contains diagnostic features not yet ported |
| `receiver_ldpc_pv_comb_head.py` | Comb-pilot receiver with 64-bit header decoding | Two-stage decoding variant |
| `emitter_ldpc_comb.py` | LDPC emitter with comb pilots | Companion emitter for old receivers |
| `emitter_ldpc_N8192.py` | N=8192 emitter (pre-guard-band) | Earlier variant of the emitter |
