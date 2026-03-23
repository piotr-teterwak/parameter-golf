# TTT (Test-Time Training) Diary

## What we built

Online TTT with optional layer duplication, no data leakage, and document boundary resets.

Based on PR #338, modified to address:
- **Issue #402**: score each window BEFORE adapting (no future token leakage)
- **PR #77 insight**: reset params between documents (no cross-document leakage)

## 4 TTT modes

| Mode | Env vars | Leaks? | Description |
|---|---|---|---|
| No TTT | `TTT_ENABLED=0` | - | Baseline |
| Batch TTT | `TTT_ENABLED=1 TTT_ONLINE=0` | yes | Original: trains on all eval tokens, then scores (issue #402) |
| Online in-place | `TTT_ENABLED=1 TTT_ONLINE=1 TTT_DUPLICATE=0` | no | Score-then-adapt per window, modifies original layers |
| Online duplicate | `TTT_ENABLED=1 TTT_ONLINE=1 TTT_DUPLICATE=1` | no | Same, but duplicates non-frozen blocks (model grows at test time) |

Defaults: `TTT_ONLINE=1 TTT_DUPLICATE=1`, so `TTT_ENABLED=1` alone gives the correct duplicate mode.

## How to run

### Quick eval (200k token subset, ~4 min per mode)

```bash
python3 eval_ttt_modes.py
```

Set `VAL_TOKENS=500000` to use more data:
```bash
VAL_TOKENS=500000 python3 eval_ttt_modes.py
```

### Full training + eval (single seed)

Shared base config:
```bash
BASE="NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 SEED=42"
```

No TTT:
```bash
env $BASE TTT_ENABLED=0 RUN_ID=no_ttt \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Online in-place:
```bash
env $BASE TTT_ENABLED=1 TTT_ONLINE=1 TTT_DUPLICATE=0 \
  TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 RUN_ID=ttt_online \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Online duplicate:
```bash
env $BASE TTT_ENABLED=1 TTT_ONLINE=1 TTT_DUPLICATE=1 \
  TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 RUN_ID=ttt_dup \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## TTT env vars

| Var | Default | Description |
|---|---|---|
| `TTT_ENABLED` | `0` | Enable test-time training |
| `TTT_ONLINE` | `1` | Online (correct) vs batch (leaks) |
| `TTT_DUPLICATE` | `1` | Duplicate non-frozen blocks (grow model) |
| `TTT_LR` | `0.002` | SGD learning rate for TTT |
| `TTT_MOMENTUM` | `0.9` | SGD momentum |
| `TTT_EPOCHS` | `3` | Epochs (only used in batch mode) |
| `TTT_BATCH_SEQS` | `32` | Batch size (only used in batch mode) |
| `TTT_FREEZE_BLOCKS` | `2` | Number of early blocks to freeze |

## Results (200k token subset, 1 GPU, poorly trained model)

| Mode | BPB | Time |
|---|---|---|
| No TTT | 3.7687 | 18s |
| Batch TTT (leaks) | 3.7372 | 19s |
| Online in-place (no reset) | 3.3485 | 204s |
| Online duplicate (no reset) | 3.3460 | 259s |
| Online in-place (with reset) | 3.6157 | 205s |
| Online duplicate (with reset) | 3.6122 | 255s |

Resets reduce TTT benefit because docs are short (~1,240 tokens avg, ~19 windows per doc at stride=64).

## Key design decisions

1. **Score before adapt**: forward pass produces logits, we record NLL for scored tokens, THEN backward + optimizer step. Same forward pass serves both.

2. **Document boundary resets**: when any scored token is a boundary token (BOS/control), reset all trainable params to initial values and clear optimizer state.

3. **Layer duplication**: `ttt_setup` deep-copies blocks >= `ttt_freeze_blocks`. Both originals and duplicates are trained. The duplicate runs sequentially after the original in the forward pass.

4. **No batching in online mode**: windows processed one at a time to ensure each window benefits from the previous adaptation. Batching would be faster but changes results.

## Ideas to explore

- **Smaller stride** for more TTT updates per document (stride isn't fixed by competition rules)
- **Higher LR** to adapt faster within short documents
- **LoRA instead of full-weight** TTT — cheaper resets, fewer params to adapt
- **Freeze embeddings** in duplicate mode to enable detach optimization (~12% backward speedup)
- **Batching with small batch size** (e.g. 4-8) as speed/quality tradeoff
