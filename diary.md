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

## 1-hour single-GPU run (2026-03-23)

Ran the same 11L XSA+EMA+TTT config on 1× A40 (46GB) for 1 hour instead of 10 min on 8 GPUs.

### Scaled parameters

The LR warmdown is time-based (`warmdown_ms = WARMDOWN_ITERS * step_ms`), so with ~3.3s/step on 1 GPU (vs ~85ms on 8 GPUs), WARMDOWN_ITERS and MUON_MOMENTUM_WARMUP_STEPS had to be scaled down to avoid spending the entire run in warmdown.

| Param | 8-GPU (10 min) | 1-GPU (1 hour) | Rationale |
|---|---|---|---|
| `MAX_WALLCLOCK_SECONDS` | 600 | 3600 | 1 hour |
| `WARMDOWN_ITERS` | 3000 | 460 | Same ~42% warmdown ratio for ~1090 steps |
| `MUON_MOMENTUM_WARMUP_STEPS` | 1500 | 230 | Same ~21% ratio |
| Everything else | same | same | |

### Command

```bash
env \
  NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
  EMA_ENABLED=1 EMA_DECAY=0.997 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
  MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=230 \
  WARMDOWN_ITERS=460 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=3600 EVAL_STRIDE=64 SEED=42 \
  TTT_ENABLED=1 TTT_ONLINE=1 TTT_DUPLICATE=1 \
  TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
  RUN_ID=1hr_s42 \
  python3 train_gpt.py 2>&1 | tee logs/1hr_s42.txt
```

### Output files

| File | Description |
|---|---|
| `logs/1hr_s42.txt` | Full training log |
| `final_model.pt` | Full-precision saved model (~106MB) |
| `final_model.int6.ptz` | Int6+zstd quantized model (~15MB) |

### Results

- 1089 steps in 3601s (~3.31s/step), `grad_accum_steps=8`
- Val BPB at stop: **1.2840** (step 1089, full precision, no sliding window)
- Int6 quantized BPB: **1.3995** (no sliding window, no TTT)
- Sliding window + online TTT duplicate: killed (was going to take hours on 1 GPU)

### Quick eval on 2M token subset (eval_quick.py)

Ran all 3 TTT modes on the saved int6 model (`final_model.int6.ptz`) with 2M val tokens (3.2% of full set):

| Mode | Val loss | BPB | Time |
|---|---|---|---|
| No TTT (sliding window) | 2.3219 | **1.3914** | 124s |
| TTT online in-place (no dup) | 2.3089 | **1.3836** | 33 min |
| TTT online duplicate | *(killed at 70%)* | ~1.87 (diverging) | — |

TTT in-place gained **0.0078 BPB**. TTT duplicate performed much worse — the model was trained with 11 layers but duplication creates a 20-effective-layer architecture at eval, causing distribution shift that short documents + frequent resets can't overcome.

### Notes

- ~1089 steps vs ~7068 on 8 GPUs — model sees ~6.5× less data, so expect worse BPB than the 10-min 8-GPU run (1.1254).

## Tied-duplicate training experiment (2026-03-23)

### Motivation

TTT duplicate performed terribly on the non-dup model because the model was trained with 11 effective layers but duplication creates 20. To fix this architecture mismatch, we train with **tied duplicate blocks**: blocks 2-10 each run twice in the forward pass with shared weights, giving 20 effective layers during training. At TTT eval time, the duplicates are deep-copied (untied) so TTT can adapt them independently.

### Architecture (all modes use 20 effective layers)

Forward pass: `block 0 → block 1 → block 2 → block 2 → block 3 → block 3 → ... → block 10 → block 10`

Each block runs, then immediately runs again before moving to the next.

| Mode | Weights | Extra params |
|---|---|---|
| No TTT (tied dup) | blocks 2-10 shared across both passes | 0 (26.8M) |
| TTT in-place | same, TTT adapts the shared weights | 0 |
| TTT duplicate | second pass uses untied deep-copies, TTT adapts all | ~21M free at eval time |

### Implementation

Added `TIED_DUP_FROM` env var to `train_gpt.py`. Sets `model.tied_dup_from`, and in both `forward()` and `forward_logits()`:
```python
elif dup_from >= 0 and i >= dup_from:
    x = self.blocks[i](x, x0)  # run block again
```
At TTT eval time, `ttt_setup` deep-copies as before — no conflict since eval model has `tied_dup_from=-1`.

### Training

```bash
env \
  NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
  EMA_ENABLED=1 EMA_DECAY=0.997 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
  MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=130 \
  WARMDOWN_ITERS=260 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=3600 EVAL_STRIDE=64 SEED=42 \
  TIED_DUP_FROM=2 \
  TTT_ENABLED=1 TTT_ONLINE=1 TTT_DUPLICATE=1 \
  TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
  RUN_ID=1hr_tieddup_s42 \
  python3 train_gpt.py 2>&1 | tee logs/1hr_tieddup_s42.txt
```

- ~5.86s/step (1.78x slower than non-dup due to doubled forward/backward for 9 blocks)
- 613 steps in 3606s, `grad_accum_steps=8`
- `WARMDOWN_ITERS=260`, `MUON_MOMENTUM_WARMUP_STEPS=130` (scaled for fewer steps)

### Results (100k token subset, eval_quick.py)

| Mode | BPB | TTT gain |
|---|---|---|
| No TTT (tied dup) | **1.6297** | — |
| TTT in-place | **1.5870** | -0.043 |
| TTT duplicate | **1.5945** | -0.035 |

Comparison with non-dup model:

| | Non-dup | Tied-dup |
|---|---|---|
| Base BPB (no TTT) | 1.3914 | 1.6297 |
| TTT in-place gain | -0.008 | -0.043 |
| TTT duplicate gain | N/A (diverged) | -0.035 |
| Best BPB | **1.3836** | 1.5870 |

### Conclusions

- Tied-dup training **fixes the architecture mismatch**: TTT duplicate no longer diverges.
- TTT gains are **5x larger** (0.043 vs 0.008 for in-place).
- But the base model is much worse (1.63 vs 1.39) because 1.78x slower steps → 613 vs 1089 steps in 1 hour.
- Net: TTT gains don't compensate for the training deficit on a fixed wall-clock budget.
- TTT in-place beats TTT duplicate — the weight tying acts as a regularizer that helps with short documents + frequent resets.
- The "free parameters" from duplication (~21M) aren't valuable enough given limited TTT adaptation per document.

## Untied-duplicate experiment (2026-03-24)

### Motivation

Test whether training with 20 layers (no weight tying) performs better than tied-duplicate training, given the same 1-hour wall-clock budget. This is the standard non-dup architecture but scaled to 20 layers — no `TIED_DUP_FROM`, duplication only happens at TTT via deep-copy.

### Training

```bash
env \
  NUM_LAYERS=20 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
  EMA_ENABLED=1 EMA_DECAY=0.997 ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 QAT_THRESHOLD=0.1 \
  MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
  MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=115 \
  WARMDOWN_ITERS=230 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=3600 EVAL_STRIDE=64 SEED=42 \
  TTT_ENABLED=1 TTT_ONLINE=1 TTT_DUPLICATE=1 \
  TTT_LR=0.002 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=2 \
  RUN_ID=1hr_untieddup_s42 \
  python3 train_gpt.py 2>&1 | tee logs/1hr_untieddup_s42.txt
```

- ~5.90s/step (similar to tied-dup since 20 real layers ≈ 11 tied-dup layers in compute)
- 611 steps in 3604s, `grad_accum_steps=8`
- Model size: 190MB full / 22MB int6+zstd (vs 106MB/12MB for tied-dup — nearly 2x due to untied weights)

### Training loss comparison vs tied-duplicate

| Step | Untied (20L) | Tied-dup (11L×2) | Delta |
|---|---|---|---|
| 200 | 2.7407 | 2.7528 | -0.012 |
| 400 | 2.5070 | 2.5114 | -0.004 |
| 600 | 2.2917 | 2.3091 | -0.017 |
| **Final val** | **2.3100 (1.3681 BPB)** | **2.3285 (1.3790 BPB)** | **-0.011 BPB** |

### Preliminary conclusions

- Untied 20L consistently beats tied-dup 11L in training loss — having independent weights helps even at the same effective depth.
- Final val BPB 1.3681 vs 1.3790 — a meaningful 0.011 gap.
- But the model is ~2x larger (22MB vs 12MB int6+zstd), which may matter for submission size limits.
- TTT sliding window eval not yet available — need to compare final BPB with TTT to see if the gains hold.
