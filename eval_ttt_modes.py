"""Eval-only script: load saved int6 model, run 4 TTT modes on a subset of val data."""
import copy
import io
import math
import os
import time

os.environ.setdefault("NUM_LAYERS", "11")
os.environ.setdefault("BIGRAM_VOCAB_SIZE", "2048")
os.environ.setdefault("XSA_LAST_N", "4")
os.environ.setdefault("ROPE_DIMS", "16")
os.environ.setdefault("LN_SCALE", "1")
os.environ.setdefault("EVAL_STRIDE", "64")
os.environ.setdefault("TORCH_COMPILE", "0")  # skip compile for faster startup

import torch
import torch.nn.functional as F
import zstandard

# Import everything from the training script
from train_gpt import (
    GPT, CastedLinear, Hyperparameters,
    build_sentencepiece_luts, dequantize_mixed_int6,
    eval_val_sliding, eval_val_sliding_with_ttt,
    load_data_shard, restore_low_dim_params_to_fp32,
    ttt_adapt, ttt_setup,
)

VAL_TOKENS_LIMIT = int(os.environ.get("VAL_TOKENS", 200_000))  # small subset

def build_eval_model(args, deq_state, device):
    """Create a fresh eval model from dequantized state dict."""
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(deq_state, strict=True)
    return model

def log(msg):
    print(msg, flush=True)

def main():
    args = Hyperparameters()
    device = torch.device("cuda")
    stride = args.eval_stride
    seq_len = args.eval_seq_len or args.train_seq_len

    # Load val data (subset)
    import glob as glob_mod
    from pathlib import Path
    val_files = sorted(glob_mod.glob(args.val_files))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    total = val_tokens.numel()
    val_tokens = val_tokens[:min(VAL_TOKENS_LIMIT + 1, total)]
    log(f"val_tokens: {val_tokens.numel()-1} (of {total-1} total)")

    # Build LUTs
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device,
    )

    # Load quantized model
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob)),
        map_location="cpu",
    )
    # Build a dummy sd_cpu for dequantization (need original shapes)
    dummy_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    )
    sd_cpu = {k: v.clone() for k, v in dummy_model.state_dict().items()}
    del dummy_model
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)

    results = {}

    # ---- MODE 1: No TTT ----
    log("\n=== MODE 1: No TTT ===")
    model = build_eval_model(args, deq_state, device)
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding(
        args, model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=seq_len,
    )
    elapsed = time.perf_counter() - t0
    log(f"  sliding_bpb={bpb:.6f}  loss={loss:.6f}  time={elapsed:.1f}s")
    results["no_ttt"] = bpb
    del model; torch.cuda.empty_cache()

    # ---- MODE 2: Original TTT (in-place, leaks) ----
    log("\n=== MODE 2: Original TTT (in-place, leaks) ===")
    model = build_eval_model(args, deq_state, device)
    t0 = time.perf_counter()
    ttt_adapt(args, model, device, val_tokens, log_fn=log)
    loss, bpb = eval_val_sliding(
        args, model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=seq_len,
    )
    elapsed = time.perf_counter() - t0
    log(f"  sliding_bpb={bpb:.6f}  loss={loss:.6f}  time={elapsed:.1f}s")
    results["ttt_inplace_leak"] = bpb
    del model; torch.cuda.empty_cache()

    # ---- MODE 3: Online TTT in-place (no leak) ----
    log("\n=== MODE 3: Online TTT in-place (no leak) ===")
    model = build_eval_model(args, deq_state, device)
    for i, block in enumerate(model.blocks):
        if i < args.ttt_freeze_blocks:
            for p in block.parameters():
                p.requires_grad_(False)
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding_with_ttt(
        args, model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=seq_len, log_fn=log,
    )
    elapsed = time.perf_counter() - t0
    log(f"  sliding_bpb={bpb:.6f}  loss={loss:.6f}  time={elapsed:.1f}s")
    results["ttt_online_inplace"] = bpb
    del model; torch.cuda.empty_cache()

    # ---- MODE 4: Online TTT with duplication (no leak) ----
    log("\n=== MODE 4: Online TTT with duplication (no leak) ===")
    model = build_eval_model(args, deq_state, device)
    ttt_setup(args, model, device, log_fn=log)
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding_with_ttt(
        args, model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=seq_len, log_fn=log,
    )
    elapsed = time.perf_counter() - t0
    log(f"  sliding_bpb={bpb:.6f}  loss={loss:.6f}  time={elapsed:.1f}s")
    results["ttt_online_dup"] = bpb
    del model; torch.cuda.empty_cache()

    # ---- Summary ----
    log("\n" + "=" * 50)
    log("RESULTS SUMMARY")
    log("=" * 50)
    for name, bpb in results.items():
        log(f"  {name:25s}  bpb={bpb:.6f}")

if __name__ == "__main__":
    main()
