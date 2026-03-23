"""Quick eval-only script: loads saved int6 model, runs 3 TTT modes on a token subset."""

import copy
import io
import math
import os
import sys
import time
import zstandard

import torch
import torch.nn.functional as F

# Reuse everything from train_gpt
from train_gpt import (
    Hyperparameters,
    GPT,
    CastedLinear,
    restore_low_dim_params_to_fp32,
    dequantize_mixed_int6,
    load_validation_tokens,
    build_sentencepiece_luts,
    eval_val_sliding,
    eval_val_sliding_with_ttt,
    ttt_setup,
)
import sentencepiece as spm


def main():
    val_tokens_limit = int(os.environ.get("VAL_TOKENS", 2_000_000))
    model_path = os.environ.get("MODEL_PATH", "final_model.int6.ptz")
    stride = int(os.environ.get("EVAL_STRIDE", 64))
    device = torch.device("cuda")

    # Build args with same defaults as the training run
    args = Hyperparameters()
    args.num_layers = int(os.environ.get("NUM_LAYERS", 11))
    args.bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 2048))
    args.xsa_last_n = int(os.environ.get("XSA_LAST_N", 4))
    args.rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    args.ln_scale = float(os.environ.get("LN_SCALE", 1))
    args.ttt_lr = float(os.environ.get("TTT_LR", 0.002))
    args.ttt_momentum = float(os.environ.get("TTT_MOMENTUM", 0.9))
    args.ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 2))
    args.eval_stride = stride

    seq_len = args.train_seq_len  # 2048

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Load val tokens (subset)
    val_tokens = load_validation_tokens(args.val_files, seq_len)
    total = val_tokens.numel()
    if val_tokens_limit > 0 and val_tokens_limit < total:
        usable = ((val_tokens_limit - 1) // seq_len) * seq_len + 1
        val_tokens = val_tokens[:usable]
    print(f"val_tokens: {val_tokens.numel()} / {total} ({100*val_tokens.numel()/total:.1f}%)")

    # Load quantized model
    print(f"loading model from {model_path}")
    with open(model_path, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(
        io.BytesIO(zstandard.ZstdDecompressor().decompress(quant_blob)),
        map_location="cpu",
    )
    # Need a reference state dict for dequantization shapes
    ref_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    )
    ref_sd = ref_model.state_dict()
    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], ref_sd)
    del ref_model, ref_sd

    def make_model(state_dict):
        m = GPT(
            vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
            num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
            tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
            logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
            mtp_num_heads=0, mtp_loss_weight=0.0,
            bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
            xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        ).to(device).bfloat16()
        for mod in m.modules():
            if isinstance(mod, CastedLinear):
                mod.float()
        restore_low_dim_params_to_fp32(m)
        m.load_state_dict(state_dict, strict=True)
        return m

    tied_dup_from = int(os.environ.get("TIED_DUP_FROM", -1))
    log = lambda s: print(s, flush=True)

    # --- Mode 1: No TTT ---
    dup_label = f" (tied_dup_from={tied_dup_from})" if tied_dup_from >= 0 else ""
    log(f"\n=== Mode 1: No TTT (sliding window){dup_label} ===")
    model = make_model(deq_state)
    if tied_dup_from >= 0:
        model.tied_dup_from = tied_dup_from
    model.eval()
    compiled = torch.compile(model, dynamic=False, fullgraph=True) if os.environ.get("TORCH_COMPILE", "1") != "0" else model
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding(
        args, compiled, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=seq_len,
    )
    torch.cuda.synchronize()
    log(f"no_ttt: val_loss={loss:.4f} val_bpb={bpb:.4f} time={time.perf_counter()-t0:.1f}s")
    del model, compiled
    torch.cuda.empty_cache()

    # --- Mode 2: TTT online in-place (no duplication) ---
    log(f"\n=== Mode 2: TTT online in-place (no duplication){dup_label} ===")
    model = make_model(deq_state)
    if tied_dup_from >= 0:
        model.tied_dup_from = tied_dup_from
    # Freeze early blocks
    for i, block in enumerate(model.blocks):
        if i < args.ttt_freeze_blocks:
            for p in block.parameters():
                p.requires_grad_(False)
    model.ttt_blocks = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding_with_ttt(
        args, model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=seq_len, log_fn=log,
    )
    torch.cuda.synchronize()
    log(f"ttt_inplace: val_loss={loss:.4f} val_bpb={bpb:.4f} time={time.perf_counter()-t0:.1f}s")
    del model
    torch.cuda.empty_cache()

    # --- Mode 3: TTT online duplicate ---
    log("\n=== Mode 3: TTT online duplicate ===")
    model = make_model(deq_state)
    args.ttt_duplicate = True
    ttt_setup(args, model, device, log_fn=log)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss, bpb = eval_val_sliding_with_ttt(
        args, model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, eval_seq_len=seq_len, log_fn=log,
    )
    torch.cuda.synchronize()
    log(f"ttt_duplicate: val_loss={loss:.4f} val_bpb={bpb:.4f} time={time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
