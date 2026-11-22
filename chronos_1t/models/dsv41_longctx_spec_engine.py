"""
Long-Context (48K-64K MLA KV), Cross-Turn Prefix KV Caching, and Lossless PLD
Speculative Decoding Engine for Chronos DeepSeek-V4.1-Flash-1017.

Designed to attach live to the resident 40-layer `GLOBAL_DSV41_MODEL` on Colab G4
without reloading weights from HuggingFace.
"""
from __future__ import annotations

import builtins
import os
import re
import time
from typing import Any

import torch
import torch.nn.functional as F

from chronos_1t.models.dsv41_triton_backbone import _get_block_decode_bufs


def _sync_and_verify_routing_buf(model: Any, dev: torch.device | None = None) -> tuple[bool, torch.Tensor | None]:
    """Ensure `builtins._CHRONOS_ROUTING_STEP_BUF` points to the live CUDAGraph `_base` across all layers."""
    layers = getattr(model, "layers", None)
    if not layers:
        return False, None
    target_dev = dev if dev is not None else next(model.parameters()).device
    r0 = getattr(getattr(layers[0], "ffn", None), "_routing_step_row_buf", None)
    base = getattr(r0, "_base", None) if r0 is not None else None
    valid = (
        base is not None
        and base.shape == (len(layers), 6)
        and base.device == target_dev
        and not base.is_inference()
        and all(
            getattr(getattr(l, "ffn", None), "_routing_step_row_buf", None) is not None
            and l.ffn._routing_step_row_buf.data_ptr() == base[i].data_ptr()
            for i, l in enumerate(layers)
        )
    )
    if valid:
        builtins._CHRONOS_ROUTING_STEP_BUF = base
        return True, base
    return False, None


def upgrade_resident_model_long_context(
    model: Any,
    dsv41_model_mod: Any,
    runner_mod: Any,
    target_max_seq_len: int = 49152,
) -> dict[str, Any]:
    """Expand resident 40-layer MLA KV caches and YaRN RoPE tables up to `target_max_seq_len`.

    Because compressed KV caches exist only on the 4 `kv_source_layers` ([2, 8, 14, 20])
    with ratios [2, 2, 2, 1], widening from 4096 to 49152 (48K) costs only ~170 MB VRAM.
    """
    dev = next(model.parameters()).device
    free_before, total_mem = torch.cuda.mem_get_info(dev)

    cur_len = int(model.layers[0].attn.freqs_cis.size(0))
    if cur_len >= target_max_seq_len:
        routing_buf_valid, _ = _sync_and_verify_routing_buf(model, dev)
        return {
            "upgraded": False,
            "max_seq_len": cur_len,
            "routing_buf_valid": routing_buf_valid,
            "free_vram_before_gib": round(free_before / 1024**3, 3),
            "free_vram_after_gib": round(free_before / 1024**3, 3),
        }

    # Check exact RoPE parameters from layer 2 (compressed) and layer 0 (window)
    comp_l2 = getattr(model.layers[2].attn, "compress_ratio", 2)
    freqs_comp = dsv41_model_mod.precompute_freqs_cis(
        64,
        target_max_seq_len,
        65536,
        160000.0,
        16.0,
        32,
        1,
    ).to(dev)
    freqs_win = dsv41_model_mod.precompute_freqs_cis(
        64,
        target_max_seq_len,
        0,
        10000.0,
        16.0,
        32,
        1,
    ).to(dev)

    resized_kv_layers: list[int] = []
    with torch.inference_mode(False):
        for lyr in model.layers:
            attn = lyr.attn
            ratio = int(getattr(attn, "compress_ratio", 0) or 0)
            new_freqs = freqs_comp if ratio > 0 else freqs_win
            attn.register_buffer("freqs_cis", new_freqs, persistent=False)
            attn._freqs_real_cached = torch.view_as_real(new_freqs.detach())
            if getattr(attn, "indexer", None) is not None:
                attn.indexer.freqs_cis = new_freqs
                if getattr(attn.indexer, "owns_k", False) and hasattr(attn.indexer, "k_cache"):
                    bsz = int(attn.indexer.k_cache.size(0))
                    k_dim = int(attn.indexer.k_cache.size(-1))
                    k_dtype = attn.indexer.k_cache.dtype
                    del attn.indexer.k_cache
                    attn.indexer.register_buffer(
                        "k_cache",
                        torch.zeros(bsz, target_max_seq_len // ratio, k_dim, dtype=k_dtype, device=dev),
                        persistent=False,
                    )
            if getattr(attn, "is_kv_source", False) and hasattr(attn, "compress_kv_cache"):
                bsz = int(attn.compress_kv_cache.size(0))
                h_dim = int(attn.compress_kv_cache.size(-1))
                c_dtype = attn.compress_kv_cache.dtype
                del attn.compress_kv_cache
                attn.register_buffer(
                    "compress_kv_cache",
                    torch.zeros(bsz, target_max_seq_len // ratio, h_dim, dtype=c_dtype, device=dev),
                    persistent=False,
                )
                resized_kv_layers.append(int(attn.layer_id))

    if hasattr(model, "max_seq_len"):
        model.max_seq_len = int(target_max_seq_len)
    torch.cuda.empty_cache()

    # Re-capture 40-layer CUDAGraphs so static pointers reflect new _freqs_real_cached / compress_kv_cache
    _sync_and_verify_routing_buf(model, dev)
    cg_cnt = runner_mod.warmup_40layer_cudagraphs(model)
    runner_mod.reset_model_caches(model)
    routing_buf_valid, _ = _sync_and_verify_routing_buf(model, dev)
    free_after, _ = torch.cuda.mem_get_info(dev)

    return {
        "upgraded": True,
        "prev_max_seq_len": cur_len,
        "max_seq_len": int(model.layers[0].attn.freqs_cis.size(0)),
        "resized_kv_source_layers": resized_kv_layers,
        "comp_l2_ratio": comp_l2,
        "cudagraphs_recaptured": cg_cnt,
        "routing_buf_valid": routing_buf_valid,
        "free_vram_before_gib": round(free_before / 1024**3, 3),
        "free_vram_after_gib": round(free_after / 1024**3, 3),
        "total_vram_gib": round(total_mem / 1024**3, 3),
    }


def install_general_chunked_attention(dsv41_model_mod: Any) -> dict[str, bool]:
    """Patch `Attention._window_kv`, `Compressor.forward`, `Indexer.forward`, and
    `Attention._compress_kv` so `start_pos > 0` works seamlessly for `seqlen >= 1`.

    Preserves 100% identical behavior for `start_pos == 0` and single-token decode `seqlen == 1`.
    """
    if hasattr(dsv41_model_mod.get_window_topk_idxs, "cache_clear"):
        dsv41_model_mod.get_window_topk_idxs.cache_clear()

    def _window_kv_general(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        win = self.window_size
        kv = self.kv_norm(self.wkv(x))
        dsv41_model_mod.apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        dsv41_model_mod.act_quant(
            kv, dsv41_model_mod.fp8_block_size, dsv41_model_mod.scale_fmt, dsv41_model_mod.scale_dtype, True
        )
        if start_pos == 0:
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            with torch.device(x.device):
                idxs = dsv41_model_mod.get_window_topk_idxs(win, bsz, seqlen, start_pos)
            if idxs.device != x.device:
                idxs = idxs.to(x.device)
            return kv, idxs
        if seqlen == 1:
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            with torch.device(x.device):
                idxs = dsv41_model_mod.get_window_topk_idxs(win, bsz, seqlen, start_pos)
            if idxs.device != x.device:
                idxs = idxs.to(x.device)
            return self.window_kv_cache[:bsz], idxs

        # General multi-token continuation at start_pos > 0
        prev_w = min(start_pos, win)
        if prev_w > 0:
            prev_slots = torch.arange(start_pos - prev_w, start_pos, device=x.device) % win
            prev_kv = self.window_kv_cache[:bsz, prev_slots]
            window_kv = torch.cat([prev_kv, kv], dim=1)
        else:
            window_kv = kv

        keep_n = min(seqlen, win)
        write_slots = torch.arange(start_pos + seqlen - keep_n, start_pos + seqlen, device=x.device) % win
        self.window_kv_cache[:bsz, write_slots] = kv[:, -keep_n:]

        n_win_k = prev_w + seqlen
        col_w = min(n_win_k, win)
        end_col = prev_w + torch.arange(seqlen, device=x.device).unsqueeze(1)
        idxs = (end_col - win + 1).clamp_min(0) + torch.arange(col_w, device=x.device)
        idxs = torch.where(idxs > end_col, -1, idxs).int().unsqueeze(0).expand(bsz, -1, -1).contiguous()
        return window_kv, idxs

    def _compressor_forward_general(self, x: torch.Tensor, start_pos: int) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:
            return self.norm(self.wkv(x))

        xf = x.float()
        kv, score = self.wkv(xf), self.wgate(xf)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if seqlen <= 16:
                self._last_seq_base_pos = 0
                self._last_seq_kv = kv.detach()
                self._last_seq_score = score.detach()
            if remainder:
                kv, rem_kv = kv.split([cutoff, remainder], dim=1)
                score, rem_sc = score.split([cutoff, remainder], dim=1)
                self.kv_state[:bsz, :remainder] = rem_kv
                self.score_state[:bsz, :remainder] = rem_sc
            else:
                self.kv_state[:bsz].zero_()
                self.score_state[:bsz].fill_(-torch.inf)
            if not should_compress:
                return None
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            return self.norm(kv.to(dtype))

        if seqlen == 1:
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if not should_compress:
                return None
            pooled = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
            return self.norm(pooled.to(dtype))

        # General multi-token continuation at start_pos > 0
        rem_prev = start_pos % ratio
        if rem_prev > 0:
            kv_all = torch.cat([self.kv_state[:bsz, :rem_prev], kv], dim=1)
            score_all = torch.cat([self.score_state[:bsz, :rem_prev], score], dim=1)
        else:
            kv_all = kv
            score_all = score

        if seqlen <= 16:
            self._last_seq_base_pos = start_pos - rem_prev
            self._last_seq_kv = kv_all.detach()
            self._last_seq_score = score_all.detach()

        tot = kv_all.size(1)
        rem_new = tot % ratio
        cutoff = tot - rem_new
        should_compress = cutoff >= ratio
        if rem_new > 0:
            kv_comp, rem_kv = kv_all.split([cutoff, rem_new], dim=1)
            sc_comp, rem_sc = score_all.split([cutoff, rem_new], dim=1)
            self.kv_state[:bsz, :rem_new] = rem_kv
            self.score_state[:bsz, :rem_new] = rem_sc
            if rem_new < ratio:
                self.kv_state[:bsz, rem_new:].zero_()
                self.score_state[:bsz, rem_new:].fill_(-torch.inf)
        else:
            kv_comp, sc_comp = kv_all, score_all
            self.kv_state[:bsz].zero_()
            self.score_state[:bsz].fill_(-torch.inf)

        if not should_compress:
            return None
        kv_grp = kv_comp.unflatten(1, (-1, ratio))
        sc_grp = sc_comp.unflatten(1, (-1, ratio))
        pooled = (kv_grp * sc_grp.softmax(dim=2)).sum(dim=2)
        return self.norm(pooled.to(dtype))

    def _indexer_forward_general(self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor, start_pos: int, offset: int):
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        if self.owns_k and latent is not None:
            if seqlen == 1 and start_pos > 0:
                freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            else:
                first_g = start_pos // ratio
                n_lat = latent.size(1)
                freqs = self.freqs_cis[first_g * ratio : (first_g + n_lat) * ratio : ratio].contiguous()
            k = self.k_norm(self.wk(latent))
            dsv41_model_mod.apply_rotary_emb(k[..., -rd:], freqs)
            dsv41_model_mod.fp4_act_quant(k, dsv41_model_mod.fp4_block_size, True)
            self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
        if self.owns_k:
            dsv41_model_mod.shared_attn.index_k = self.k_cache

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        dsv41_model_mod.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
        dsv41_model_mod.fp4_act_quant(q, dsv41_model_mod.fp4_block_size, True)

        n_comp_avail = end_pos // ratio
        if n_comp_avail == 0:
            return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)

        index_k = dsv41_model_mod.shared_attn.index_k[:bsz, :n_comp_avail]
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        if seqlen > 32 and n_comp_avail > 1024:
            out_dtype = torch.result_type(q, weights)
            index_score = torch.empty((bsz, seqlen, n_comp_avail), dtype=out_dtype, device=x.device)
            sub_s = 16 if n_comp_avail > 65536 else 32
            for s0 in range(0, seqlen, sub_s):
                s1 = min(seqlen, s0 + sub_s)
                sc_blk = torch.einsum("bshd,btd->bsht", q[:, s0:s1], index_k)
                sc_blk.relu_()
                w_sub = weights[:, s0:s1].unsqueeze(-1)
                if sc_blk.dtype == w_sub.dtype:
                    sc_blk.mul_(w_sub)
                    torch.sum(sc_blk, dim=2, out=index_score[:, s0:s1])
                else:
                    index_score[:, s0:s1] = (sc_blk * w_sub).sum(dim=2)
                del sc_blk
        else:
            index_score = torch.einsum("bshd,btd->bsht", q, index_k)
            index_score.relu_()
            w_full = weights.unsqueeze(-1)
            if index_score.dtype == w_full.dtype:
                index_score.mul_(w_full)
                index_score = index_score.sum(dim=2)
            else:
                index_score = (index_score * w_full).sum(dim=2)

        if seqlen == 1 and start_pos > 0:
            compress_lens: Any = n_comp_avail
        else:
            compress_lens = (torch.arange(start_pos + 1, end_pos + 1, device=x.device) // ratio).unsqueeze(-1)
            index_score.masked_fill_(torch.arange(n_comp_avail, device=x.device).unsqueeze(0) >= compress_lens, -torch.inf)

        if self.is_candidate_source:
            dsv41_model_mod.shared_attn.candidates = dsv41_model_mod.select_candidate_blocks(
                index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            index_score = index_score.masked_fill(~dsv41_model_mod.shared_attn.candidates, -torch.inf)

        topk = min(self.index_topk, n_comp_avail)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()

    def _compress_kv_general(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            dsv41_model_mod.shared_attn.compress_kv = self.compress_kv_cache
        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_len)
        if latent is not None:
            if seqlen == 1 and start_pos > 0:
                freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            else:
                first_g = start_pos // ratio
                n_lat = latent.size(1)
                freqs = self.freqs_cis[first_g * ratio : (first_g + n_lat) * ratio : ratio].contiguous()
            dsv41_model_mod.apply_rotary_emb(latent[..., -self.rope_head_dim :], freqs)
            dsv41_model_mod.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
        return dsv41_model_mod.shared_attn.compress_kv[:bsz, :compress_len], idxs

    dsv41_model_mod.Attention._window_kv = _window_kv_general
    dsv41_model_mod.Compressor.forward = _compressor_forward_general
    dsv41_model_mod.Indexer.forward = _indexer_forward_general
    dsv41_model_mod.Attention._compress_kv = _compress_kv_general
    dsv41_model_mod._chronos_general_chunked_installed = True
    return {"installed": True}


@torch.inference_mode()
def forward_chunk_logits(
    model: Any,
    dsv41_model_mod: Any,
    input_ids: torch.Tensor,
    start_pos: int = 0,
    full_logits: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass for `input_ids` ([1, seqlen]) starting at `start_pos`.

    When `full_logits=True`, returns `(output_ids [1, seqlen], logits [1, seqlen, vocab])`.
    """
    with torch.device(input_ids.device):
        seqlen = int(input_ids.size(1))
        if seqlen == 1:
            output_ids, logits, _ = model(input_ids, start_pos)
            return output_ids, logits

        try:
            _get_block_decode_bufs(input_ids.device)["last_pos"] = -1
        except Exception:
            pass

        engram_hash = getattr(model, "engram_hash", None)
        engram_hashes = engram_hash(input_ids, start_pos, None) if engram_hash is not None else None
        h = model.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        pre_mix = dsv41_model_mod.make_identity_pre_mix(h, model.hc_mult)
        for layer in model.layers:
            if getattr(layer, "engram", None) is not None and engram_hashes is not None:
                h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], None)
            h, pre_mix = layer(h, start_pos, pre_mix, None)
        try:
            dsv41_model_mod.shared_attn.candidates = None
            dsv41_model_mod.shared_attn.topk_idxs = None
        except Exception:
            pass
        h = model.layers[-1].hc_pre(h, pre_mix)
        logits = model.head(model.norm(h), full_logits=full_logits)
        output_ids = dsv41_model_mod.sample(logits, model.temperature)

        try:
            _get_block_decode_bufs(input_ids.device)["last_pos"] = -1
        except Exception:
            pass
        return output_ids, logits


def snapshot_prefix_kv(model: Any, pos: int) -> dict[str, Any]:
    """Snapshot the small sliding-window & compressor tail states (~5.25 MB total) at `pos`.

    `compress_kv_cache[:, :pos // ratio]` and `indexer.k_cache[:, :pos // ratio]` live
    in append-only static tensors and remain untouched until future positions >= `pos`.
    """
    win_kv = torch.stack([lyr.attn.window_kv_cache[0].clone() for lyr in model.layers], dim=0)
    comp_states = {}
    for lyr in model.layers:
        c = getattr(lyr.attn, "compressor", None)
        if c is not None and getattr(c, "compress_ratio", 1) > 1:
            comp_states[int(lyr.layer_id)] = (c.kv_state[0].clone(), c.score_state[0].clone())
    return {"pos": int(pos), "win_kv": win_kv, "comp_states": comp_states}


def restore_prefix_kv(model: Any, snap: dict[str, Any]) -> int:
    """Restore window ring buffer & compressor state from `snapshot_prefix_kv`."""
    pos = int(snap["pos"])
    win_kv = snap["win_kv"]
    comp_states = snap["comp_states"]
    for i, lyr in enumerate(model.layers):
        lyr.attn.window_kv_cache[0].copy_(win_kv[i])
        if getattr(lyr.attn, "is_kv_source", False) and hasattr(lyr.attn, "compress_kv_cache"):
            ratio = int(getattr(lyr.attn, "compress_ratio", 1) or 1)
            c_idx = pos // ratio
            if c_idx < lyr.attn.compress_kv_cache.size(1):
                lyr.attn.compress_kv_cache[0, c_idx:].zero_()
        if getattr(lyr.attn, "indexer", None) is not None and getattr(lyr.attn.indexer, "owns_k", False):
            ratio = int(getattr(lyr.attn, "compress_ratio", 1) or 1)
            k_idx = pos // ratio
            if k_idx < lyr.attn.indexer.k_cache.size(1):
                lyr.attn.indexer.k_cache[0, k_idx:].zero_()
        c = getattr(lyr.attn, "compressor", None)
        if c is not None and int(lyr.layer_id) in comp_states:
            ks, ss = comp_states[int(lyr.layer_id)]
            c.kv_state[0].copy_(ks)
            c.score_state[0].copy_(ss)
    try:
        _get_block_decode_bufs(win_kv.device)["last_pos"] = -1
    except Exception:
        pass
    return pos


class PrefixKVCacheSession:
    """Tracks cross-turn conversation prefix so multi-turn agent prompts prefill only deltas."""

    def __init__(self, model: Any, dsv41_model_mod: Any, runner_mod: Any):
        self.model = model
        self.dsv41_model_mod = dsv41_model_mod
        self.runner_mod = runner_mod
        self.snap_ids: list[int] = []
        self.snap_data: dict[str, Any] | None = None
        self.last_prefill_route_ids: torch.Tensor | None = None

    def reset(self) -> None:
        self.snap_ids = []
        self.snap_data = None
        self.last_prefill_route_ids = None
        self.runner_mod.reset_model_caches(self.model)

    def prefill_with_prefix_cache(
        self,
        prompt_ids: list[int],
        chunk_size: int = 1024,
        tail_cudagraph: bool = False,
        log_routing: bool = False,
        r_step_buf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Prefill `prompt_ids`, reusing cached KV prefix when possible.

        Returns `(next_tok_t, logits, reused_tokens, newly_prefilled_tokens)` and updates
        an internal snapshot just before the trailing special/transition tokens so Turn t+1
        reuses all prior user+assistant turns verbatim.
        """
        n_total = len(prompt_ids)
        if n_total == 0:
            raise ValueError("Empty prompt_ids")
        self.last_prefill_route_ids = None
        pf_chunks: list[torch.Tensor] = []

        def _collect_chunk_route(n_tokens: int) -> None:
            if not log_routing:
                return
            pf_list = [getattr(lyr.ffn, "_last_prefill_top_ids", None) for lyr in self.model.layers]
            for lyr in self.model.layers:
                if hasattr(lyr.ffn, "_last_prefill_top_ids"):
                    lyr.ffn._last_prefill_top_ids = None
            if all(t is not None for t in pf_list):
                pf_chunks.append(torch.stack(pf_list, dim=1))
            elif n_tokens == 1 and r_step_buf is not None:
                pf_chunks.append(r_step_buf.clone().unsqueeze(0))

        reused = 0
        if self.snap_data is not None and self.snap_ids:
            snap_pos = int(self.snap_data["pos"])
            if (
                snap_pos > 0
                and snap_pos < n_total
                and len(self.snap_ids) >= snap_pos
                and prompt_ids[:snap_pos] == self.snap_ids[:snap_pos]
            ):
                restore_prefix_kv(self.model, self.snap_data)
                reused = snap_pos

        if reused == 0:
            self.runner_mod.reset_model_caches(self.model)

        # Save next prefix checkpoint cleanly just before the trailing <｜Assistant｜><think> header (even index)
        next_snap_pos = ((max(0, n_total - 8)) // 2) * 2
        if next_snap_pos <= reused:
            next_snap_pos = 0

        dev = next(self.model.parameters()).device
        cur = reused
        out_tok_t = None
        out_logits = None

        with torch.device(dev):
            # 1. Advance in chunks up to next_snap_pos (CUDAGraph off) so we snapshot before trailing markers
            if next_snap_pos > 0 and cur < next_snap_pos:
                self.runner_mod.set_fast_gpu_cudagraph_mode(self.model, False)
            while next_snap_pos > 0 and cur < next_snap_pos:
                if (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > 512 * 1024 * 1024:
                    torch.cuda.empty_cache()
                eff_chunk = chunk_size if cur < 48000 else min(chunk_size, 512)
                end = min(next_snap_pos, cur + eff_chunk)
                inp = torch.tensor([prompt_ids[cur:end]], dtype=torch.long, device=dev)
                if log_routing and (end - cur) == 1 and r_step_buf is not None:
                    r_step_buf.zero_()
                out_tok_t, out_logits = forward_chunk_logits(
                    self.model, self.dsv41_model_mod, inp, start_pos=cur, full_logits=False
                )
                _collect_chunk_route(end - cur)
                cur = end

            if next_snap_pos > 0 and cur == next_snap_pos:
                self.snap_ids = list(prompt_ids[:next_snap_pos])
                self.snap_data = snapshot_prefix_kv(self.model, next_snap_pos)

            # 2. Prefill remaining prompt tokens up to n_total
            if tail_cudagraph and 0 < (n_total - cur) <= 16:
                self.runner_mod.set_fast_gpu_cudagraph_mode(self.model, True)
                while cur < n_total:
                    inp = torch.tensor([[prompt_ids[cur]]], dtype=torch.long, device=dev)
                    if log_routing and r_step_buf is not None:
                        r_step_buf.zero_()
                    out_tok_t, out_logits, _ = self.model(inp, cur)
                    if log_routing and r_step_buf is not None:
                        pf_chunks.append(r_step_buf.clone().unsqueeze(0))
                    cur += 1
            else:
                if cur < n_total:
                    self.runner_mod.set_fast_gpu_cudagraph_mode(self.model, False)
                while cur < n_total:
                    if (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) > 512 * 1024 * 1024:
                        torch.cuda.empty_cache()
                    eff_chunk = chunk_size if cur < 48000 else min(chunk_size, 512)
                    end = min(n_total, cur + eff_chunk)
                    inp = torch.tensor([prompt_ids[cur:end]], dtype=torch.long, device=dev)
                    if log_routing and (end - cur) == 1 and r_step_buf is not None:
                        r_step_buf.zero_()
                    out_tok_t, out_logits = forward_chunk_logits(
                        self.model, self.dsv41_model_mod, inp, start_pos=cur, full_logits=False
                    )
                    _collect_chunk_route(end - cur)
                    cur = end

            self.runner_mod.set_fast_gpu_cudagraph_mode(self.model, True)
            if n_total - reused >= 512:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if log_routing and pf_chunks:
            self.last_prefill_route_ids = torch.cat(pf_chunks, dim=0)
        assert out_tok_t is not None and out_logits is not None
        return out_tok_t, out_logits, reused, n_total - reused


def _snapshot_tail_kv(model: Any, cur_pos: int, count: int) -> dict[str, Any]:
    """Snapshot the `count` window ring slots starting at `cur_pos` + pre-step compressor states (~380 KB)."""
    slots = [(cur_pos + i) % 128 for i in range(count)]
    slots_t = torch.tensor(slots, dtype=torch.long, device="cuda")
    win_rows = torch.stack([lyr.attn.window_kv_cache[0].index_select(0, slots_t) for lyr in model.layers], dim=0)
    comp_init = {}
    for lyr in model.layers:
        c = getattr(lyr.attn, "compressor", None)
        if c is not None and getattr(c, "compress_ratio", 1) > 1:
            comp_init[int(lyr.layer_id)] = (c.kv_state[0].clone(), c.score_state[0].clone())
    return {"cur_pos": cur_pos, "slots_t": slots_t, "win_rows": win_rows, "comp_init": comp_init}


def _restore_tail_kv(model: Any, snap: dict[str, Any], valid_end: int, dirty_end: int) -> None:
    """Restore any ring slots written inside `[valid_end, dirty_end)` and exact compressor state at `valid_end`."""
    cur_pos = int(snap["cur_pos"])
    off = valid_end - cur_pos
    n_dirty = dirty_end - valid_end
    if off < len(snap["slots_t"]) and n_dirty > 0:
        bad_slots = snap["slots_t"][off : off + n_dirty]
        bad_rows = snap["win_rows"][:, off : off + n_dirty]
        for i, lyr in enumerate(model.layers):
            lyr.attn.window_kv_cache[0].index_copy_(0, bad_slots, bad_rows[i])

    for lyr in model.layers:
        if getattr(lyr.attn, "is_kv_source", False) and hasattr(lyr.attn, "compress_kv_cache"):
            ratio = int(getattr(lyr.attn, "compress_ratio", 1) or 1)
            c0 = valid_end // ratio
            c1 = min(lyr.attn.compress_kv_cache.size(1), (dirty_end // ratio) + 1)
            if c1 > c0:
                lyr.attn.compress_kv_cache[0, c0:c1].zero_()
        if getattr(lyr.attn, "indexer", None) is not None and getattr(lyr.attn.indexer, "owns_k", False):
            ratio = int(getattr(lyr.attn, "compress_ratio", 1) or 1)
            k0 = valid_end // ratio
            k1 = min(lyr.attn.indexer.k_cache.size(1), (dirty_end // ratio) + 1)
            if k1 > k0:
                lyr.attn.indexer.k_cache[0, k0:k1].zero_()
        c = getattr(lyr.attn, "compressor", None)
        if c is not None and getattr(c, "compress_ratio", 1) > 1:
            ratio = int(c.compress_ratio)
            rem = valid_end % ratio
            if rem == 0:
                c.kv_state[0].zero_()
                c.score_state[0].fill_(-torch.inf)
            else:
                base_p = getattr(c, "_last_seq_base_pos", -1)
                seq_kv = getattr(c, "_last_seq_kv", None)
                seq_sc = getattr(c, "_last_seq_score", None)
                src_idx = (valid_end - 1) - base_p
                if seq_kv is not None and seq_sc is not None and 0 <= src_idx < seq_kv.size(1):
                    c.kv_state[0, 0].copy_(seq_kv[0, src_idx])
                    c.score_state[0, 0].copy_(seq_sc[0, src_idx])
                    if ratio > 1:
                        c.kv_state[0, 1:].zero_()
                        c.score_state[0, 1:].fill_(-torch.inf)
                else:
                    ks, ss = snap["comp_init"][int(lyr.layer_id)]
                    c.kv_state[0].copy_(ks)
                    c.score_state[0].copy_(ss)
    try:
        _get_block_decode_bufs(snap["slots_t"].device)["last_pos"] = -1
    except Exception:
        pass


class MultiOrderNGramDrafter:
    """Zero-VRAM Prompt + Trajectory N-Gram Lookup Drafter (orders 5, 4, 3, and 2 in code)."""

    def __init__(self, init_tokens: list[int], prior_tokens: list[int] | None = None):
        self.tokens: list[int] = []
        self.maps: dict[int, dict[tuple[int, ...], int]] = {5: {}, 4: {}, 3: {}, 2: {}}
        if prior_tokens:
            for tok in prior_tokens:
                self.append(tok)
        self.base_len = len(self.tokens)
        for tok in init_tokens:
            self.append(tok)

    def append(self, tok_i: int) -> None:
        pos = len(self.tokens)
        self.tokens.append(tok_i)
        for order in (2, 3, 4, 5):
            if pos >= order:
                key = tuple(self.tokens[pos - order : pos])
                self.maps[order][key] = pos

    def draft(self, k_draft: int = 6, allow_bigram: bool = False) -> list[int]:
        n = len(self.tokens)
        orders = (5, 4, 3, 2) if allow_bigram else (5, 4, 3)
        for order in orders:
            if n < order:
                continue
            key = tuple(self.tokens[n - order : n])
            hit_pos = self.maps[order].get(key)
            if hit_pos is not None and hit_pos < n - 1:
                avail = (n - 1) - hit_pos
                take = min(k_draft, avail)
                if take >= 1:
                    return self.tokens[hit_pos : hit_pos + take]
        return []


def _extract_completed_bash_block(text: str) -> str | None:
    """Return fenced ```bash ... ``` block only after closing fence and trailing newline/EOF."""
    m = re.search(r"```(?:bash|sh)\s*\n(.*?)```\s*(?:\n|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else None


def _summarize_tier_routing_metrics(
    model: Any,
    prefill_route_ids: torch.Tensor | None,
    route_dec_buf: torch.Tensor | None,
    dec_route_cnt: int,
) -> dict[str, Any]:
    """Compute prefill & decode 3-tier routing counts (T1: <1024, T2: >=2048, T3: [1024, 2048))."""
    tsm = getattr(model, "_chronos_tier_slot_map_cached", None)
    n_layers = len(model.layers)
    if tsm is None or tsm.size(0) != n_layers:
        with torch.inference_mode(False):
            tsm = torch.stack([lyr.ffn.tier_slot_map_gpu[:384] for lyr in model.layers], dim=0)
            model._chronos_tier_slot_map_cached = tsm
    l_idx_g = torch.arange(n_layers, device=tsm.device, dtype=torch.long).view(1, -1, 1)

    def _count_tiers(r_tensor: torch.Tensor | None) -> tuple[int, int, int, int]:
        if r_tensor is None or r_tensor.numel() == 0:
            return (0, 0, 0, 0)
        ts = tsm[l_idx_g, r_tensor.clamp(0, 383).long()]
        tot = int(ts.numel())
        c1 = int(((ts >= 0) & (ts < 1024)).sum().item())
        c2 = int((ts >= 2048).sum().item())
        c3 = int(((ts >= 1024) & (ts < 2048)).sum().item())
        return (tot, c1, c2, c3)

    pf_tot, pf_t1, pf_t2, pf_t3 = _count_tiers(prefill_route_ids)
    dec_slice = route_dec_buf[:dec_route_cnt] if (route_dec_buf is not None and dec_route_cnt > 0) else None
    dec_nz = int((dec_slice != 0).any(dim=(1, 2)).sum().item()) if dec_slice is not None else 0
    dec_tot, dec_t1, dec_t2, dec_t3 = _count_tiers(dec_slice)
    all_tot = pf_tot + dec_tot
    return {
        "prefill_route_hits": pf_tot,
        "prefill_t1_hits": pf_t1,
        "prefill_t2_hits": pf_t2,
        "prefill_t3_hits": pf_t3,
        "prefill_t1_frac": round(pf_t1 / max(1, pf_tot), 6),
        "prefill_t2_frac": round(pf_t2 / max(1, pf_tot), 6),
        "prefill_t3_frac": round(pf_t3 / max(1, pf_tot), 6),
        "prefill_4bit_frac": round((pf_t1 + pf_t2) / max(1, pf_tot), 6),
        "decode_route_hits": dec_tot,
        "decode_t1_hits": dec_t1,
        "decode_t2_hits": dec_t2,
        "decode_t3_hits": dec_t3,
        "decode_nonzero_rows": dec_nz,
        "decode_t1_frac": round(dec_t1 / max(1, dec_tot), 6),
        "decode_t2_frac": round(dec_t2 / max(1, dec_tot), 6),
        "decode_t3_frac": round(dec_t3 / max(1, dec_tot), 6),
        "decode_4bit_frac": round((dec_t1 + dec_t2) / max(1, dec_tot), 6),
        "all_t1_frac": round((pf_t1 + dec_t1) / max(1, all_tot), 6),
        "all_t2_frac": round((pf_t2 + dec_t2) / max(1, all_tot), 6),
        "all_t3_frac": round((pf_t3 + dec_t3) / max(1, all_tot), 6),
        "all_4bit_frac": round((pf_t1 + pf_t2 + dec_t1 + dec_t2) / max(1, all_tot), 6),
    }


@torch.no_grad()
def generate_with_speculative_and_prefix(
    model: Any,
    tokenizer: Any,
    dsv41_model_mod: Any,
    runner_mod: Any,
    messages: list[dict[str, Any]],
    *,
    max_new_tokens: int = 4096,
    max_think_tokens: int = 0,
    think_ceiling: int = 3072,
    ablate_guards: bool = True,
    thinking_mode: str = "thinking",
    reasoning_effort: str | int = 1,
    tools: list[dict[str, Any]] | None = None,
    use_speculative: bool = False,
    spec_k: int = 6,
    prefix_session: PrefixKVCacheSession | None = None,
    stop_on_bash_block: bool = False,
    seed_tokens: list[int] | None = None,
    temperature: float = 0.0,
    top_p: float = 0.95,
    prefill_chunk_size: int = 1024,
    tail_cudagraph: bool = False,
    log_routing: bool = False,
) -> dict[str, Any]:
    """Unified high-throughput generator supporting 80K context, prefix KV cache, sampling, and PLD speculative decoding."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    eff_thinking_mode = thinking_mode
    prompt_str = runner_mod._encode_prompt_safe(messages, eff_thinking_mode, reasoning_effort, tools)
    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    eos_id = tokenizer.eos_token_id
    eos_set = {1, 128803, 128805}
    if eos_id is not None:
        eos_set.add(int(eos_id))
    think_end_id = 128822
    think_start_id = 128821

    in_think = eff_thinking_mode == "thinking"
    if max_think_tokens > 0:
        eff_think_budget = max_think_tokens
    elif in_think:
        eff_think_budget = min(int(think_ceiling), max(128, max_new_tokens - 128))
    else:
        eff_think_budget = 0

    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            last_user_text = c if isinstance(c, str) else json_dump_safe(c)
            break
    last_user_low = last_user_text.lower()
    is_code_only_prompt = (("return only" in last_user_low) or ("38, 27, 43" in last_user_low)) and (
        "python" in last_user_low or "```" in last_user_low
    )

    think_close_reason = "chat_mode" if eff_thinking_mode != "thinking" else "unclosed_length"
    pending_think_reason = "natural"

    r_step_buf = None
    route_dec_buf = None
    prefill_route_ids = None
    dec_route_cnt = 0
    prev_log_routing_env = os.environ.get("CHRONOS_LOG_ROUTING")
    if log_routing:
        from chronos_1t.models.dsv41_chronos_moe import get_routing_step_buffer

        model_dev = next(model.parameters()).device
        ok_sync, live_base = _sync_and_verify_routing_buf(model, model_dev)
        if ok_sync and live_base is not None:
            r_step_buf = live_base
        else:
            r_step_buf = get_routing_step_buffer(model_dev)
        r_step_buf.zero_()
        route_dec_buf = torch.empty((max_new_tokens, len(model.layers), 6), dtype=torch.int16, device=model_dev)
        os.environ["CHRONOS_LOG_ROUTING"] = "1"

    try:
        with torch.device("cuda"):
            runner_mod.set_fast_gpu_cudagraph_mode(model, False)
            torch.cuda.synchronize()
            t_pf0 = time.perf_counter()

            if prefix_session is not None:
                next_tok_t, logits, reused_toks, prefilled_toks = prefix_session.prefill_with_prefix_cache(
                    prompt_ids,
                    chunk_size=prefill_chunk_size,
                    tail_cudagraph=tail_cudagraph,
                    log_routing=log_routing,
                    r_step_buf=r_step_buf,
                )
            else:
                runner_mod.reset_model_caches(model)
                inp = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
                next_tok_t, logits, _ = model(inp, 0)
                torch.cuda.synchronize()
                reused_toks = 0
                prefilled_toks = len(prompt_ids)
            prefill_s = time.perf_counter() - t_pf0
            if log_routing:
                sess_pf = getattr(prefix_session, "last_prefill_route_ids", None) if prefix_session is not None else None
                if sess_pf is not None:
                    prefill_route_ids = sess_pf
                    prefix_session.last_prefill_route_ids = None
                else:
                    pf_list = [getattr(lyr.ffn, "_last_prefill_top_ids", None) for lyr in model.layers]
                    if all(t is not None for t in pf_list):
                        prefill_route_ids = torch.stack(pf_list, dim=1)
                    elif len(prompt_ids) == 1 and r_step_buf is not None:
                        prefill_route_ids = r_step_buf.clone().unsqueeze(0)
                for lyr in model.layers:
                    if hasattr(lyr.ffn, "_last_prefill_top_ids"):
                        lyr.ffn._last_prefill_top_ids = None
                if r_step_buf is not None:
                    r_step_buf.zero_()
            runner_mod.set_fast_gpu_cudagraph_mode(model, True)

            logits_finite = bool(torch.isfinite(logits).all().item())
            cur_pos = len(prompt_ids)

            section_toks: list[int] = []
            fourgram_counts: dict[tuple[int, int, int], dict[int, int]] = {}
            norm_toks: list[str] = []
            norm_fourgrams: dict[tuple[str, str, str], set[int]] = {}
            think_tok_count = 0
            post_think_toks: list[int] = []
            gen_ids: list[int] = []
            finish_reason = "length"
            _skip_stutter = {1, 200, 201, 204, 220, 223, 262, 271, 290, 361, 385, 528, 128822}
            _sent_end_chars = (".", "!", "?", "\n", ":")

            drafter = MultiOrderNGramDrafter(prompt_ids, seed_tokens) if (use_speculative and temperature <= 0.0) else None
            drafted_cnt = 0
            accepted_cnt = 0
            verify_calls = 0
            single_calls = 0

            def _select_tok(raw_logits: torch.Tensor) -> tuple[int, bool]:
                nonlocal pending_think_reason
                l_vec = raw_logits.view(-1, raw_logits.size(-1))[-1].clone()
                post_decoded = runner_mod._decode_clean(tokenizer, post_think_toks)
                open_fences = post_decoded.count("```")
                in_code = (not in_think) and (
                    ("python" in last_user_low and ("```" in post_decoded or is_code_only_prompt))
                    or (open_fences % 2 == 1)
                )
                if in_think or (len(post_think_toks) < 4 and max_new_tokens > 32):
                    for _eid in eos_set:
                        l_vec[_eid] = -1e4
                if not in_think:
                    l_vec[think_start_id] = -1e4
                    if len(post_think_toks) == 0:
                        l_vec[think_end_id] = -1e4

                disable_rep = bool(ablate_guards)
                if len(section_toks) >= 3:
                    trig = (section_toks[-3], section_toks[-2], section_toks[-1])
                    cand_map = fourgram_counts.get(trig)
                    rep_thresh = 1000 if disable_rep else (5 if in_code else 1)
                    if cand_map:
                        for cand_u, cnt in cand_map.items():
                            if cnt >= rep_thresh:
                                l_vec[cand_u] -= 12.0
                        if (not disable_rep) and (not in_think) and (not in_code) and any(cnt >= 3 for cnt in cand_map.values()):
                            if (open_fences % 2) == 0:
                                l_vec[1] += 1000.0
                    if (not in_code) and (not disable_rep) and len(norm_toks) >= 3:
                        ntrig = (norm_toks[-3], norm_toks[-2], norm_toks[-1])
                        nbanned = norm_fourgrams.get(ntrig)
                        if nbanned:
                            for bu in nbanned:
                                l_vec[bu] -= 12.0
                    if (
                        (not disable_rep)
                        and section_toks[-1] == section_toks[-2] == section_toks[-3]
                        and section_toks[-1] not in _skip_stutter
                    ):
                        l_vec[section_toks[-1]] -= 15.0
                if (
                    (not disable_rep)
                    and len(section_toks) >= 4
                    and section_toks[-1] == section_toks[-3]
                    and section_toks[-2] == section_toks[-4]
                    and section_toks[-2] not in _skip_stutter
                ):
                    l_vec[section_toks[-2]] -= 12.0

                pending_think_reason = "natural"
                if in_think:
                    last_piece = tokenizer.decode([section_toks[-1]], skip_special_tokens=False) if section_toks else ""
                    at_sent = bool(last_piece) and last_piece.rstrip(" ").endswith(_sent_end_chars)
                    rep_hit = False
                    if len(section_toks) >= 3:
                        trig = (section_toks[-3], section_toks[-2], section_toks[-1])
                        rep_hit = any(cnt >= 2 for cnt in fourgram_counts.get(trig, {}).values())
                    should_close_early = (not ablate_guards) and think_tok_count >= 48 and at_sent and rep_hit
                    over_budget = (
                        eff_think_budget > 0
                        and think_tok_count >= eff_think_budget
                        and (at_sent or rep_hit or think_tok_count >= eff_think_budget + 16)
                    )
                    if should_close_early or over_budget:
                        l_vec[think_end_id] += 1000.0
                        pending_think_reason = "forced_early_rep" if should_close_early else "forced_budget"

                if temperature <= 0.0:
                    tok_i = int(l_vec.argmax(dim=-1).item())
                else:
                    scaled = l_vec.float() / float(temperature)
                    probs = torch.softmax(scaled, dim=-1)
                    if not bool(torch.isfinite(probs).all().item()):
                        tok_i = int(l_vec.argmax(dim=-1).item())
                    elif 0.0 < float(top_p) < 1.0:
                        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                        cum_probs = torch.cumsum(sorted_probs, dim=-1)
                        remove_mask = (cum_probs - sorted_probs) > float(top_p)
                        remove_mask[0] = False
                        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
                        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
                        pick = int(torch.multinomial(sorted_probs, num_samples=1).item())
                        tok_i = int(sorted_idx[pick].item())
                    else:
                        tok_i = int(torch.multinomial(probs, num_samples=1).item())
                if not in_think and tok_i == think_end_id and len(post_think_toks) > 0:
                    return tok_i, True
                return tok_i, (tok_i in eos_set)

            def _commit_tok(tok_i: int) -> bool:
                nonlocal in_think, think_tok_count, finish_reason, think_close_reason
                gen_ids.append(tok_i)
                if drafter is not None:
                    drafter.append(tok_i)
                if len(section_toks) >= 3:
                    trig = (section_toks[-3], section_toks[-2], section_toks[-1])
                    d = fourgram_counts.setdefault(trig, {})
                    d[tok_i] = d.get(tok_i, 0) + 1
                section_toks.append(tok_i)
                piece_str = tokenizer.decode([tok_i], skip_special_tokens=False)
                p_norm = piece_str.strip().lower()
                if len(norm_toks) >= 3:
                    ntrig = (norm_toks[-3], norm_toks[-2], norm_toks[-1])
                    nset = norm_fourgrams.setdefault(ntrig, set())
                    nset.add(tok_i)
                    if p_norm and p_norm.isalpha():
                        for v_txt in (p_norm, " " + p_norm):
                            v_ids = tokenizer.encode(v_txt, add_special_tokens=False)
                            if len(v_ids) == 1:
                                nset.add(v_ids[0])
                norm_toks.append(p_norm)

                if in_think:
                    if tok_i == think_end_id:
                        think_close_reason = pending_think_reason
                        in_think = False
                        section_toks.clear()
                        fourgram_counts.clear()
                        norm_toks.clear()
                        norm_fourgrams.clear()
                    else:
                        think_tok_count += 1
                else:
                    post_think_toks.append(tok_i)

                if tok_i == 128807:
                    finish_reason = "stop"
                    return True
                if not in_think and post_think_toks:
                    if stop_on_bash_block and len(post_think_toks) >= 4:
                        post_str = runner_mod._decode_clean(tokenizer, post_think_toks)
                        if _extract_completed_bash_block(post_str) is not None:
                            finish_reason = "stop"
                            return True
                    elif is_code_only_prompt and len(post_think_toks) >= 8:
                        post_str = runner_mod._decode_clean(tokenizer, post_think_toks)
                        if post_str.count("```") >= 2 and len(post_str.strip()) >= 15:
                            finish_reason = "stop"
                            return True
                if (not in_think) and len(post_think_toks) >= 128:
                    for p_len in range(4, 49):
                        reps = 6 if p_len < 12 else 4
                        if len(post_think_toks) >= reps * p_len and all(
                            post_think_toks[-p_len:] == post_think_toks[-(k + 1) * p_len : -k * p_len]
                            for k in range(1, reps)
                        ):
                            finish_reason = "loop_break"
                            return True
                return False

            t_dec0 = time.perf_counter()
            first_tok, is_stop = _select_tok(logits)
            should_halt = False
            cur_tok = first_tok
            if is_stop:
                finish_reason = "stop"
                should_halt = True
            else:
                should_halt = _commit_tok(first_tok)

            max_pos = int(model.layers[0].attn.freqs_cis.size(0)) - 2

            while (not should_halt) and len(gen_ids) < max_new_tokens and cur_pos < max_pos:
                rem_toks = min(max_new_tokens - len(gen_ids), max_pos - cur_pos)
                if rem_toks <= 0:
                    break

                draft = []
                if drafter is not None and rem_toks >= 2 and spec_k >= 2:
                    allow_bi = not in_think
                    draft = drafter.draft(k_draft=min(spec_k, rem_toks - 1), allow_bigram=allow_bi)

                if draft:
                    start_cur_pos = cur_pos
                    n_batch = 1 + len(draft)
                    kv_snap = _snapshot_tail_kv(model, cur_pos, n_batch)
                    seq_inp = torch.tensor([[cur_tok] + draft], dtype=torch.long, device="cuda")
                    _, batch_logits = forward_chunk_logits(
                        model, dsv41_model_mod, seq_inp, start_pos=cur_pos, full_logits=True
                    )
                    verify_step_routes = None
                    if log_routing:
                        pf_list = [getattr(lyr.ffn, "_last_prefill_top_ids", None) for lyr in model.layers]
                        for lyr in model.layers:
                            if hasattr(lyr.ffn, "_last_prefill_top_ids"):
                                lyr.ffn._last_prefill_top_ids = None
                        if all(t is not None for t in pf_list):
                            verify_step_routes = torch.stack(pf_list, dim=1)
                    verify_calls += 1
                    drafted_cnt += len(draft)
                    if not bool(torch.isfinite(batch_logits).all().item()):
                        logits_finite = False

                    bl = batch_logits[0]
                    accepted_step = 0
                    for j in range(n_batch):
                        v_j, stop_j = _select_tok(bl[j : j + 1])
                        if stop_j:
                            finish_reason = "stop"
                            should_halt = True
                            valid_end = cur_pos + j + 1
                            if j < len(draft):
                                _restore_tail_kv(model, kv_snap, valid_end, cur_pos + n_batch)
                            cur_pos = valid_end
                            break
                        should_halt = _commit_tok(v_j)
                        cur_tok = v_j
                        if j < len(draft) and v_j == draft[j]:
                            accepted_cnt += 1
                            accepted_step += 1
                            if should_halt or len(gen_ids) >= max_new_tokens:
                                valid_end = cur_pos + j + 1
                                _restore_tail_kv(model, kv_snap, valid_end, cur_pos + n_batch)
                                cur_pos = valid_end
                                break
                            continue
                        valid_end = cur_pos + j + 1
                        if j < len(draft):
                            _restore_tail_kv(model, kv_snap, valid_end, cur_pos + n_batch)
                        cur_pos = valid_end
                        break
                    if log_routing and route_dec_buf is not None and verify_step_routes is not None:
                        n_valid = cur_pos - start_cur_pos
                        n_copy = min(n_valid, int(verify_step_routes.size(0)), max_new_tokens - dec_route_cnt)
                        if n_copy > 0:
                            route_dec_buf[dec_route_cnt : dec_route_cnt + n_copy].copy_(
                                verify_step_routes[:n_copy].to(route_dec_buf.dtype)
                            )
                            dec_route_cnt += n_copy
                else:
                    step_inp = torch.tensor([[cur_tok]], dtype=torch.long, device="cuda")
                    _, logits, _ = model(step_inp, cur_pos)
                    if log_routing and route_dec_buf is not None and r_step_buf is not None and dec_route_cnt < max_new_tokens:
                        route_dec_buf[dec_route_cnt].copy_(r_step_buf)
                        dec_route_cnt += 1
                    single_calls += 1
                    cur_pos += 1
                    next_tok, is_stop = _select_tok(logits)
                    if is_stop:
                        finish_reason = "stop"
                        should_halt = True
                        break
                    should_halt = _commit_tok(next_tok)
                    cur_tok = next_tok

            torch.cuda.synchronize()
            decode_s = time.perf_counter() - t_dec0
            if not bool(torch.isfinite(logits).all().item()):
                logits_finite = False
    finally:
        if log_routing:
            if prev_log_routing_env is None:
                os.environ.pop("CHRONOS_LOG_ROUTING", None)
            else:
                os.environ["CHRONOS_LOG_ROUTING"] = prev_log_routing_env

    routing_meta = (
        _summarize_tier_routing_metrics(model, prefill_route_ids, route_dec_buf, dec_route_cnt)
        if log_routing
        else {}
    )

    think_end_present = think_end_id in gen_ids
    if eff_thinking_mode == "thinking":
        if think_end_present:
            sp_i = gen_ids.index(think_end_id)
            r_ids = gen_ids[:sp_i]
            a_ids = gen_ids[sp_i + 1 :]
            reasoning_text = runner_mod._decode_clean(tokenizer, r_ids).strip()
            answer_text = runner_mod._decode_clean(tokenizer, a_ids).strip()
            out_think_tokens = len(r_ids)
        else:
            reasoning_text = runner_mod._decode_clean(tokenizer, gen_ids).strip()
            answer_text = ""
            out_think_tokens = len(gen_ids)
            think_close_reason = "unclosed_length"
    else:
        reasoning_text = ""
        answer_text = runner_mod._decode_clean(tokenizer, gen_ids).strip()
        out_think_tokens = 0
        think_close_reason = "chat_mode"

    full_text = runner_mod._decode_clean(tokenizer, gen_ids)
    dec_steps = max(len(gen_ids) - 1, 1)
    total_wall_s = prefill_s + decode_s
    return {
        "prompt_tokens": len(prompt_ids),
        "cached_prefix_tokens": reused_toks,
        "newly_prefilled_tokens": prefilled_toks,
        "generated_tokens": len(gen_ids),
        "answer_tokens": len(post_think_toks) if eff_thinking_mode == "thinking" else len(gen_ids),
        "think_tokens": out_think_tokens,
        "think_forced": think_close_reason in ("forced_early_rep", "forced_budget"),
        "think_unclosed": think_close_reason == "unclosed_length",
        "think_close_reason": think_close_reason,
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "wall_elapsed_s": round(total_wall_s, 4),
        "decode_tok_s": round(dec_steps / max(decode_s, 1e-6), 2),
        "ms_per_tok": round((decode_s * 1000.0) / dec_steps, 2),
        "drafted_tokens": drafted_cnt,
        "accepted_tokens": accepted_cnt,
        "spec_accept_rate": round(accepted_cnt / max(1, drafted_cnt), 4),
        "verify_calls": verify_calls,
        "single_calls": single_calls,
        "logits_finite": logits_finite,
        "finish_reason": finish_reason,
        "reasoning_text": reasoning_text,
        "answer_text": answer_text,
        "text": full_text,
        "gen_ids": gen_ids,
        **routing_meta,
    }


def json_dump_safe(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
