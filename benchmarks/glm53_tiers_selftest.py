#!/usr/bin/env python3
"""Self-test for `chronos_1t.models.glm53_tiers` (PyTorch reference vs Triton kernels)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chronos_1t.models import glm53_tiers as T  # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    ok = True
    n, k = 512, 768
    w = torch.randn(3, n, k, device=dev) * 0.02
    print(f"device={dev} triton={T.triton_enabled(dev)}")
    for tier, name, lo in ((T.TIER_T1, "T1 fp4", 0.99), (T.TIER_T2, "T2 lm2", 0.93), (T.TIER_T3, "T3 sign1", 0.78)):
        q, s = T.PACKERS[tier](w)
        r = T.UNPACKERS[tier](q, s, dtype=torch.float32)
        c = cos(r, w)
        bpw = (q.numel() * 8 + s.numel() * 16) / w.numel()
        print(f"{name}: cos={c:.5f} bpw={bpw:.3f} q={tuple(q.shape)} s={tuple(s.shape)}")
        ok &= c > lo

    # FP8 block round-trip incl. non-multiple-of-128 rows (kv_a_proj_with_mqa has 576 rows)
    wa = torch.randn(576, 640, device=dev) * 0.03
    q8, s8 = T.quantize_fp8_block(wa)
    c8 = cos(T.dequant_fp8_block(q8, s8, dtype=torch.float32), wa)
    print(f"fp8 block 576x640 cos={c8:.5f} scale={tuple(s8.shape)}")
    ok &= c8 > 0.999

    if T.triton_enabled(dev):
        # Pairwise tier GEMV against reference
        e = 12
        tier_of = [1, 1, 2, 2, 3, 3, 3, 0, 4, 4, 2, 1]
        bufs = {}
        tier_map = torch.zeros(e, dtype=torch.int32)
        slots = {1: 0, 2: 0, 3: 0, 4: 0}
        wt = torch.randn(e, n, k, device=dev) * 0.02
        per_tier: dict[int, list[torch.Tensor]] = {1: [], 2: [], 3: [], 4: []}
        for ei, t in enumerate(tier_of):
            if t == 0:
                continue
            per_tier[t].append(wt[ei])
            tier_map[ei] = (t << T.TIER_SHIFT) | slots[t]
            slots[t] += 1
        for t, lst in per_tier.items():
            fmt = T.TIER_T3 if t == T.TIER_STAGE else t
            bufs[t] = T.PACKERS[fmt](torch.stack(lst))
        tier_map = tier_map.to(dev)
        p = 40
        ids = torch.randint(0, e, (p,), device=dev, dtype=torch.int32)
        x = torch.randn(p, k, device=dev, dtype=torch.bfloat16)
        ref = T.tier_gemv_torch(x, ids, tier_map, bufs, n)
        out = T.tier_gemv(x, ids, tier_map, bufs, n)
        torch.cuda.synchronize()
        err = (out - ref).abs().max().item()
        c = cos(out, ref)
        absent_zero = bool((out[ids == 7].abs().max() == 0).item()) if (ids == 7).any() else True
        print(f"tier_gemv vs torch: cos={c:.6f} max_abs={err:.3e} absent->0={absent_zero}")
        ok &= c > 0.9999 and absent_zero

        # FP8 GEMV (decode) vs dequant GEMM
        wq, ws = T.quantize_fp8_block(torch.randn(1000, 704, device=dev) * 0.02)
        xb = torch.randn(3, 704, device=dev, dtype=torch.bfloat16)
        y_tri = T.fp8_linear(xb, wq, ws, out_dtype=torch.float32)
        y_ref = xb.float() @ T.dequant_fp8_block(wq, ws, dtype=torch.float32).T
        c = cos(y_tri, y_ref)
        print(f"fp8 gemv vs dequant: cos={c:.6f} max_abs={(y_tri - y_ref).abs().max().item():.3e}")
        ok &= c > 0.9999

        # Timing at GLM-5.3 expert shape (gate_up 4096x6144, 8 pairs)
        for tier in (T.TIER_T1, T.TIER_T2, T.TIER_T3):
            wbig = torch.randn(8, 4096, 6144, device=dev, dtype=torch.bfloat16) * 0.02
            qb, sb = T.PACKERS[tier](wbig)
            del wbig
            tm = torch.tensor([(tier << T.TIER_SHIFT) | i for i in range(8)], dtype=torch.int32, device=dev)
            ids8 = torch.arange(8, dtype=torch.int32, device=dev)
            x8 = torch.randn(8, 6144, device=dev, dtype=torch.bfloat16)
            for bn, bk in ((16, 32), (16, 64), (32, 32), (32, 64), (8, 128), (16, 128)):
                T.tier_gemv(x8, ids8, tm, {tier: (qb, sb)}, 4096, block_n=bn, block_kb=bk)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    T.tier_gemv(x8, ids8, tm, {tier: (qb, sb)}, 4096, block_n=bn, block_kb=bk)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / 20
                gbs = (qb.numel() + sb.numel() * 2) / dt / 1e9
                print(f"  tier {tier} gate_up x8 BN={bn:3d} BK={bk:3d}: {dt * 1e3:.3f} ms ({gbs:.0f} GB/s)")
            del qb, sb
            torch.cuda.empty_cache()
    print("ALL OK" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
