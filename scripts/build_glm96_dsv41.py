import gc
import os
import sys
import time
from pathlib import Path
import torch
from safetensors.torch import save_file

os.environ["CHRONOS_BUILD_GLM96"] = "1"
os.environ["CHRONOS_TIER1_4BIT_SLOTS"] = "96"
os.environ["CHRONOS_TIER2_2BIT_SLOTS"] = "0"
os.environ["CHRONOS_TIER3_1BIT_SLOTS"] = "0"
os.environ["CHRONOS_OPTION_A_ALPHA"] = "0.05"
os.environ["CHRONOS_OPTION_A_PATH"] = "/home/defnotean/pb27/serving/pb27_vllm/dsv41_shared_experts_option_a_r64.pt"

sys.path.insert(0, "/home/defnotean/pb27/serving/pb27_vllm")
from chronos_1t.models.dsv41_chronos_moe import ChronosDsv41SparseMoeBlock

MODEL_DIR = Path("/home/defnotean/pb27/hf/dsv41_chronos_v75_abliterated")


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda:0")
    for l in range(40):
        out_p = MODEL_DIR / f"chronos-dsv41-glm96-layer-{l:05d}-of-00040.safetensors"
        if out_p.exists() and out_p.stat().st_size > 1_000_000_000:
            print(f"[{l+1:02d}/40] Already exists: {out_p.name} ({out_p.stat().st_size/1e9:.2f} GB)", flush=True)
            continue
        tl0 = time.time()
        shard_p = MODEL_DIR / f"chronos-dsv41-layer-{l:05d}-of-00040.safetensors"
        blk = ChronosDsv41SparseMoeBlock(
            layer_idx=l,
            shard_path=shard_p,
            device=device,
            use_fp4_slot_cache=True,
        )
        tensors = {
            "w4_gu_u8": blk.w4_gu_u8.cpu().contiguous(),
            "s4_gu_bf16": blk.s4_gu_bf16.cpu().contiguous(),
            "w4_d_u8": blk.w4_d_u8.cpu().contiguous(),
            "s4_d_bf16": blk.s4_d_bf16.cpu().contiguous(),
            "tier_slot_map": blk.tier_slot_map_gpu.cpu().contiguous(),
        }
        save_file(tensors, str(out_p))
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            for fp in (str(shard_p), str(out_p)):
                try:
                    fd = os.open(fp, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                except OSError:
                    pass
        del blk, tensors
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{l+1:02d}/40] Saved {out_p.name} in {time.time()-tl0:.2f}s (elapsed={time.time()-t0:.1f}s)", flush=True)
    print(f"ALL 40 GLM-96 SHARDS BUILT in {time.time()-t0:.2f}s!", flush=True)


if __name__ == "__main__":
    main()
