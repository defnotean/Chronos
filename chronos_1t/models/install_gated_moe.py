import os, sys, gc, ctypes, importlib, torch, torch.nn.functional as F
import pb27_vllm.chronos_patches.antiloop_think_breaker as _atb_mod
importlib.reload(_atb_mod)

lm = getattr(self.model, 'language_model', self.model)
dev = lm.model.embed_tokens.weight.device

ha = torch.load('/pb27_vllm/chronos_healer_A.pt', map_location='cpu', weights_only=True)
hb = torch.load('/pb27_vllm/chronos_healer_B.pt', map_location='cpu', weights_only=True)
lm._chronos_healer_A = (ha['ids'].to(device=dev), ha['deltas'].to(device=dev, dtype=torch.bfloat16))
lm._chronos_healer_B = (hb['ids'].to(device=dev), hb['deltas'].to(device=dev, dtype=torch.bfloat16))
del ha, hb

target_tids = [11936, 14594, 22748, 34082, 43537, 45033, 65739, 91269]
emb_w = lm.model.embed_tokens.weight.data
lm._chronos_V_emb_trig = F.normalize(emb_w[target_tids].float(), dim=-1).to(device=dev, dtype=torch.float32)

if not hasattr(lm, '_chronos_mode_A') or not hasattr(lm, '_chronos_mode_B'):
    orig_r32 = torch.load('/pb27_vllm/shared_experts_option_a_bf16_r32.pt.orig_bak', map_location='cpu', weights_only=True)
    hyb_r32 = torch.load('/pb27_vllm/true_full288_hybrid_best_r32.pt', map_location='cpu', weights_only=True)
    r1 = torch.load('/pb27_vllm/unablit_sh_rank1_bf16.pt', map_location='cpu', weights_only=True)
    mode_A = {}
    mode_B = {}
    for l in range(3, 45):
        vt_b = orig_r32[l]['V'].T.contiguous().to(device=dev, dtype=torch.bfloat16)
        ut_b = (orig_r32[l]['U'].T.contiguous() * 0.05).to(device=dev, dtype=torch.bfloat16)
        mode_B[l] = (vt_b, ut_b)
        vt_a = hyb_r32[l]['V'].T.contiguous().to(device=dev, dtype=torch.bfloat16)
        ut_a = hyb_r32[l]['U'].T.contiguous().to(device=dev, dtype=torch.bfloat16)
        vt_a[:, 31] = r1[l]['dv_clean'].to(device=dev, dtype=torch.bfloat16)
        ut_a[31, :] = (r1[l]['u_sem'].float() * 1.5).to(device=dev, dtype=torch.bfloat16)
        mode_A[l] = (vt_a, ut_a)
    lm._chronos_mode_A = mode_A
    lm._chronos_mode_B = mode_B
    del orig_r32, hyb_r32, r1

def _set_chronos_moe_mode(runner_self, target_mode):
    lm_in = getattr(runner_self.model, 'language_model', runner_self.model)
    lm_in._chronos_cur_mode = target_mode
    runner_self.req_states._chronos_think_budget = 208 if target_mode == 'A' else 560
    r_scale = 1.0 if target_mode == 'A' else 2.5
    store = lm_in._chronos_mode_A if target_mode == 'A' else lm_in._chronos_mode_B
    h_ids, h_deltas = lm_in._chronos_healer_A if target_mode == 'A' else lm_in._chronos_healer_B
    mt_sync = os.path.getmtime('/pb27_vllm/chronos_subword_healer.pt')
    layers = lm_in.model.layers
    with torch.inference_mode():
        if hasattr(lm_in, '_lm_heal_ids'):
            lm_in._lm_heal_ids.copy_(h_ids)
            lm_in._lm_heal_deltas.copy_(h_deltas)
            lm_in._lm_heal_mtime = mt_sync
        for l in range(3, 45):
            blk = layers[l]
            blk.mlp.routed_scaling_factor = r_scale
            blk.mlp.experts.routed_scaling_factor = 1.0
            blk.mlp.experts.router.routed_scaling_factor = r_scale
            m_mod = blk.mlp.shared_experts.down_proj
            vt, ut = store[l]
            m_mod._opt_a_VT.copy_(vt)
            m_mod._opt_a_UT.copy_(ut)
        if hasattr(runner_self, 'speculator') and runner_self.speculator is not None:
            mtp_mod = getattr(runner_self.speculator.model, 'model', runner_self.speculator.model)
            l45 = mtp_mod.layers['45']
            if hasattr(l45, '_chronos_heal_ids'):
                l45._chronos_heal_ids.copy_(h_ids)
                l45._chronos_heal_deltas.copy_(h_deltas)
                l45._chronos_heal_mtime = mt_sync
            blk45 = l45.mtp_block
            blk45.mlp.routed_scaling_factor = r_scale
            blk45.mlp.experts.routed_scaling_factor = 1.0
            blk45.mlp.experts.router.routed_scaling_factor = r_scale

if not hasattr(mr.GPUModelRunner, '_orig_exec_before_gated_moe'):
    mr.GPUModelRunner._orig_exec_before_gated_moe = mr.GPUModelRunner.execute_model

def _gated_moe_execute_model(runner_self, scheduler_output, *args, **kwargs):
    try:
        new_reqs = getattr(scheduler_output, 'scheduled_new_reqs', None)
        if new_reqs:
            lm_in = getattr(runner_self.model, 'language_model', runner_self.model)
            all_ids = []
            for nr in new_reqs:
                pids = getattr(nr, 'prefill_token_ids', None) or getattr(nr, 'prompt_token_ids', None)
                if pids:
                    all_ids.extend(pids)
            if all_ids:
                t_ids = torch.tensor(all_ids, dtype=torch.long, device=dev)
                x_emb = F.normalize(lm_in.model.embed_tokens.weight.data[t_ids].float(), dim=-1)
                max_cos = (x_emb @ lm_in._chronos_V_emb_trig.T).amax().item()
                target_m = 'A' if max_cos > 0.95 else 'B'
                _set_chronos_moe_mode(runner_self, target_m)
    except Exception as e:
        with open('/pb27_vllm/gated_err.txt', 'w') as f:
            f.write(repr(e) + '\n')
    return mr.GPUModelRunner._orig_exec_before_gated_moe(runner_self, scheduler_output, *args, **kwargs)

mr.GPUModelRunner.execute_model = _gated_moe_execute_model
_set_chronos_moe_mode(self, 'B')
gc.collect()
torch.cuda.empty_cache()
try:
    ctypes.CDLL("libc.so.6").malloc_trim(0)
except Exception:
    pass
with open('/pb27_vllm/gated_status.txt', 'w') as f:
    f.write('DUAL_HEALER_TRIMMED_OK\n')
