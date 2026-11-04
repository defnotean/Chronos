import os
import traceback
import torch
from vllm.triton_utils import triton, tl

THINK_START_ID = 154841  # <think>
THINK_END_ID = 154842    # </think>


@triton.jit
def _canon_tok(t):
    is_fam_a = (t == 22748) | (t == 50780) | (t == 88852) | (t == 93873)
    is_fam_b = (t == 19687) | (t == 27963) | (t == 33891) | (t == 39745) | (t == 78996) | (t == 89147)
    is_fam_c = (t == 34082) | (t == 69413)
    return tl.where(is_fam_a, 22748, tl.where(is_fam_b, 19687, tl.where(is_fam_c, 34082, t)))


@triton.jit
def _antiloop_think_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    s_ids_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    prefill_len_ptr,
    think_budget,
    THINK_START_ID: tl.constexpr,
    THINK_END_ID: tl.constexpr,
    WIN_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    req_idx = tl.load(expanded_idx_mapping_ptr + row_idx)
    if req_idx < 0:
        return
    local_pos = tl.load(expanded_local_pos_ptr + row_idx)
    total_len = tl.load(total_len_ptr + req_idx)
    prefill_len = tl.load(prefill_len_ptr + req_idx)
    eff_len = total_len + local_pos
    if eff_len < prefill_len:
        return

    first_row = row_idx - local_pos
    offs = tl.arange(0, WIN_SIZE)
    pos = eff_len - WIN_SIZE + offs

    in_bounds = pos >= 0
    is_committed = pos < total_len
    c_pos = tl.maximum(pos, 0)
    committed_tok = tl.load(
        all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + c_pos,
        mask=in_bounds & is_committed,
        other=-1,
    )
    draft_offset = first_row + (pos - total_len) + 1
    draft_tok = tl.load(
        s_ids_ptr + draft_offset,
        mask=in_bounds & (~is_committed) & (pos < eff_len),
        other=-1,
    )
    win_tok = tl.where(is_committed, committed_tok, draft_tok)

    p_last_pos = tl.maximum(prefill_len - 1, 0)
    p_last_tok = tl.load(
        all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + p_last_pos,
        mask=prefill_len > 0,
        other=-1,
    )
    prompt_think = tl.where((prefill_len > 0) & (p_last_tok == THINK_START_ID), p_last_pos, -1)

    valid_think_scan = pos >= tl.maximum(prefill_len - 4, 0)
    win_last_start = tl.max(tl.where(valid_think_scan & (win_tok == THINK_START_ID), pos, -1), axis=0)
    last_start = tl.maximum(prompt_think, win_last_start)
    last_end = tl.max(tl.where(valid_think_scan & (win_tok == THINK_END_ID), pos, -1), axis=0)

    in_think = (last_start >= 0) & (last_start > last_end)
    think_closed = (last_end >= tl.maximum(prefill_len - 1, 0)) & (last_end > last_start)
    think_len = tl.where(in_think, eff_len - (last_start + 1), 0)

    section_start = tl.where(think_closed & (last_end >= prefill_len), last_end + 1, prefill_len)
    sec_len = eff_len - section_start

    t1 = tl.sum(tl.where(offs == WIN_SIZE - 1, win_tok, 0), axis=0)
    t2 = tl.sum(tl.where(offs == WIN_SIZE - 2, win_tok, 0), axis=0)
    t3 = tl.sum(tl.where(offs == WIN_SIZE - 3, win_tok, 0), axis=0)
    t4 = tl.sum(tl.where(offs == WIN_SIZE - 4, win_tok, 0), axis=0)
    t5 = tl.sum(tl.where(offs == WIN_SIZE - 5, win_tok, 0), axis=0)
    t6 = tl.sum(tl.where(offs == WIN_SIZE - 6, win_tok, 0), axis=0)

    ct1 = _canon_tok(t1)
    ct2 = _canon_tok(t2)
    ct3 = _canon_tok(t3)
    ct4 = _canon_tok(t4)
    ct5 = _canon_tok(t5)

    row_logits_ptr = logits_ptr + row_idx * logits_stride

    # 1) Immediate adjacent duplicate word token deterrent (only for word tokens >= 3600, excluding digits/operators/LaTeX):
    if (sec_len >= 1) & (t1 >= 3600) & (t1 < 154820):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -16.0, dtype=tl.bfloat16))

    # 2) Same-family 4-token radius + cross-family Distance-1 bare noun-noun block:
    in_fam_a_4 = ((sec_len >= 1) & (ct1 == 22748)) | ((sec_len >= 2) & (ct2 == 22748)) | ((sec_len >= 3) & (ct3 == 22748)) | ((sec_len >= 4) & (ct4 == 22748))
    in_fam_b_4 = ((sec_len >= 1) & (ct1 == 19687)) | ((sec_len >= 2) & (ct2 == 19687)) | ((sec_len >= 3) & (ct3 == 19687)) | ((sec_len >= 4) & (ct4 == 19687))
    in_fam_c_4 = ((sec_len >= 1) & (ct1 == 34082)) | ((sec_len >= 2) & (ct2 == 34082)) | ((sec_len >= 3) & (ct3 == 34082)) | ((sec_len >= 4) & (ct4 == 34082))
    t1_is_abc = (sec_len >= 1) & ((ct1 == 22748) | (ct1 == 19687) | (ct1 == 34082))

    if in_fam_a_4 | t1_is_abc:
        tl.atomic_add(row_logits_ptr + 22748, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 50780, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 88852, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 93873, tl.full((), -26.0, dtype=tl.bfloat16))

    if in_fam_b_4 | t1_is_abc:
        tl.atomic_add(row_logits_ptr + 19687, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 27963, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 33891, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 39745, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 78996, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 89147, tl.full((), -26.0, dtype=tl.bfloat16))

    if in_fam_c_4 | t1_is_abc:
        tl.atomic_add(row_logits_ptr + 34082, tl.full((), -26.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 69413, tl.full((), -26.0, dtype=tl.bfloat16))

    # 3) Any token repeated twice in a row (ct1 == ct2): block a 3rd repeat for word tokens, or 4th repeat for punctuation/formatting
    if (sec_len >= 2) & (ct1 == ct2) & (t1 >= 1000) & (t1 < 154820):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -35.0, dtype=tl.bfloat16))
    if (sec_len >= 3) & (ct1 == ct2) & (ct2 == ct3) & (t1 >= 0) & (t1 < 1000):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -35.0, dtype=tl.bfloat16))

    is_short_def = think_budget <= 256
    cw_pair = is_short_def | ((t1 >= 3600) & (t2 >= 3600))
    cw_trip = is_short_def | ((t1 >= 3600) & (t2 >= 3600) & (t3 >= 3600))

    # 4) Distance-2 tandem (A, B, A, B, A): when ct1 == ct3 and ct3 == ct5 and ct2 == ct4, block t2
    if (sec_len >= 5) & cw_pair & (ct1 == ct3) & (ct3 == ct5) & (ct2 == ct4) & (t2 >= 0) & (t2 < 154820):
        tl.atomic_add(row_logits_ptr + t2, tl.full((), -30.0, dtype=tl.bfloat16))

    # 5) Distance-3 tandem (A, B, C, A, B, C, A, B): when ct1 == ct4 and ct2 == ct5 and t3 == t6, block t3
    if (sec_len >= 6) & cw_trip & (ct1 == ct4) & (ct2 == ct5) & (t3 == t6) & (t3 >= 0) & (t3 < 154820):
        tl.atomic_add(row_logits_ptr + t3, tl.full((), -30.0, dtype=tl.bfloat16))

    pos_m1 = pos - 1
    pos_m2 = pos - 2
    pos_m3 = pos - 3
    pos_m4 = pos - 4
    pos_m5 = pos - 5

    v_m2 = pos_m2 >= section_start
    v_m3 = pos_m3 >= section_start
    v_m4 = pos_m4 >= section_start
    v_m5 = pos_m5 >= section_start

    prev1 = tl.where(
        pos_m1 < total_len,
        tl.load(all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + tl.maximum(pos_m1, 0), mask=pos_m1 >= section_start, other=-2),
        tl.load(s_ids_ptr + first_row + (pos_m1 - total_len) + 1, mask=(pos_m1 >= total_len) & (pos_m1 < eff_len), other=-2),
    )
    prev2 = tl.where(
        pos_m2 < total_len,
        tl.load(all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + tl.maximum(pos_m2, 0), mask=v_m2, other=-2),
        tl.load(s_ids_ptr + first_row + (pos_m2 - total_len) + 1, mask=v_m2 & (pos_m2 >= total_len) & (pos_m2 < eff_len), other=-2),
    )
    prev3 = tl.where(
        pos_m3 < total_len,
        tl.load(all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + tl.maximum(pos_m3, 0), mask=v_m3, other=-2),
        tl.load(s_ids_ptr + first_row + (pos_m3 - total_len) + 1, mask=v_m3 & (pos_m3 >= total_len) & (pos_m3 < eff_len), other=-2),
    )
    prev4 = tl.where(
        pos_m4 < total_len,
        tl.load(all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + tl.maximum(pos_m4, 0), mask=v_m4, other=-2),
        tl.load(s_ids_ptr + first_row + (pos_m4 - total_len) + 1, mask=v_m4 & (pos_m4 >= total_len) & (pos_m4 < eff_len), other=-2),
    )
    prev5 = tl.where(
        pos_m5 < total_len,
        tl.load(all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + tl.maximum(pos_m5, 0), mask=v_m5, other=-2),
        tl.load(s_ids_ptr + first_row + (pos_m5 - total_len) + 1, mask=v_m5 & (pos_m5 >= total_len) & (pos_m5 < eff_len), other=-2),
    )

    use_strict_ngram = is_short_def
    m2 = v_m2 & (prev1 == t1) & (prev2 == t2)
    m3 = m2 & v_m3 & (prev3 == t3)
    m4 = use_strict_ngram & m3 & v_m4 & (prev4 == t4)
    m5 = use_strict_ngram & m4 & v_m5 & (prev5 == t5)
    local_m2 = is_short_def & think_closed & m2 & (offs >= (WIN_SIZE - 24)) & ((t1 >= 3600) | (t2 >= 3600))
    local_m3 = is_short_def & think_closed & m3 & ((t1 >= 3600) | (t2 >= 3600))

    pen = tl.where(
        m5, -18.0,
        tl.where(
            m4, -10.0,
            tl.where(
                local_m3 | local_m2, -12.0,
                0.0
            )
        )
    )

    has_pen = use_strict_ngram & (m5 | m4 | local_m3 | local_m2) & (win_tok >= 0) & (win_tok < 154820)
    tl.atomic_add(row_logits_ptr + tl.maximum(win_tok, 0), pen.to(tl.bfloat16), mask=has_pen)

    is_prior = offs < (WIN_SIZE - 1)
    a3 = is_prior & v_m2 & (win_tok == t1) & (prev1 == t2) & (prev2 == t3)
    a4 = a3 & v_m3 & (prev3 == t4)
    a6 = a4 & v_m5 & (prev4 == t5) & (prev5 == t6)
    ag4 = is_prior & v_m4 & (win_tok == t1) & (
        ((prev1 == t2) & (prev3 == t4) & (prev4 == t5))
        | ((prev2 == t3) & (prev3 == t4) & (prev4 == t5))
    )

    prior_3gram = tl.sum(a3.to(tl.int32), axis=0)
    prior_4gram = tl.sum(a4.to(tl.int32), axis=0)
    prior_6gram = tl.sum(a6.to(tl.int32), axis=0)
    prior_g4 = tl.sum(ag4.to(tl.int32), axis=0)

    stutter_4 = (sec_len >= 4) & (is_short_def | (t1 >= 3600)) & (ct1 == ct2) & (ct2 == ct3) & (ct3 == ct4)
    tandem_2 = (sec_len >= 6) & cw_pair & (ct1 == ct3) & (ct3 == ct5) & (ct2 == ct4)
    tandem_3 = (sec_len >= 6) & cw_trip & (ct1 == ct4) & (ct2 == ct5) & (t3 == t6)

    loop_in_think = tl.where(
        is_short_def,
        (prior_6gram >= 1) | (prior_4gram >= 2) | (prior_3gram >= 4) | stutter_4 | tandem_2 | tandem_3,
        (prior_6gram >= 3) | (prior_4gram >= 6) | stutter_4 | tandem_2 | tandem_3,
    )

    if think_closed:
        tl.store(row_logits_ptr + THINK_START_ID, tl.full((), -1000.0, dtype=tl.bfloat16))
        post_think_len = eff_len - (last_end + 1)
        if post_think_len < 1:
            tl.store(row_logits_ptr + 154827, tl.full((), -1000.0, dtype=tl.bfloat16))
            tl.store(row_logits_ptr + 154820, tl.full((), -1000.0, dtype=tl.bfloat16))
            tl.store(row_logits_ptr + 154825, tl.full((), -1000.0, dtype=tl.bfloat16))
            tl.store(row_logits_ptr + 154829, tl.full((), -1000.0, dtype=tl.bfloat16))
        elif (is_short_def & ((prior_6gram >= 2) | (prior_g4 >= 5))) | (prior_6gram >= 4) | stutter_4 | tandem_2 | tandem_3:
            tl.store(row_logits_ptr + 154820, tl.full((), 1000.0, dtype=tl.bfloat16))
            tl.store(row_logits_ptr + 154827, tl.full((), 1000.0, dtype=tl.bfloat16))

    if in_think & (think_len >= (think_budget - 48)):
        tl.atomic_add(row_logits_ptr + THINK_END_ID, tl.full((), 8.5, dtype=tl.bfloat16))
    if in_think & ((think_len >= think_budget) | loop_in_think):
        tl.store(row_logits_ptr + THINK_END_ID, tl.full((), 1000.0, dtype=tl.bfloat16))


_BAD_UTF8_CACHE: dict[torch.device, torch.Tensor] = {}

def apply_antiloop_and_think_breaker(
    logits: torch.Tensor,
    input_batch,
    req_states,
    s_ids: torch.Tensor,
    think_budget: int = 256,
) -> torch.Tensor:
    try:
        if logits is not None and logits.shape[0] > 0:
            dev = logits.device
            bad_ids = _BAD_UTF8_CACHE.get(dev)
            if bad_ids is None:
                for cand in ('/pb27_vllm/bad_utf8_ids.pt', os.path.join(os.path.dirname(__file__), 'bad_utf8_ids.pt')):
                    if os.path.exists(cand):
                        bad_ids = torch.load(cand, map_location=dev, weights_only=True)
                        _BAD_UTF8_CACHE[dev] = bad_ids
                        break
            if bad_ids is not None:
                logits[:, bad_ids] -= 15.0
        if logits is None or input_batch is None or req_states is None or s_ids is None:
            return logits
        n_rows = logits.shape[0]
        if n_rows == 0 or s_ids.shape[0] != n_rows:
            return logits
        all_tok_gpu = req_states.all_token_ids.gpu
        _antiloop_think_kernel[(n_rows,)](
            logits,
            logits.stride(0),
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            s_ids,
            all_tok_gpu,
            all_tok_gpu.stride(0),
            req_states.total_len.gpu,
            req_states.prefill_len.gpu,
            int(getattr(req_states, '_chronos_think_budget', 2048)),
            THINK_START_ID=THINK_START_ID,
            THINK_END_ID=THINK_END_ID,
            WIN_SIZE=512,
            num_warps=4,
        )
    except Exception:
        with open('/pb27_vllm/antiloop_err.txt', 'w') as f:
            f.write(traceback.format_exc())
    return logits
