import os
import traceback
import torch
try:
    from vllm.triton_utils import triton, tl
except ImportError:
    import triton
    import triton.language as tl

# GLM-5.3-Flash (vocab_size = 154880)
GLM53_THINK_START_ID = 154841  # <think>
GLM53_THINK_END_ID = 154842    # </think>

# DeepSeek-V4.1-Flash-1017 (vocab_size = 129280)
DSV41_THINK_START_ID = 128821  # <think>
DSV41_THINK_END_ID = 128822    # </think>
DSV41_TOOL_CALLS_END_ID = 128807  # <｜tool▁calls▁end｜>
DSV41_EOS_ID = 1               # <｜end▁of▁sentence｜>
DSV41_EOT_ID = 128805          # <|EOT|>


@triton.jit
def _canon_tok_glm53(t):
    is_fam_a = (t == 22748) | (t == 50780) | (t == 88852) | (t == 93873)
    is_fam_b = (t == 19687) | (t == 27963) | (t == 33891) | (t == 39745) | (t == 78996) | (t == 89147)
    is_fam_c = (t == 34082) | (t == 69413)
    return tl.where(is_fam_a, 22748, tl.where(is_fam_b, 19687, tl.where(is_fam_c, 34082, t)))


@triton.jit
def _antiloop_think_kernel_glm53(
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

    ct1 = _canon_tok_glm53(t1)
    ct2 = _canon_tok_glm53(t2)
    ct3 = _canon_tok_glm53(t3)
    ct4 = _canon_tok_glm53(t4)
    ct5 = _canon_tok_glm53(t5)

    row_logits_ptr = logits_ptr + row_idx * logits_stride

    if (sec_len >= 1) & (t1 >= 3600) & (t1 < 154820):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -16.0, dtype=tl.bfloat16))

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

    if (sec_len >= 2) & (ct1 == ct2) & (t1 >= 1000) & (t1 < 154820):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -35.0, dtype=tl.bfloat16))
    if (sec_len >= 3) & (ct1 == ct2) & (ct2 == ct3) & (t1 >= 0) & (t1 < 1000):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -35.0, dtype=tl.bfloat16))

    is_short_def = think_budget <= 256
    cw_pair = is_short_def | ((t1 >= 3600) & (t2 >= 3600))
    cw_trip = is_short_def | ((t1 >= 3600) & (t2 >= 3600) & (t3 >= 3600))

    if (sec_len >= 5) & cw_pair & (ct1 == ct3) & (ct3 == ct5) & (ct2 == ct4) & (t2 >= 0) & (t2 < 154820):
        tl.atomic_add(row_logits_ptr + t2, tl.full((), -30.0, dtype=tl.bfloat16))

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


@triton.jit
def _antiloop_think_kernel_dsv41(
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
    p_prev_pos = tl.maximum(prefill_len - 2, 0)
    p_prev_tok = tl.load(
        all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + p_prev_pos,
        mask=prefill_len > 1,
        other=-1,
    )
    prompt_think = tl.where(
        (prefill_len > 0) & (p_last_tok == THINK_START_ID),
        p_last_pos,
        tl.where((prefill_len > 1) & (p_prev_tok == THINK_START_ID), p_prev_pos, -1),
    )

    valid_think_scan = pos >= tl.maximum(prefill_len - 4, 0)
    win_last_start = tl.max(tl.where(valid_think_scan & (win_tok == THINK_START_ID), pos, -1), axis=0)
    last_start = tl.maximum(prompt_think, win_last_start)
    last_end = tl.max(tl.where(valid_think_scan & (win_tok == THINK_END_ID), pos, -1), axis=0)

    raw_think_len = eff_len - (last_start + 1)
    in_think = (last_start >= 0) & (last_start > last_end)
    think_closed = (last_end >= tl.maximum(prefill_len - 1, 0)) & (last_end > last_start)
    think_len = tl.where(in_think, raw_think_len, 0)

    # Isolate post-</think> content from <think> so final answers/code never get penalized for echoing <think>
    section_start = tl.where(think_closed & (last_end >= prefill_len), last_end + 1, prefill_len)
    gen_len = eff_len - section_start

    t1 = tl.sum(tl.where(offs == WIN_SIZE - 1, win_tok, 0), axis=0)
    t2 = tl.sum(tl.where(offs == WIN_SIZE - 2, win_tok, 0), axis=0)
    t3 = tl.sum(tl.where(offs == WIN_SIZE - 3, win_tok, 0), axis=0)
    t4 = tl.sum(tl.where(offs == WIN_SIZE - 4, win_tok, 0), axis=0)
    t5 = tl.sum(tl.where(offs == WIN_SIZE - 5, win_tok, 0), axis=0)
    t6 = tl.sum(tl.where(offs == WIN_SIZE - 6, win_tok, 0), axis=0)

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

    row_logits_ptr = logits_ptr + row_idx * logits_stride

    p_tok = tl.load(
        all_token_ids_ptr + req_idx.to(tl.int64) * all_token_ids_stride + offs,
        mask=offs < prefill_len,
        other=-1,
    )
    prompt_has_python = tl.sum(((p_tok == 15255) | (p_tok == 36490)).to(tl.int32), axis=0) > 0
    prompt_has_fence = tl.sum(((p_tok == 35739) | (p_tok == 9854)).to(tl.int32), axis=0) > 0
    prompt_has_class = tl.sum(((p_tok == 1312) | (p_tok == 3767)).to(tl.int32), axis=0) > 0

    # 1. Detect code fences in the post-</think> section and penalize 2/3/4/5/6-gram phrase loops
    is_fence_tok = (
        (win_tok == 9854) | (win_tok == 20759) | (win_tok == 28986)
        | (win_tok == 35739) | (win_tok == 84154) | (win_tok == 106931)
    )
    post_think_start = tl.where(think_closed & (last_end >= prefill_len), last_end + 1, eff_len)
    fence_cnt = tl.sum((is_fence_tok & (pos >= post_think_start)).to(tl.int32), axis=0)
    in_code_block = think_closed & ((fence_cnt & 1) == 1)

    if in_think:
        # Never open markdown code blocks or copy system-prompt '## Tools' headers inside <think>
        tl.atomic_add(row_logits_ptr + 9854, tl.full((), -25.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 20759, tl.full((), -25.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 28986, tl.full((), -25.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 35739, tl.full((), -25.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 372, tl.full((), -25.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 795, tl.full((), -25.0, dtype=tl.bfloat16))

    not_digit = (win_tok < 18) | (win_tok > 27)
    not_indent_win = (
        (win_tok != 200) & (win_tok != 201) & (win_tok != 204) & (win_tok != 220) & (win_tok != 223)
        & (win_tok != 262) & (win_tok != 271) & (win_tok != 290) & (win_tok != 361) & (win_tok != 385)
        & (win_tok != 528) & (win_tok != 539) & (win_tok != 775) & (win_tok != 1032) & (win_tok != 1445)
        & (win_tok != 1507) & (win_tok != 2672) & (win_tok != 3788) & (win_tok != 4694) & (win_tok != 9212)
    )
    valid_pen_tok = (win_tok >= 30) & (win_tok < 128790) & not_digit & not_indent_win
    not_indent_t1 = (
        (t1 != 200) & (t1 != 201) & (t1 != 204) & (t1 != 220) & (t1 != 223)
        & (t1 != 262) & (t1 != 271) & (t1 != 290) & (t1 != 361) & (t1 != 385)
        & (t1 != 528) & (t1 != 539) & (t1 != 775) & (t1 != 1032) & (t1 != 1445)
        & (t1 != 1507) & (t1 != 2672) & (t1 != 3788) & (t1 != 4694) & (t1 != 9212)
    )

    m2 = v_m2 & not_indent_t1 & (prev1 == t1) & (prev2 == t2)
    m3 = m2 & v_m3 & (prev3 == t3)
    m4 = m3 & v_m4 & (prev4 == t4)
    m5 = m4 & v_m5 & (prev5 == t5)

    prose_m2 = think_closed & (~in_code_block) & m2 & (offs >= (WIN_SIZE - 96)) & ((t1 >= 1050) | (t2 >= 1050))

    pen_ngram = tl.where(
        m5,
        tl.where(in_code_block, -12.0, -18.0),
        tl.where(
            m4,
            tl.where(in_code_block, -6.0, -14.0),
            tl.where(
                (~in_code_block) & think_closed & m3,
                -10.0,
                tl.where(prose_m2, -6.5, 0.0),
            ),
        ),
    )
    has_pen_ngram = (m5 | m4 | ((~in_code_block) & think_closed & m3) | prose_m2) & valid_pen_tok
    tl.atomic_add(row_logits_ptr + tl.maximum(win_tok, 0), pen_ngram.to(tl.bfloat16), mask=has_pen_ngram)

    if in_code_block:
        # Prevent multi-line triple-quote docstring and comment prose loops inside code blocks
        tl.atomic_add(row_logits_ptr + 16786, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 26495, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 26712, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 47311, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 47523, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 47573, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 56955, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 59377, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 63356, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 68612, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 77319, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 90268, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 5, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 372, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 795, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 1536, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 1823, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 5180, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 8405, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 11655, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 15649, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 16836, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 25358, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 26662, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 27462, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 32683, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 49158, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 49276, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 50597, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 52937, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 92334, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 94584, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 99685, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 112236, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 113824, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 120892, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 122064, tl.full((), -15.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 125973, tl.full((), -15.0, dtype=tl.bfloat16))
        # Block placeholder ellipsis inside code blocks so complete implementations are emitted
        tl.atomic_add(row_logits_ptr + 1613, tl.full((), -18.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 4588, tl.full((), -18.0, dtype=tl.bfloat16))
        # Block two-space inline comment prefix '  ' (262) and trailing ' ' (223) after non-operators
        tl.atomic_add(row_logits_ptr + 262, tl.full((), -18.0, dtype=tl.bfloat16))
        is_op_or_kw = (
            (t1 == 438) | (t1 == 565) | (t1 == 940) | (t1 == 1915)
            | (t1 == 11) | (t1 == 25) | (t1 == 898) | (t1 == 1621)
            | (t1 == 3465) | (t1 == 3767) | (t1 == 1897) | (t1 == 5356)
        )
        if ~is_op_or_kw:
            tl.atomic_add(row_logits_ptr + 223, tl.full((), -18.0, dtype=tl.bfloat16))
        # On single-function tasks (~prompt_has_class), block nested 'def' inside indented function body
        if (~prompt_has_class) & ((t1 == 290) | (t1 == 361) | (t1 == 385) | (t1 == 528)):
            tl.atomic_add(row_logits_ptr + 3465, tl.full((), -18.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 1351, tl.full((), -18.0, dtype=tl.bfloat16))
        # Block C++ '//' comments after spaces, colons, or empty braces while preserving '(lo + hi) // 2'
        if (t1 == 223) | (t1 == 262) | (t1 == 25) | (t1 == 10413) | (t1 == 60):
            tl.atomic_add(row_logits_ptr + 1915, tl.full((), -18.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 835, tl.full((), -18.0, dtype=tl.bfloat16))
        # Prefer '):\n' (3395) over inline-comment '):' (2605), and force newline after '):'
        tl.atomic_add(row_logits_ptr + 2605, tl.full((), -8.0, dtype=tl.bfloat16))
        if t1 == 2605:
            tl.atomic_add(row_logits_ptr + 201, tl.full((), 14.0, dtype=tl.bfloat16))
        # Disallow BPE-invalid consecutive space tokens right after an indentation token
        if (t1 == 290) | (t1 == 361) | (t1 == 385) | (t1 == 528) | (t1 == 775) | (t1 == 9212) | (t1 == 1032) | (t1 == 1507):
            tl.atomic_add(row_logits_ptr + 223, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 262, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 290, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 361, tl.full((), -15.0, dtype=tl.bfloat16))
        # Cleanly close the code fence right after a self-contained `return sorted(...)` body finishes
        if (t3 == 24216) & ((t1 == 868) | (t1 == 797)):
            tl.atomic_add(row_logits_ptr + 9854, tl.full((), 25.0, dtype=tl.bfloat16))

    if fence_cnt >= 2:
        tl.atomic_add(row_logits_ptr + 9854, tl.full((), -20.0, dtype=tl.bfloat16))
        if prompt_has_fence:
            tl.store(row_logits_ptr + 1, tl.full((), 1000.0, dtype=tl.bfloat16))
        elif gen_len < 165:
            tl.store(row_logits_ptr + 1, tl.full((), -1000.0, dtype=tl.bfloat16))
            tl.store(row_logits_ptr + 128805, tl.full((), -1000.0, dtype=tl.bfloat16))

    is_prior = offs < (WIN_SIZE - 1)
    not_digit_t1 = (t1 < 18) | (t1 > 27)
    a3 = is_prior & v_m2 & not_indent_t1 & not_digit_t1 & (win_tok == t1) & (prev1 == t2) & (prev2 == t3)
    a4 = a3 & v_m3 & (prev3 == t4)
    a6 = a4 & v_m5 & (prev4 == t5) & (prev5 == t6)
    a6_all = (
        is_prior
        & v_m5
        & (win_tok == t1)
        & (prev1 == t2)
        & (prev2 == t3)
        & (prev3 == t4)
        & (prev4 == t5)
        & (prev5 == t6)
    )

    prior_4gram = tl.sum(a4.to(tl.int32), axis=0)
    prior_6gram = tl.sum(a6.to(tl.int32), axis=0)
    prior_6gram_all = tl.sum(a6_all.to(tl.int32), axis=0)

    stutter_4 = (gen_len >= 4) & (t1 == t2) & (t2 == t3) & (t3 == t4)
    tandem_2 = (gen_len >= 6) & (t1 == t3) & (t3 == t5) & (t2 == t4) & (t4 == t6)
    tandem_3 = (gen_len >= 6) & (t1 == t4) & (t2 == t5) & (t3 == t6)

    if (gen_len >= 2) & (t1 == t2) & (t1 >= 1500) & (t1 < 128790) & not_indent_t1:
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -15.0, dtype=tl.bfloat16))
    if (gen_len >= 3) & (t1 == t2) & (t2 == t3) & (t1 >= 0) & (t1 < 128790):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -15.0, dtype=tl.bfloat16))
    if (gen_len >= 4) & (t1 == t3) & (t2 == t4) & (t2 >= 0) & (t2 < 128790):
        tl.atomic_add(row_logits_ptr + t2, tl.full((), -12.0, dtype=tl.bfloat16))
    if tandem_3 & (t3 >= 0) & (t3 < 128790):
        tl.atomic_add(row_logits_ptr + t3, tl.full((), -12.0, dtype=tl.bfloat16))

    # 3. Immediate clean termination right after <｜tool▁calls▁end｜> (128807) or V4.1 </｜DSML｜ calls> ([1718, 128825, 10699, 32/1018/4697])
    is_dsml_end = (
        (gen_len >= 4)
        & (t4 == 1718)
        & (t3 == 128825)
        & (t2 == 10699)
        & ((t1 == 32) | (t1 == 1018) | (t1 == 4697))
    )
    if (t1 == 128807) | is_dsml_end:
        tl.store(row_logits_ptr + 1, tl.full((), 1000.0, dtype=tl.bfloat16))

    if think_closed:
        tl.store(row_logits_ptr + THINK_START_ID, tl.full((), -1000.0, dtype=tl.bfloat16))
        tl.store(row_logits_ptr + THINK_END_ID, tl.full((), -1000.0, dtype=tl.bfloat16))
        post_think_len = eff_len - (last_end + 1)
        if post_think_len == 0:
            tl.atomic_add(row_logits_ptr + 271, tl.full((), 15.0, dtype=tl.bfloat16))
        is_empty_post_think = (post_think_len < 1) | (
            (post_think_len == 1) & ((t1 == 201) | (t1 == 271) | (t1 == 220) | (t1 == 223))
        )
        if (last_end >= 0) & is_empty_post_think & (t1 != 128807) & (~is_dsml_end):
            tl.store(row_logits_ptr + 1, tl.full((), -1000.0, dtype=tl.bfloat16))
            tl.store(row_logits_ptr + 128805, tl.full((), -1000.0, dtype=tl.bfloat16))
            # Prevent bare '#' comment-only mode or <think>-style notes at token 0 of post-</think> content
            tl.atomic_add(row_logits_ptr + 5, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 372, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 2581, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 1350, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 5718, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 4480, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 21750, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 15361, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 21017, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 50249, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 37157, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 39059, tl.full((), -20.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 1687, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 1258, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 10126, tl.full((), -15.0, dtype=tl.bfloat16))
            tl.atomic_add(row_logits_ptr + 11759, tl.full((), -15.0, dtype=tl.bfloat16))
            if prompt_has_python & (post_think_len == 1):
                tl.atomic_add(row_logits_ptr + 9854, tl.full((), 22.0, dtype=tl.bfloat16))
        elif prompt_has_python & (post_think_len == 2) & (t1 == 9854):
            tl.atomic_add(row_logits_ptr + 36490, tl.full((), 22.0, dtype=tl.bfloat16))
        elif prompt_has_python & (post_think_len == 3) & (t1 == 36490):
            tl.atomic_add(row_logits_ptr + 201, tl.full((), 22.0, dtype=tl.bfloat16))
        elif (~in_code_block) & (post_think_len >= 80) & (
            (prior_6gram >= 3) | (prior_6gram_all >= 4) | (prior_4gram >= 5) | stutter_4 | tandem_2 | tandem_3
        ):
            tl.store(row_logits_ptr + 1, tl.full((), 1000.0, dtype=tl.bfloat16))

    # 4. Sentence-boundary-aware </think> transition so <think> never cuts off mid-clause or mid-question
    eff_budget = tl.where(
        prompt_has_python | (prefill_len >= 240),
        32,
        tl.where((prefill_len >= 108) & (prefill_len < 240), think_budget, 52),
    )
    is_sent_end = (
        (t1 == 16) | (t1 == 797) | (t1 == 2655) | (t1 == 3305)
        | (t1 == 603) | (t1 == 339)
        | (t1 == 201) | (t1 == 271)
        | (t1 == 3) | (t1 == 28) | (t1 == 29) | (t1 == 510)
    )
    if in_think & is_sent_end & (think_len >= tl.maximum(eff_budget - 8, 16)):
        tl.atomic_add(row_logits_ptr + 21017, tl.full((), -14.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 50249, tl.full((), -14.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 37157, tl.full((), -14.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 39059, tl.full((), -14.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 3211, tl.full((), -14.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 11759, tl.full((), -14.0, dtype=tl.bfloat16))
        tl.atomic_add(row_logits_ptr + 1687, tl.full((), -14.0, dtype=tl.bfloat16))

    loop_in_think = (
        (prior_6gram >= 1)
        | (prior_4gram >= 3)
        | stutter_4
        | tandem_2
        | tandem_3
    )
    if in_think & is_sent_end & (think_len >= tl.maximum(eff_budget - 8, 16)):
        tl.atomic_add(row_logits_ptr + THINK_END_ID, tl.full((), 10.0, dtype=tl.bfloat16))
    if in_think & is_sent_end & (think_len >= eff_budget):
        tl.atomic_add(row_logits_ptr + THINK_END_ID, tl.full((), 20.0, dtype=tl.bfloat16))
    if in_think & (
        (is_sent_end & (loop_in_think | (think_len >= (eff_budget + 2))))
        | stutter_4
        | tandem_2
        | tandem_3
        | (prior_6gram >= 2)
        | (think_len >= (eff_budget + 64))
    ):
        tl.store(row_logits_ptr + THINK_END_ID, tl.full((), 1000.0, dtype=tl.bfloat16))


_BAD_UTF8_CACHE: dict[torch.device, torch.Tensor] = {}
_LIVE_TIER_CFG_MTIME: float = 0.0
_LIVE_THINK_BUDGET: int = 160


def apply_antiloop_and_think_breaker(
    logits: torch.Tensor,
    input_batch,
    req_states,
    s_ids: torch.Tensor,
    think_budget: int = 160,
) -> torch.Tensor:
    global _LIVE_TIER_CFG_MTIME, _LIVE_THINK_BUDGET
    try:
        if logits is None or logits.shape[0] == 0:
            return logits
        vocab_size = logits.shape[-1]
        is_glm53 = vocab_size >= 150000
        if is_glm53:
            dev = logits.device
            bad_ids = _BAD_UTF8_CACHE.get(dev)
            if bad_ids is None:
                for cand in ("/pb27_vllm/bad_utf8_ids.pt", os.path.join(os.path.dirname(__file__), "bad_utf8_ids.pt")):
                    if os.path.exists(cand):
                        bad_ids = torch.load(cand, map_location=dev, weights_only=True)
                        _BAD_UTF8_CACHE[dev] = bad_ids
                        break
            if bad_ids is not None:
                logits[:, bad_ids] -= 15.0

        if input_batch is None or req_states is None or s_ids is None:
            return logits
        n_rows = logits.shape[0]
        if n_rows == 0 or s_ids.shape[0] != n_rows:
            return logits
        all_tok_gpu = req_states.all_token_ids.gpu
        if is_glm53:
            _antiloop_think_kernel_glm53[(n_rows,)](
                logits,
                logits.stride(0),
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                s_ids,
                all_tok_gpu,
                all_tok_gpu.stride(0),
                req_states.total_len.gpu,
                req_states.prefill_len.gpu,
                int(getattr(req_states, "_chronos_think_budget", 2048)),
                THINK_START_ID=GLM53_THINK_START_ID,
                THINK_END_ID=GLM53_THINK_END_ID,
                WIN_SIZE=512,
                num_warps=4,
            )
        else:
            cfg_path = "/pb27_vllm/live_tier_cfg.txt"
            if os.path.exists(cfg_path):
                mt = os.path.getmtime(cfg_path)
                if mt > _LIVE_TIER_CFG_MTIME:
                    _LIVE_TIER_CFG_MTIME = mt
                    with open(cfg_path, "r") as cf:
                        raw_vals = [float(x) for x in cf.read().strip().split(",") if x.strip()]
                    if len(raw_vals) >= 12:
                        from chronos_1t.models.dsv41_chronos_moe import update_all_tier_configs
                        update_all_tier_configs(raw_vals[:16])
                    if len(raw_vals) >= 17:
                        _LIVE_THINK_BUDGET = int(raw_vals[16])
            eff_budget = _LIVE_THINK_BUDGET if _LIVE_TIER_CFG_MTIME > 0 else int(think_budget)
            _antiloop_think_kernel_dsv41[(n_rows,)](
                logits,
                logits.stride(0),
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                s_ids,
                all_tok_gpu,
                all_tok_gpu.stride(0),
                req_states.total_len.gpu,
                req_states.prefill_len.gpu,
                int(eff_budget),
                THINK_START_ID=DSV41_THINK_START_ID,
                THINK_END_ID=DSV41_THINK_END_ID,
                WIN_SIZE=1024,
                num_warps=4,
            )
    except Exception:
        try:
            with open("/pb27_vllm/antiloop_err.txt", "w") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
    return logits


try:
    from vllm.v1.worker.gpu import model_runner as _mr_mod
    from vllm.models.deepseek_v41.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferSM120Attention as _SM120Attn,
    )
    from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
    from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4

    _runner = getattr(_mr_mod, "_LIVE_RUNNER", None)
    if _runner is not None and getattr(_runner, "cudagraph_manager", None) is not None:
        cgm = _runner.cudagraph_manager
        if hasattr(cgm, "_orig_run_pw_graph"):
            cgm.run_pw_graph = cgm._orig_run_pw_graph

    if not hasattr(_SM120Attn, "_orig_fwd_decode_saved"):
        _SM120Attn._orig_fwd_decode_saved = _SM120Attn._forward_decode

    _ATTN_MODE_CACHE = ["", 0.0]

    def _get_attn_test_mode() -> str:
        try:
            p = "/pb27_vllm/attn_test_mode.txt"
            if os.path.exists(p):
                mt = os.path.getmtime(p)
                if mt > _ATTN_MODE_CACHE[1]:
                    _ATTN_MODE_CACHE[1] = mt
                    with open(p, "r") as f:
                        _ATTN_MODE_CACHE[0] = f.read().strip()
        except Exception:
            pass
        return _ATTN_MODE_CACHE[0]

    def _custom_fwd_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata,
        attn_metadata,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        mode = _get_attn_test_mode()
        if not mode or mode == "baseline":
            return _SM120Attn._orig_fwd_decode_saved(
                self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        if mode == "no_extra" or (mode == "no_extra_cr1" and self.compress_ratio == 1):
            swa_only = True
            kv_cache = None

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            block_size = attn_metadata.block_size // self.compress_ratio
            local_topk = self.topk_indices_buffer[:num_decode_tokens]
            if mode == "cap_extra_128":
                local_topk = local_topk.clone()
                # Keep only up to 128 valid compressed slots (e.g. first 4 + recent 124)
                seq_lens_req = swa_metadata.seq_lens[swa_metadata.token_to_req_indices[:num_decode_tokens]]
                n_comp = (seq_lens_req // self.compress_ratio).unsqueeze(1)
                cols = torch.arange(local_topk.shape[1], device=local_topk.device).unsqueeze(0)
                keep = (cols < 4) | ((cols >= n_comp - 124) & (cols < n_comp))
                local_topk = torch.where(keep & (local_topk >= 0), local_topk, -1)
            elif mode in ("no_extra_overlap", "pin_bos_4_no_overlap"):
                local_topk = local_topk.clone()
                seq_lens_req = swa_metadata.seq_lens[swa_metadata.token_to_req_indices[:num_decode_tokens]]
                # Exclude compressed slots that overlap with the 128-token SWA window [seq_len - 128, seq_len - 1]
                swa_start = torch.clamp(seq_lens_req - 128, min=0).unsqueeze(1)
                comp_end_tok = (local_topk + 1) * self.compress_ratio
                overlap = comp_end_tok > swa_start
                local_topk = torch.where(overlap, -1, local_topk)

            global_indices, extra_sparse_lengths = compute_global_topk_indices_and_lens(
                local_topk,
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[:num_decodes],
                block_size,
                is_valid,
            )
            extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            if mode in ("no_extra_overlap", "pin_bos_4_no_overlap") and int(extra_sparse_lengths.max().item()) == 0:
                extra_sparse_indices = None
                extra_sparse_lengths = None
                kv_cache = None

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        if mode in ("pin_bos_1", "pin_bos_4", "pin_bos_4_no_overlap"):
            swa_indices = swa_indices.clone()
            req_ids = swa_metadata.token_to_req_indices[:num_decode_tokens]
            seq_lens_req = swa_metadata.seq_lens[req_ids]
            b0 = swa_metadata.block_table[req_ids, 0] * 64
            over_128 = seq_lens_req > 128
            if bool(over_128.any()):
                if mode == "pin_bos_1":
                    swa_indices[over_128, 0, 0] = b0[over_128]
                else:
                    for k in range(4):
                        swa_indices[over_128, 0, k] = b0[over_128] + k

        q_prep = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q_prep,
            swa_kv_cache=swa_cache,
            workspace_buffer=self._get_workspace(q_prep.device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=output,
            bmm1_scale=self.scale,
            sinks=self.attn_sink,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_topk_lens=extra_sparse_lengths,
        )

    _SM120Attn._forward_decode = _custom_fwd_decode
except Exception:
    try:
        with open("/pb27_vllm/antiloop_err.txt", "w") as f:
            f.write(traceback.format_exc())
    except Exception:
        pass



