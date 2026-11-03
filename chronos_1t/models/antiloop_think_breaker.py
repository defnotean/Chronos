import os
import torch
try:
    from vllm.triton_utils import triton, tl
except ImportError:
    import triton
    import triton.language as tl

THINK_START_ID = 128821     # <think> in DeepSeek-V4.1-Flash-1017
THINK_END_ID = 128822       # </think> in DeepSeek-V4.1-Flash-1017
TOOL_CALLS_END_ID = 128807  # <｜tool▁calls▁end｜>
EOS_ID = 1                  # <｜end▁of▁sentence｜>
EOT_ID = 128805             # <|EOT|>


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
    in_think = (last_start >= 0) & (last_start > last_end) & (raw_think_len <= think_budget + 24)
    think_closed = ((last_end >= tl.maximum(prefill_len - 1, 0)) & (last_end > last_start)) | (raw_think_len > think_budget + 24)
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

    # Penalize 6-token exact phrase loops within the current section
    m5 = v_m5 & (prev1 == t1) & (prev2 == t2) & (prev3 == t3) & (prev4 == t4) & (prev5 == t5)
    has_pen = m5 & (win_tok >= 0) & (win_tok < 129280)
    tl.atomic_add(row_logits_ptr + tl.maximum(win_tok, 0), tl.full((WIN_SIZE,), -6.0, dtype=tl.bfloat16), mask=has_pen)

    is_prior = offs < (WIN_SIZE - 1)
    a3 = is_prior & v_m2 & (win_tok == t1) & (prev1 == t2) & (prev2 == t3)
    a4 = a3 & v_m3 & (prev3 == t4)
    a6 = a4 & v_m5 & (prev4 == t5) & (prev5 == t6)

    prior_4gram = tl.sum(a4.to(tl.int32), axis=0)
    prior_6gram = tl.sum(a6.to(tl.int32), axis=0)

    stutter_4 = (gen_len >= 4) & (t1 == t2) & (t2 == t3) & (t3 == t4)
    tandem_2 = (gen_len >= 6) & (t1 == t3) & (t3 == t5) & (t2 == t4) & (t4 == t6)
    tandem_3 = (gen_len >= 6) & (t1 == t4) & (t2 == t5) & (t3 == t6)

    if (gen_len >= 3) & (t1 == t2) & (t2 == t3) & (t1 >= 0) & (t1 < 129280):
        tl.atomic_add(row_logits_ptr + t1, tl.full((), -12.0, dtype=tl.bfloat16))
    if (gen_len >= 4) & (t1 == t3) & (t2 == t4) & (t2 >= 0) & (t2 < 129280):
        tl.atomic_add(row_logits_ptr + t2, tl.full((), -12.0, dtype=tl.bfloat16))
    if tandem_3 & (t3 >= 0) & (t3 < 129280):
        tl.atomic_add(row_logits_ptr + t3, tl.full((), -12.0, dtype=tl.bfloat16))

    # Immediate clean termination right after <｜tool▁calls▁end｜> (128807)
    if t1 == 128807:
        tl.store(row_logits_ptr + 1, tl.full((), 1000.0, dtype=tl.bfloat16))

    if think_closed:
        tl.store(row_logits_ptr + THINK_START_ID, tl.full((), -1000.0, dtype=tl.bfloat16))
        tl.store(row_logits_ptr + THINK_END_ID, tl.full((), -1000.0, dtype=tl.bfloat16))
        post_think_len = eff_len - (last_end + 1)
        if (last_end >= 0) & (post_think_len < 2) & (t1 != 128807):
            tl.store(row_logits_ptr + 1, tl.full((), -1000.0, dtype=tl.bfloat16))
            tl.store(row_logits_ptr + 128805, tl.full((), -1000.0, dtype=tl.bfloat16))

    loop_in_think = (
        (prior_6gram >= 1)
        | (prior_4gram >= 3)
        | stutter_4
        | tandem_2
        | tandem_3
    )
    if in_think & (think_len >= (think_budget - 16)):
        tl.atomic_add(row_logits_ptr + THINK_END_ID, tl.full((), 5.5, dtype=tl.bfloat16))
    if in_think & (think_len >= think_budget):
        tl.atomic_add(row_logits_ptr + THINK_END_ID, tl.full((), 8.5, dtype=tl.bfloat16))
    if in_think & ((think_len >= (think_budget + 20)) | loop_in_think):
        tl.store(row_logits_ptr + THINK_END_ID, tl.full((), 1000.0, dtype=tl.bfloat16))


def apply_antiloop_and_think_breaker(
    logits: torch.Tensor,
    input_batch,
    req_states,
    s_ids: torch.Tensor,
    think_budget: int = 80,
) -> torch.Tensor:
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
        int(think_budget),
        THINK_START_ID=THINK_START_ID,
        THINK_END_ID=THINK_END_ID,
        WIN_SIZE=1024,
        num_warps=4,
    )
    return logits
