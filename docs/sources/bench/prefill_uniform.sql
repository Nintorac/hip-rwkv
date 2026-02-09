SELECT
    p.run_id, p.case_id, p.repeat_index, p.status,
    p.model_sha, p.backend, p.batch_size, p.token_chunk_size,
    p.seq_len, p.prefill_total_ms, p.total_prompt_tokens,
    p.prefill_tok_per_s, p.num_infer_calls,
    p.ttft_min_ms, p.ttft_p50_ms, p.ttft_max_ms,
    p.error_kind, p.error_message,
    m.model_name, m.model_size,
    r.human_name, r.profile, r.gpu_short, r.started_at_utc, r.git_sha
FROM prefill_uniform p
JOIN models m ON p.model_sha = m.model_sha
JOIN runs r ON p.run_id = r.run_id
WHERE p.status = 'ok'
