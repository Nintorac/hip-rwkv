SELECT
    d.run_id, d.case_id, d.repeat_index, d.status,
    d.model_sha, d.backend, d.batch_size, d.token_chunk_size,
    d.decode_steps, d.decode_total_ms, d.decode_tokens,
    d.decode_tok_per_s, d.step_ms_p50, d.step_ms_p95,
    d.error_kind, d.error_message,
    m.model_name, m.model_size,
    r.human_name, r.profile, r.gpu_short, r.started_at_utc, r.git_sha
FROM decode d
JOIN models m ON d.model_sha = m.model_sha
JOIN runs r ON d.run_id = r.run_id
WHERE d.status = 'ok'
