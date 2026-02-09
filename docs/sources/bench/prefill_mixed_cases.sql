SELECT
    p.*,
    list_max(p.seq_lens) - list_min(p.seq_lens) AS prompt_range_tokens,
    m.model_name,
    m.model_size,
    r.human_name,
    r.profile,
    r.gpu_short,
    r.started_at_utc,
    r.git_sha,
    r.git_dirty,
    r.human_name || ' · ' || r.profile || ' · ' || r.gpu_short || ' · ' || strftime(r.started_at_utc, '%Y-%m-%d %H:%M') AS run_label
FROM prefill_mixed p
JOIN models m ON p.model_sha = m.model_sha
JOIN runs r ON p.run_id = r.run_id
WHERE p.status = 'ok';
