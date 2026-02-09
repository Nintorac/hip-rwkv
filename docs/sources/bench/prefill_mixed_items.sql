SELECT
    p.run_id,
    p.case_id,
    p.model_sha,
    p.backend,
    (CASE WHEN p.backend LIKE 'wgpu%' THEN 'wgpu' ELSE p.backend END) AS backend_short,
    p.batch_size,
    p.token_chunk_size,
    p.mixed_case_id,
    CASE p.mixed_case_id
        WHEN 'one_long_rest_short' THEN 'one_long'
        WHEN 'realistic_chat_scaled' THEN 'realistic'
        WHEN 'staircase_8' THEN 'staircase'
        WHEN 'bimodal_half' THEN 'bimodal'
        ELSE p.mixed_case_id
    END AS mixed_case_short,
    s.idx::INT AS batch_item_index,
    s.seq_len,
    t.ttft_ms,
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
CROSS JOIN LATERAL unnest(p.seq_lens) WITH ORDINALITY AS s(seq_len, idx)
CROSS JOIN LATERAL unnest(p.ttft_ms_local) WITH ORDINALITY AS t(ttft_ms, idx2)
WHERE p.status = 'ok'
  AND s.idx = t.idx2;
