SELECT
    d.*,
    m.model_name,
    m.model_size,
    r.human_name,
    r.profile,
    r.gpu_short,
    r.started_at_utc,
    r.git_sha,
    r.git_dirty,
    r.human_name || ' · ' || r.profile || ' · ' || r.gpu_short || ' · ' || strftime(r.started_at_utc, '%Y-%m-%d %H:%M') AS run_label
FROM decode d
JOIN models m ON d.model_sha = m.model_sha
JOIN runs r ON d.run_id = r.run_id
WHERE d.status = 'ok';
