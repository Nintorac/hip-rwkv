WITH quick_stats AS (
    SELECT COUNT(*) AS quick_count
    FROM runs
    WHERE profile = 'quick_coverage'
),
ranked AS (
    SELECT
        run_id,
        human_name,
        profile,
        gpu_short,
        started_at_utc,
        git_sha,
        git_dirty,
        human_name || ' · ' || profile || ' · ' || gpu_short || ' · ' || strftime(started_at_utc, '%Y-%m-%d %H:%M') AS run_label,
        ROW_NUMBER() OVER (ORDER BY started_at_utc DESC) AS recency_rank,
        ROW_NUMBER() OVER (PARTITION BY profile ORDER BY started_at_utc DESC) AS profile_rank
    FROM runs
)
SELECT
    run_id,
    human_name,
    profile,
    gpu_short,
    started_at_utc,
    git_sha,
    git_dirty,
    run_label,
    recency_rank,
    profile_rank,
    CASE
        WHEN quick_count > 0 THEN profile = 'quick_coverage' AND profile_rank <= 2
        ELSE recency_rank <= 2 OR profile_rank = 1
    END AS is_default
FROM ranked
CROSS JOIN quick_stats
ORDER BY started_at_utc DESC;
