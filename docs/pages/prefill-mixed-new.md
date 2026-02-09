---
title: Prefill Mixed (New)
---

## TTFT p50 vs Prompt Size (Default Runs)

```sql mixed_ttft_vs_prompt
SELECT
    total_prompt_tokens,
    (
        CASE mixed_case_id
            WHEN 'one_long_rest_short' THEN 'one_long'
            WHEN 'realistic_chat_scaled' THEN 'realistic'
            WHEN 'staircase_8' THEN 'staircase'
            WHEN 'bimodal_half' THEN 'bimodal'
            ELSE mixed_case_id
        END
    ) || '/' || (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) AS series_label,
    AVG(ttft_p50_ms) AS ttft_p50_ms
FROM bench.prefill_mixed_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY total_prompt_tokens, series_label
ORDER BY total_prompt_tokens, series_label
```

<LineChart
    data={mixed_ttft_vs_prompt}
    x=total_prompt_tokens
    xType="value"
    y=ttft_p50_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Total Prompt Tokens"
    yAxisTitle="TTFT p50 (ms)"
    yFmt="#,##0.0"
    markers=true
    chartAreaHeight=280
    echartsOptions={{
        legend: {
            type: 'scroll',
            top: 0,
            left: 0,
            right: 140
        }
    }}
/>

## Prompt Range vs Time Range (Default Runs)

```sql mixed_prompt_range_vs_time_range
SELECT
    prompt_range_tokens,
    (
        CASE mixed_case_id
            WHEN 'one_long_rest_short' THEN 'one_long'
            WHEN 'realistic_chat_scaled' THEN 'realistic'
            WHEN 'staircase_8' THEN 'staircase'
            WHEN 'bimodal_half' THEN 'bimodal'
            ELSE mixed_case_id
        END
    ) || '/' || (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) AS series_label,
    AVG(ttft_max_ms - ttft_min_ms) AS ttft_range_ms
FROM bench.prefill_mixed_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY prompt_range_tokens, series_label
ORDER BY prompt_range_tokens, series_label
```

<LineChart
    data={mixed_prompt_range_vs_time_range}
    x=prompt_range_tokens
    xType="value"
    y=ttft_range_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Prompt Range (max seq - min seq, tokens)"
    yAxisTitle="Time Range (TTFT max - min, ms)"
    yFmt="#,##0.0"
    markers=true
    chartAreaHeight=280
    echartsOptions={{
        legend: {
            type: 'scroll',
            top: 0,
            left: 0,
            right: 140
        }
    }}
/>

## Per-Batch TTFT vs Sequence Length (Default Runs)

<ButtonGroup name=item_chunk_filter title="Chunk Size Focus" defaultValue="all" display="tabs">
    <ButtonGroupItem valueLabel="All Chunks" value="all" default />
    <ButtonGroupItem valueLabel="c32" value="32" />
    <ButtonGroupItem valueLabel="c64" value="64" />
</ButtonGroup>

```sql mixed_item_ttft_vs_seq_len
SELECT
    seq_len,
    mixed_case_short || '/' || backend_short || '/c' || token_chunk_size AS series_label,
    AVG(ttft_ms) AS ttft_ms
FROM bench.prefill_mixed_items
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
  AND (
      '${inputs.item_chunk_filter.value}' = 'undefined'
      OR '${inputs.item_chunk_filter.value}' = 'all'
      OR CAST(token_chunk_size AS VARCHAR) = '${inputs.item_chunk_filter.value}'
  )
GROUP BY seq_len, mixed_case_short, backend_short, token_chunk_size
ORDER BY seq_len, series_label
```

<LineChart
    data={mixed_item_ttft_vs_seq_len}
    x=seq_len
    xType="value"
    y=ttft_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Sequence Length (tokens)"
    yAxisTitle="TTFT per Batch Item (ms)"
    yFmt="#,##0.0"
    markers=true
    chartAreaHeight=300
    echartsOptions={{
        xAxis: {
            type: 'log',
            logBase: 2,
            minorTick: { show: true }
        },
        legend: {
            type: 'scroll',
            top: 0,
            left: 0,
            right: 140
        }
    }}
/>

## Prefill Time vs Prompt Size (Default Runs)

```sql mixed_prefill_vs_prompt
SELECT
    total_prompt_tokens,
    (
        CASE mixed_case_id
            WHEN 'one_long_rest_short' THEN 'one_long'
            WHEN 'realistic_chat_scaled' THEN 'realistic'
            WHEN 'staircase_8' THEN 'staircase'
            WHEN 'bimodal_half' THEN 'bimodal'
            ELSE mixed_case_id
        END
    ) || '/' || (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) AS series_label,
    AVG(prefill_total_ms) AS prefill_total_ms
FROM bench.prefill_mixed_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY total_prompt_tokens, series_label
ORDER BY total_prompt_tokens, series_label
```

<LineChart
    data={mixed_prefill_vs_prompt}
    x=total_prompt_tokens
    xType="value"
    y=prefill_total_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Total Prompt Tokens"
    yAxisTitle="Prefill Time (ms)"
    yFmt="#,##0.0"
    markers=true
    chartAreaHeight=280
    echartsOptions={{
        legend: {
            type: 'scroll',
            top: 0,
            left: 0,
            right: 140
        }
    }}
/>

## Raw Default-Run Data

```sql mixed_default_rows
SELECT
    run_label,
    model_name,
    backend,
    gpu_short,
    batch_size,
    mixed_case_id,
    prefill_total_ms,
    ttft_min_ms,
    ttft_p50_ms,
    ttft_max_ms,
    (ttft_max_ms - ttft_min_ms) AS ttft_range_ms
FROM bench.prefill_mixed_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
ORDER BY run_label, model_name, backend, batch_size, mixed_case_id
```

<DataTable data={mixed_default_rows} rows=200 search=true>
    <Column id=run_label title="Run" />
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=gpu_short title="Adapter" />
    <Column id=batch_size title="Batch" fmt="#" />
    <Column id=mixed_case_id title="Mixed Case" />
    <Column id=prefill_total_ms title="Prefill (ms)" fmt="#,##0.0" />
    <Column id=ttft_min_ms title="TTFT min (ms)" fmt="#,##0.0" />
    <Column id=ttft_p50_ms title="TTFT p50 (ms)" fmt="#,##0.0" />
    <Column id=ttft_max_ms title="TTFT max (ms)" fmt="#,##0.0" />
    <Column id=ttft_range_ms title="TTFT range (ms)" fmt="#,##0.0" />
</DataTable>

<Details title="Run visibility (default shown, older hidden)">

```sql run_visibility
SELECT
    run_label,
    profile,
    started_at_utc,
    CASE WHEN is_default THEN 'shown by default' ELSE 'hidden by default' END AS visibility
FROM bench.runs_with_defaults
ORDER BY started_at_utc DESC
```

<DataTable data={run_visibility} rows=20>
    <Column id=run_label title="Run" />
    <Column id=profile title="Profile" />
    <Column id=started_at_utc title="Started (UTC)" fmt="yyyy-MM-dd HH:mm" />
    <Column id=visibility title="Visibility" />
</DataTable>

</Details>
