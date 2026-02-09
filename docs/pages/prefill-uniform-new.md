---
title: Prefill Uniform (New)
---

## Prefill Time vs Batch Size (Default Runs)

```sql uniform_time_by_batch
SELECT
    batch_size,
    (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) || '/c' || token_chunk_size AS series_label,
    AVG(prefill_total_ms) AS prefill_total_ms
FROM bench.prefill_uniform_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY batch_size, series_label
ORDER BY batch_size, series_label
```

<LineChart
    data={uniform_time_by_batch}
    x=batch_size
    xType="value"
    y=prefill_total_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Batch Size"
    yAxisTitle="Prefill Time (ms)"
    yFmt="#,##0.0"
    markers=true
    chartAreaHeight=260
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
            right: 120
        }
    }}
/>

## Prefill Time vs Chunk Size (Default Runs)

```sql uniform_time_by_chunk
SELECT
    token_chunk_size,
    (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) || '/b' || batch_size AS series_label,
    AVG(prefill_total_ms) AS prefill_total_ms
FROM bench.prefill_uniform_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY token_chunk_size, series_label
ORDER BY token_chunk_size, series_label
```

<LineChart
    data={uniform_time_by_chunk}
    x=token_chunk_size
    xType="value"
    y=prefill_total_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Chunk Size"
    yAxisTitle="Prefill Time (ms)"
    yFmt="#,##0.0"
    markers=true
    chartAreaHeight=260
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
            right: 120
        }
    }}
/>

## TTFT p50 vs Batch Size (Default Runs)

```sql uniform_ttft_by_batch
SELECT
    batch_size,
    (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) || '/c' || token_chunk_size AS series_label,
    AVG(ttft_p50_ms) AS ttft_p50_ms
FROM bench.prefill_uniform_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY batch_size, series_label
ORDER BY batch_size, series_label
```

<LineChart
    data={uniform_ttft_by_batch}
    x=batch_size
    xType="value"
    y=ttft_p50_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Batch Size"
    yAxisTitle="TTFT p50 (ms)"
    yFmt="#,##0.0"
    markers=true
    chartAreaHeight=260
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
            right: 120
        }
    }}
/>

## Raw Default-Run Data

```sql uniform_default_rows
SELECT
    run_label,
    model_name,
    backend,
    gpu_short,
    batch_size,
    token_chunk_size,
    seq_len,
    prefill_total_ms,
    ttft_min_ms,
    ttft_p50_ms,
    ttft_max_ms
FROM bench.prefill_uniform_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
ORDER BY run_label, model_name, backend, batch_size, token_chunk_size
```

<DataTable data={uniform_default_rows} rows=200 search=true>
    <Column id=run_label title="Run" />
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=gpu_short title="Adapter" />
    <Column id=batch_size title="Batch" fmt="#" />
    <Column id=token_chunk_size title="Chunk" fmt="#" />
    <Column id=seq_len title="Seq Len" fmt="#" />
    <Column id=prefill_total_ms title="Prefill (ms)" fmt="#,##0.0" />
    <Column id=ttft_min_ms title="TTFT min (ms)" fmt="#,##0.0" />
    <Column id=ttft_p50_ms title="TTFT p50 (ms)" fmt="#,##0.0" />
    <Column id=ttft_max_ms title="TTFT max (ms)" fmt="#,##0.0" />
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
