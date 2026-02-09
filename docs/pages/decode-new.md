---
title: Decode Benchmarks (New)
---

## Decode Step Latency vs Batch Size (Default Runs)

```sql decode_step_latency
SELECT
    batch_size,
    (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) || '/c' || token_chunk_size AS series_label,
    AVG(decode_total_ms / NULLIF(decode_steps, 0)) AS step_ms
FROM bench.decode_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY batch_size, series_label
ORDER BY batch_size, series_label
```

<LineChart
    data={decode_step_latency}
    x=batch_size
    xType="value"
    y=step_ms
    series=series_label
    legend=true
    yLog=true
    yLogBase=10
    xAxisTitle="Batch Size"
    yAxisTitle="Step Latency (ms)"
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

## Decode Throughput vs Batch Size (Default Runs)

```sql decode_throughput
SELECT
    batch_size,
    (CASE WHEN backend LIKE 'wgpu%' THEN 'wgpu' ELSE backend END) || '/c' || token_chunk_size AS series_label,
    AVG(decode_tok_per_s) AS tok_s
FROM bench.decode_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
GROUP BY batch_size, series_label
ORDER BY batch_size, series_label
```

<LineChart
    data={decode_throughput}
    x=batch_size
    xType="value"
    y=tok_s
    series=series_label
    legend=true
    xAxisTitle="Batch Size"
    yAxisTitle="Throughput (tok/s)"
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

```sql decode_default_rows
SELECT
    run_label,
    model_name,
    backend,
    gpu_short,
    batch_size,
    token_chunk_size,
    decode_steps,
    decode_total_ms,
    step_ms_p95,
    decode_tok_per_s
FROM bench.decode_cases
WHERE run_id IN (SELECT run_id FROM bench.runs_with_defaults WHERE is_default)
ORDER BY run_label, model_name, backend, batch_size, decode_steps
```

<DataTable data={decode_default_rows} rows=200 search=true>
    <Column id=run_label title="Run" />
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=gpu_short title="Adapter" />
    <Column id=batch_size title="Batch" fmt="#" />
    <Column id=token_chunk_size title="Chunk" fmt="#" />
    <Column id=decode_steps title="Steps" fmt="#" />
    <Column id=decode_total_ms title="Decode (ms)" fmt="#,##0.0" />
    <Column id=step_ms_p95 title="Step p95 (ms)" fmt="#,##0.0" />
    <Column id=decode_tok_per_s title="tok/s" fmt="#,##0.0" />
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
