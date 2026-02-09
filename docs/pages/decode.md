---
title: Decode Benchmarks
---

# Decode Benchmarks

```sql runs
SELECT DISTINCT run_id, human_name || ' (' || started_at_utc || ')' AS run_label
FROM bench.decode
ORDER BY started_at_utc DESC
```

```sql models
SELECT DISTINCT model_name
FROM bench.decode
ORDER BY model_name
```

```sql backends
SELECT DISTINCT backend
FROM bench.decode
ORDER BY backend
```

<Dropdown data={runs} name=run_filter value=run_id label=run_label title="Run" multiple=true selectAllByDefault=true />
<Dropdown data={models} name=model_filter value=model_name title="Model" multiple=true selectAllByDefault=true />
<Dropdown data={backends} name=backend_filter value=backend title="Backend" multiple=true selectAllByDefault=true />

```sql filtered
SELECT *
FROM bench.decode
WHERE run_id IN ${inputs.run_filter.value}
  AND model_name IN ${inputs.model_filter.value}
  AND backend IN ${inputs.backend_filter.value}
```

```sql summary
SELECT
    COUNT(*) AS total_cases,
    MEDIAN(decode_tok_per_s) AS median_tok_s,
    MAX(decode_tok_per_s) AS best_tok_s
FROM ${filtered}
```

<BigValue data={summary} value=total_cases title="Total Cases" fmt="#,##0" />
<BigValue data={summary} value=median_tok_s title="Median tok/s" fmt="#,##0.0" />
<BigValue data={summary} value=best_tok_s title="Best tok/s" fmt="#,##0.0" />

## Throughput by Configuration

```sql throughput
SELECT
    model_name || ' / ' || backend || ' / bs' || batch_size AS config,
    backend,
    AVG(decode_tok_per_s) AS avg_tok_s
FROM ${filtered}
GROUP BY model_name, backend, batch_size, config
ORDER BY avg_tok_s DESC
```

<BarChart
    data={throughput}
    x=config
    y=avg_tok_s
    series=backend
    title="Average Decode tok/s by Configuration"
    swapXY=true
    yFmt="#,##0.0"
/>

## Step Latency (p50 / p95)

```sql latency
SELECT
    model_name || ' / ' || backend || ' / bs' || batch_size AS config,
    backend,
    AVG(step_ms_p50) AS avg_p50_ms,
    AVG(step_ms_p95) AS avg_p95_ms
FROM ${filtered}
WHERE step_ms_p50 IS NOT NULL
GROUP BY model_name, backend, batch_size, config
ORDER BY avg_p50_ms ASC
```

{#if latency.length > 0}

<BarChart
    data={latency}
    x=config
    y={["avg_p50_ms", "avg_p95_ms"]}
    title="Step Latency p50 / p95 (ms)"
    swapXY=true
    yFmt="#,##0.2"
/>

{/if}

## All Results

<DataTable data={filtered} rows=50>
    <Column id=human_name title="Run" />
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=batch_size title="Batch Size" fmt="#,##0" />
    <Column id=token_chunk_size title="Chunk Size" fmt="#,##0" />
    <Column id=decode_steps title="Steps" fmt="#,##0" />
    <Column id=decode_tokens title="Tokens" fmt="#,##0" />
    <Column id=decode_tok_per_s title="tok/s" fmt="#,##0.0" />
    <Column id=decode_total_ms title="Total (ms)" fmt="#,##0.0" />
    <Column id=step_ms_p50 title="p50 (ms)" fmt="#,##0.2" />
    <Column id=step_ms_p95 title="p95 (ms)" fmt="#,##0.2" />
    <Column id=gpu_short title="GPU" />
    <Column id=git_sha title="Git SHA" />
</DataTable>
