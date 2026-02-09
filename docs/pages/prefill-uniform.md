---
title: Prefill Uniform Benchmarks
---

# Prefill Uniform Benchmarks

```sql runs
SELECT DISTINCT run_id, human_name || ' (' || started_at_utc || ')' AS run_label
FROM bench.prefill_uniform
ORDER BY started_at_utc DESC
```

<Dropdown data={runs} name=run_filter value=run_id label=run_label title="Run" multiple=true selectAllByDefault=true />

```sql prefill_0_1b
SELECT batch_size, backend, AVG(prefill_tok_per_s) AS avg_tok_s, AVG(prefill_total_ms) AS avg_total_ms, AVG(ttft_p50_ms) AS avg_ttft
FROM bench.prefill_uniform
WHERE model_name = 'rwkv7_g1a_0.1b' AND run_id IN ${inputs.run_filter.value}
GROUP BY batch_size, backend
ORDER BY batch_size
```

```sql prefill_2_9b
SELECT batch_size, backend, AVG(prefill_tok_per_s) AS avg_tok_s, AVG(prefill_total_ms) AS avg_total_ms, AVG(ttft_p50_ms) AS avg_ttft
FROM bench.prefill_uniform
WHERE model_name = 'rwkv7_g1c_2.9b' AND run_id IN ${inputs.run_filter.value}
GROUP BY batch_size, backend
ORDER BY batch_size
```

## Throughput by Batch Size

<Grid cols=2>
<div>

### 0.1b

<LineChart
    data={prefill_0_1b}
    x=batch_size
    y=avg_tok_s
    series=backend
    title="rwkv7_g1a_0.1b"
    xAxisTitle="Batch Size"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</div>
<div>

### 2.9b

<LineChart
    data={prefill_2_9b}
    x=batch_size
    y=avg_tok_s
    series=backend
    title="rwkv7_g1c_2.9b"
    xAxisTitle="Batch Size"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</div>
</Grid>

## TTFT by Batch Size

<Grid cols=2>
<div>

### 0.1b

<LineChart
    data={prefill_0_1b}
    x=batch_size
    y=avg_ttft
    series=backend
    title="rwkv7_g1a_0.1b"
    xAxisTitle="Batch Size"
    yAxisTitle="TTFT (ms)"
    yFmt="#,##0.1"
    markers=true
    labels=true
/>

</div>
<div>

### 2.9b

<LineChart
    data={prefill_2_9b}
    x=batch_size
    y=avg_ttft
    series=backend
    title="rwkv7_g1c_2.9b"
    xAxisTitle="Batch Size"
    yAxisTitle="TTFT (ms)"
    yFmt="#,##0.1"
    markers=true
    labels=true
/>

</div>
</Grid>

## Total Time by Batch Size

<Grid cols=2>
<div>

### 0.1b

<LineChart
    data={prefill_0_1b}
    x=batch_size
    y=avg_total_ms
    series=backend
    title="rwkv7_g1a_0.1b"
    xAxisTitle="Batch Size"
    yAxisTitle="Total Time (ms)"
    yFmt="#,##0.1"
    markers=true
    labels=true
/>

</div>
<div>

### 2.9b

<LineChart
    data={prefill_2_9b}
    x=batch_size
    y=avg_total_ms
    series=backend
    title="rwkv7_g1c_2.9b"
    xAxisTitle="Batch Size"
    yAxisTitle="Total Time (ms)"
    yFmt="#,##0.1"
    markers=true
    labels=true
/>

</div>
</Grid>

## Data

```sql all_data
SELECT model_name, backend, batch_size, prefill_tok_per_s, prefill_total_ms, ttft_p50_ms, gpu_short
FROM bench.prefill_uniform
WHERE run_id IN ${inputs.run_filter.value}
ORDER BY model_name, backend, batch_size
```

<DataTable data={all_data} rows=50>
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=batch_size title="Batch Size" fmt="#,##0" />
    <Column id=prefill_tok_per_s title="tok/s" fmt="#,##0" contentType=colorscale colorScale=positive />
    <Column id=prefill_total_ms title="Prefill (ms)" fmt="#,##0.1" />
    <Column id=ttft_p50_ms title="TTFT p50 (ms)" fmt="#,##0.1" />
    <Column id=gpu_short title="GPU" />
</DataTable>
