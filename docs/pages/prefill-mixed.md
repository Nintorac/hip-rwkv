---
title: Prefill Mixed Benchmarks
---

# Prefill Mixed Benchmarks

```sql runs
SELECT DISTINCT run_id, human_name || ' (' || started_at_utc || ')' AS run_label
FROM bench.prefill_mixed
ORDER BY started_at_utc DESC
```

```sql batch_sizes
SELECT DISTINCT batch_size
FROM bench.prefill_mixed
ORDER BY batch_size
```

<Grid cols=2>
<Dropdown data={runs} name=run_filter value=run_id label=run_label title="Run" multiple=true selectAllByDefault=true />
<ButtonGroup data={batch_sizes} name=batch_filter value=batch_size title="Batch Size" />
</Grid>

## Throughput by Workload

```sql throughput_0_1b
SELECT mixed_case_id, backend,
    AVG(prefill_tok_per_s) AS avg_tok_s
FROM bench.prefill_mixed
WHERE model_name = 'rwkv7_g1a_0.1b'
  AND run_id IN ${inputs.run_filter.value}
  AND batch_size = ${inputs.batch_filter.value}
GROUP BY mixed_case_id, backend
ORDER BY mixed_case_id
```

```sql throughput_2_9b
SELECT mixed_case_id, backend,
    AVG(prefill_tok_per_s) AS avg_tok_s
FROM bench.prefill_mixed
WHERE model_name = 'rwkv7_g1c_2.9b'
  AND run_id IN ${inputs.run_filter.value}
  AND batch_size = ${inputs.batch_filter.value}
GROUP BY mixed_case_id, backend
ORDER BY mixed_case_id
```

<Grid cols=2>

<LineChart
    data={throughput_0_1b}
    x=mixed_case_id
    y=avg_tok_s
    series=backend
    title="0.1b"
    xAxisTitle="Mixed Case"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

<LineChart
    data={throughput_2_9b}
    x=mixed_case_id
    y=avg_tok_s
    series=backend
    title="2.9b"
    xAxisTitle="Mixed Case"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</Grid>

## TTFT by Workload

```sql ttft_0_1b
SELECT mixed_case_id, backend,
    AVG(ttft_p50_ms) AS avg_ttft
FROM bench.prefill_mixed
WHERE model_name = 'rwkv7_g1a_0.1b'
  AND run_id IN ${inputs.run_filter.value}
  AND batch_size = ${inputs.batch_filter.value}
GROUP BY mixed_case_id, backend
ORDER BY mixed_case_id
```

```sql ttft_2_9b
SELECT mixed_case_id, backend,
    AVG(ttft_p50_ms) AS avg_ttft
FROM bench.prefill_mixed
WHERE model_name = 'rwkv7_g1c_2.9b'
  AND run_id IN ${inputs.run_filter.value}
  AND batch_size = ${inputs.batch_filter.value}
GROUP BY mixed_case_id, backend
ORDER BY mixed_case_id
```

<Grid cols=2>

<LineChart
    data={ttft_0_1b}
    x=mixed_case_id
    y=avg_ttft
    series=backend
    title="0.1b"
    xAxisTitle="Mixed Case"
    yAxisTitle="TTFT p50 (ms)"
    yFmt="#,##0.1"
    markers=true
    labels=true
/>

<LineChart
    data={ttft_2_9b}
    x=mixed_case_id
    y=avg_ttft
    series=backend
    title="2.9b"
    xAxisTitle="Mixed Case"
    yAxisTitle="TTFT p50 (ms)"
    yFmt="#,##0.1"
    markers=true
    labels=true
/>

</Grid>

## Throughput by Batch Size

```sql batch_throughput_0_1b
SELECT CAST(batch_size AS VARCHAR) AS batch_size_str, backend,
    mixed_case_id,
    backend || ' / ' || mixed_case_id AS series_label,
    AVG(prefill_tok_per_s) AS avg_tok_s
FROM bench.prefill_mixed
WHERE model_name = 'rwkv7_g1a_0.1b'
  AND run_id IN ${inputs.run_filter.value}
GROUP BY batch_size, backend, mixed_case_id
ORDER BY batch_size, series_label
```

```sql batch_throughput_2_9b
SELECT CAST(batch_size AS VARCHAR) AS batch_size_str, backend,
    mixed_case_id,
    backend || ' / ' || mixed_case_id AS series_label,
    AVG(prefill_tok_per_s) AS avg_tok_s
FROM bench.prefill_mixed
WHERE model_name = 'rwkv7_g1c_2.9b'
  AND run_id IN ${inputs.run_filter.value}
GROUP BY batch_size, backend, mixed_case_id
ORDER BY batch_size, series_label
```

<Grid cols=2>

<LineChart
    data={batch_throughput_0_1b}
    x=batch_size_str
    y=avg_tok_s
    series=series_label
    title="0.1b"
    xAxisTitle="Batch Size"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

<LineChart
    data={batch_throughput_2_9b}
    x=batch_size_str
    y=avg_tok_s
    series=series_label
    title="2.9b"
    xAxisTitle="Batch Size"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</Grid>

## Data

```sql all_data
SELECT model_name, backend, batch_size, mixed_case_id,
    total_prompt_tokens, prefill_tok_per_s, prefill_total_ms,
    ttft_p50_ms, gpu_short
FROM bench.prefill_mixed
WHERE run_id IN ${inputs.run_filter.value}
ORDER BY model_name, backend, batch_size, mixed_case_id
```

<DataTable data={all_data} rows=50 search=true>
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=batch_size title="Batch" fmt="#,##0" />
    <Column id=mixed_case_id title="Mixed Case" />
    <Column id=total_prompt_tokens title="Prompt Tokens" fmt="#,##0" />
    <Column id=prefill_tok_per_s title="tok/s" fmt="#,##0" contentType=colorscale colorScale=positive />
    <Column id=prefill_total_ms title="Total (ms)" fmt="#,##0.1" />
    <Column id=ttft_p50_ms title="TTFT p50 (ms)" fmt="#,##0.1" />
    <Column id=gpu_short title="GPU" />
</DataTable>
