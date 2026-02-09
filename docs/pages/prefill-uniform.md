---
title: Prefill Uniform Benchmarks
---

# Prefill Uniform Benchmarks

```sql runs
SELECT DISTINCT run_id, human_name
FROM bench.prefill_uniform
ORDER BY human_name
```

```sql models
SELECT DISTINCT model_name
FROM bench.prefill_uniform
ORDER BY model_name
```

```sql backends
SELECT DISTINCT backend
FROM bench.prefill_uniform
ORDER BY backend
```

<Dropdown data={runs} name=run_filter value=run_id label=human_name title="Run" multiple=true selectAllByDefault=true />
<Dropdown data={models} name=model_filter value=model_name title="Model" multiple=true selectAllByDefault=true />
<Dropdown data={backends} name=backend_filter value=backend title="Backend" multiple=true selectAllByDefault=true />

```sql filtered
SELECT *
FROM bench.prefill_uniform
WHERE run_id IN ${inputs.run_filter.value}
  AND model_name IN ${inputs.model_filter.value}
  AND backend IN ${inputs.backend_filter.value}
```

```sql summary
SELECT
    COUNT(*) AS total_cases,
    MEDIAN(prefill_tok_per_s) AS median_tok_per_s,
    MAX(prefill_tok_per_s) AS best_tok_per_s
FROM ${filtered}
```

<BigValue data={summary} value=total_cases title="Total Cases" fmt="#,##0" />
<BigValue data={summary} value=median_tok_per_s title="Median tok/s" fmt="#,##0" />
<BigValue data={summary} value=best_tok_per_s title="Best tok/s" fmt="#,##0" />

```sql throughput_by_seq_len
SELECT
    seq_len,
    backend,
    AVG(prefill_tok_per_s) AS avg_tok_per_s
FROM ${filtered}
GROUP BY seq_len, backend
ORDER BY seq_len
```

<BarChart
    data={throughput_by_seq_len}
    x=seq_len
    y=avg_tok_per_s
    series=backend
    title="Avg Prefill Throughput by Sequence Length"
    yAxisTitle="tok/s"
/>

```sql ttft_by_seq_len
SELECT
    seq_len,
    backend,
    AVG(ttft_p50_ms) AS avg_ttft_p50_ms
FROM ${filtered}
GROUP BY seq_len, backend
ORDER BY seq_len
```

<LineChart
    data={ttft_by_seq_len}
    x=seq_len
    y=avg_ttft_p50_ms
    series=backend
    title="Avg TTFT (p50) by Sequence Length"
    yAxisTitle="ms"
/>

```sql table_data
SELECT
    run_id,
    human_name,
    model_name,
    backend,
    batch_size,
    token_chunk_size,
    seq_len,
    prefill_total_ms,
    total_prompt_tokens,
    prefill_tok_per_s,
    num_infer_calls,
    ttft_min_ms,
    ttft_p50_ms,
    ttft_max_ms,
    gpu_short,
    git_sha
FROM ${filtered}
ORDER BY human_name, model_name, backend, seq_len
```

<DataTable data={table_data} rows=50>
    <Column id=human_name title="Run" />
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=batch_size title="Batch" fmt="#,##0" />
    <Column id=token_chunk_size title="Chunk" fmt="#,##0" />
    <Column id=seq_len title="Seq Len" fmt="#,##0" />
    <Column id=prefill_total_ms title="Prefill (ms)" fmt="#,##0.1" />
    <Column id=total_prompt_tokens title="Tokens" fmt="#,##0" />
    <Column id=prefill_tok_per_s title="tok/s" fmt="#,##0" />
    <Column id=num_infer_calls title="Infer Calls" fmt="#,##0" />
    <Column id=ttft_min_ms title="TTFT Min (ms)" fmt="#,##0.1" />
    <Column id=ttft_p50_ms title="TTFT p50 (ms)" fmt="#,##0.1" />
    <Column id=ttft_max_ms title="TTFT Max (ms)" fmt="#,##0.1" />
    <Column id=gpu_short title="GPU" />
    <Column id=git_sha title="Git SHA" />
</DataTable>
