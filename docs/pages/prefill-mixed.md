---
title: Prefill Mixed Benchmarks
---

# Prefill Mixed Benchmarks

```sql runs
SELECT DISTINCT run_id, human_name || ' (' || started_at_utc || ')' AS run_label
FROM bench.prefill_mixed
ORDER BY started_at_utc DESC
```

```sql models
SELECT DISTINCT model_name
FROM bench.prefill_mixed
ORDER BY model_name
```

```sql backends
SELECT DISTINCT backend
FROM bench.prefill_mixed
ORDER BY backend
```

<Dropdown data={runs} name=run_filter value=run_id label=run_label title="Run" multiple=true selectAllByDefault=true />
<Dropdown data={models} name=model_filter value=model_name title="Model" multiple=true selectAllByDefault=true />
<Dropdown data={backends} name=backend_filter value=backend title="Backend" multiple=true selectAllByDefault=true />

```sql filtered
SELECT *
FROM bench.prefill_mixed
WHERE run_id IN ${inputs.run_filter.value}
  AND model_name IN ${inputs.model_filter.value}
  AND backend IN ${inputs.backend_filter.value}
```

```sql summary
SELECT
    COUNT(*) AS total_cases,
    MEDIAN(prefill_tok_per_s) AS median_tok_s,
    MAX(prefill_tok_per_s) AS best_tok_s
FROM ${filtered}
```

<BigValue data={summary} value=total_cases title="Total Cases" fmt="#,##0" />
<BigValue data={summary} value=median_tok_s title="Median tok/s" fmt="#,##0.0" />
<BigValue data={summary} value=best_tok_s title="Best tok/s" fmt="#,##0.0" />

## Throughput by Mixed Case

```sql throughput
SELECT
    mixed_case_id,
    backend,
    AVG(prefill_tok_per_s) AS avg_tok_s
FROM ${filtered}
GROUP BY mixed_case_id, backend
ORDER BY avg_tok_s DESC
```

<BarChart
    data={throughput}
    x=mixed_case_id
    y=avg_tok_s
    series=backend
    title="Average Prefill tok/s by Mixed Case"
    swapXY=true
    yFmt="#,##0.0"
/>

## All Results

<DataTable data={filtered} rows=50>
    <Column id=human_name title="Run" />
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=batch_size title="Batch Size" fmt="#,##0" />
    <Column id=token_chunk_size title="Chunk Size" fmt="#,##0" />
    <Column id=mixed_case_id title="Mixed Case" />
    <Column id=total_prompt_tokens title="Prompt Tokens" fmt="#,##0" />
    <Column id=num_infer_calls title="Infer Calls" fmt="#,##0" />
    <Column id=prefill_tok_per_s title="tok/s" fmt="#,##0.0" />
    <Column id=prefill_total_ms title="Total (ms)" fmt="#,##0.0" />
    <Column id=ttft_min_ms title="TTFT min (ms)" fmt="#,##0.2" />
    <Column id=ttft_p50_ms title="TTFT p50 (ms)" fmt="#,##0.2" />
    <Column id=ttft_max_ms title="TTFT max (ms)" fmt="#,##0.2" />
    <Column id=gpu_short title="GPU" />
    <Column id=git_sha title="Git SHA" />
</DataTable>
