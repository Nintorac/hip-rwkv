---
title: Decode Benchmarks
---

# Decode Benchmarks

```sql runs
SELECT DISTINCT run_id, human_name || ' (' || started_at_utc || ')' AS run_label
FROM bench.decode
ORDER BY started_at_utc DESC
```

<Dropdown data={runs} name=run_filter value=run_id label=run_label title="Run" multiple=true selectAllByDefault=true />

```sql decode_0_1b
SELECT decode_steps, backend, AVG(decode_tok_per_s) AS avg_tok_s, AVG(decode_total_ms) AS avg_total_ms
FROM bench.decode
WHERE model_name = 'rwkv7_g1a_0.1b'
  AND run_id IN ${inputs.run_filter.value}
GROUP BY decode_steps, backend
ORDER BY decode_steps
```

```sql decode_2_9b
SELECT decode_steps, backend, AVG(decode_tok_per_s) AS avg_tok_s, AVG(decode_total_ms) AS avg_total_ms
FROM bench.decode
WHERE model_name = 'rwkv7_g1c_2.9b'
  AND run_id IN ${inputs.run_filter.value}
GROUP BY decode_steps, backend
ORDER BY decode_steps
```

## Throughput by Decode Steps

<Grid cols=2>
<div>

<LineChart
    data={decode_0_1b}
    x=decode_steps
    y=avg_tok_s
    series=backend
    title="0.1B Model"
    xAxisTitle="Decode Steps"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</div>
<div>

<LineChart
    data={decode_2_9b}
    x=decode_steps
    y=avg_tok_s
    series=backend
    title="2.9B Model"
    xAxisTitle="Decode Steps"
    yAxisTitle="tok/s"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</div>
</Grid>

## Total Time by Decode Steps

<Grid cols=2>
<div>

<LineChart
    data={decode_0_1b}
    x=decode_steps
    y=avg_total_ms
    series=backend
    title="0.1B Model"
    xAxisTitle="Decode Steps"
    yAxisTitle="Total (ms)"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</div>
<div>

<LineChart
    data={decode_2_9b}
    x=decode_steps
    y=avg_total_ms
    series=backend
    title="2.9B Model"
    xAxisTitle="Decode Steps"
    yAxisTitle="Total (ms)"
    yFmt="#,##0"
    markers=true
    labels=true
/>

</div>
</Grid>

## All Results

```sql all_results
SELECT model_name, backend, batch_size, decode_steps, decode_tok_per_s, decode_total_ms, gpu_short
FROM bench.decode
WHERE run_id IN ${inputs.run_filter.value}
ORDER BY model_name, backend, decode_steps
```

<Details title="Raw Data">

<DataTable data={all_results} search=true>
    <Column id=model_name title="Model" />
    <Column id=backend title="Backend" />
    <Column id=batch_size title="Batch Size" fmt="#,##0" />
    <Column id=decode_steps title="Steps" fmt="#,##0" />
    <Column id=decode_tok_per_s title="tok/s" fmt="#,##0.0" contentType=colorscale />
    <Column id=decode_total_ms title="Total (ms)" fmt="#,##0.0" />
    <Column id=gpu_short title="GPU" />
</DataTable>

</Details>
