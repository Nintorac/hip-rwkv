---
title: hip-rwkv Benchmarks
---

# Bench Health

```sql runs
SELECT * FROM bench.runs_with_defaults
```

<Callout>
This is the refreshed view. The original pages remain at the old URLs for side-by-side comparison.
</Callout>

## Quick Filters

<Dropdown data={runs} name=run_filter value=run_id label=run_label title="Runs" multiple=true selectAllByDefault=false defaultValue={runs.filter(r => r.is_default).map(r => r.run_id)} />

## Recent Runs

<DataTable data={runs} rows=20 search=true>
    <Column id=run_label title="Run" />
    <Column id=profile title="Profile" />
    <Column id=gpu_short title="GPU" />
    <Column id=started_at_utc title="Started (UTC)" fmt="yyyy-MM-dd HH:mm" />
    <Column id=git_sha title="Git SHA" />
    <Column id=git_dirty title="Dirty" fmt="boolean" />
</DataTable>

## Pages

- [Decode (new)](/decode-new)
- [Prefill Uniform (new)](/prefill-uniform-new)
- [Prefill Mixed (new)](/prefill-mixed-new)
- [Decode (original)](/decode)
- [Prefill Uniform (original)](/prefill-uniform)
- [Prefill Mixed (original)](/prefill-mixed)
