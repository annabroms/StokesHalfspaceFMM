import json
from pathlib import Path
r=json.loads(Path('native/build/matlab_timing_validation.json').read_text())
m=r['metadata']; build=m['onecall_build']
a=[v for v in r['accuracy'] if v['backend']=='one-call']
vmax=lambda key: max(v[key] for v in a)
rec=next(v for v in r['reciprocity'] if v['backend']=='one-call')
t=r['timings']; complete=[v for v in t if v['scope']=='complete']
prepared=[v['speedup'] for v in t if v['scope']=='prepared']
balanced=next(v for v in complete if v['ns']==65536 and v['nt']==65536)
selected={(1024,1024),(4096,4096),(16384,16384),(16384,65536),(65536,65536),(65536,262144)}
table='\n'.join(f"| {v['ns']:,} | {v['nt']:,} | {v['two_call_s']:.6f} | {v['one_call_s']:.6f} | {v['speedup']:.3f} |" for v in complete if (v['ns'],v['nt']) in selected)
stem='../results/'+Path(r['scaling_stem']).name
cpu=next(line.split(':',1)[1].strip() for line in m['cpu'].splitlines() if 'AMD Ryzen' in line)
text=f"""## Validation measurements (2026-11-07)

- Full acceptance suite: **{r['passed']} passed, {r['failed']} failed, {r['incomplete']} incomplete**,
  {r['suite_seconds']:.3f} s. Code Analyzer: zero findings in seven changed/new MATLAB files.
- MATLAB {m['matlab']}; {cpu}; `OMP_NUM_THREADS={m['omp_num_threads']}`.
- Scaling run: default sizes/ratios above, five repetitions, tolerance 1e-8,
  seed 481516. Complete velocity calls and prepared calls are measured separately.
  Every reported duration is measured with MATLAB `tic`/`toc`, outside the native call.
- The native counter verified one call with flags (0,2); full correctness
  tests also cover pressure, vertical recentering, wall height, viscosity,
  empty data, coincidences and explicit rejection of gradients.

Across the four field fixtures, the largest one-call velocity relative error
against independent direct Blake evaluation was {vmax('velocity_relative'):.6e};
its largest constituent-normalized error was {vmax('velocity_normalized'):.6e}.
The largest pressure relative error was {vmax('pressure_relative'):.6e}.
The largest one-call/two-call velocity difference, normalized by the common
constituent scale, was {vmax('difference_normalized'):.6e}.

Flow at the wall was zero to numerical accuracy, not overwritten: the largest
raw one-call wall residual was {vmax('wall_absolute'):.6e},
and the largest constituent-normalized wall residual was
{vmax('wall_normalized'):.6e}.

With {rec['sources_A']} and {rec['sources_B']} sources in the exchanged clouds,
the one-call norm-product-normalized work defect was {rec['distributed_defect']:.6e};
the Cartesian pair-tensor defect was {rec['pair_defect']:.6e}.

Selected complete-call medians:

| Sources | Targets | Two calls (s) | One call (s) | Speedup |
|---:|---:|---:|---:|---:|
{table}

There is **no uniform speedup that grows with problem size**. Some balanced
cases improved, while some higher-target-count cases were slower. In the
65,536/65,536 case, paired complete-call speed ratios ranged from
{balanced['paired_min']:.3f} to {balanced['paired_max']:.3f}.
Across all measured configurations, prepared-geometry speedups ranged from
{min(prepared):.3f} to {max(prepared):.3f}.
The existing two-call implementation remains the default.

Use total and prepared timing columns to compare performance. The `*_fmm_s`
columns time the backend interfaces from MATLAB for both implementations. The
new backend includes packing and reconstruction inside its MEX boundary; the
old backend performs its final half-space assembly outside its FMM interfaces.
Complete-call timings include all costs for both implementations.

Full scaling data: [MAT report]({stem}.mat),
[all timing summaries]({stem}_timings.csv),
[raw repetitions]({stem}_samples.csv),
[field accuracy]({stem}_accuracy.csv),
[reciprocity]({stem}_reciprocity.csv).
Generated reports are ignored by Git. Re-run the driver with `save=true` to
create reports on another installation.

Dependency provenance from this scaling run:

- Selected Stokes wrapper: `{m['stokes_path']}`.
- Selected FMM3D MEX SHA-256: `{m['mex_sha256']}`.
- Linked static library SHA-256: `{build['archive_sha256']}`.
- One-call MEX SHA-256: `{build['mex_sha256']}`.
- The complete build record, compiler settings and source checksums are in
  `native/build/build.json` and in the saved report metadata.
"""
p=Path('docs/one-call-fmm.md')
s=p.read_text(); s=s[:s.index('## Validation measurements')]+text
p.write_text(s)
print('Updated documentation from',r['scaling_stem'])
print('SUITE',r['passed'],r['failed'],r['incomplete'])
print('ONECALL_BUILD',build['mex_sha256'])
