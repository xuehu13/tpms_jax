# M4 review validation, 2026-10-01

> 历史阶段记录：以下结果、测试数量及“下一步”保留当时口径。当前状态和后续顺序以 [当前 TPMS 计算精度验证范围](../../docs/RESEARCH_STATUS.md) 为准；本次未改动该阶段原始数值证据。

The original baseline is commit `a71c3b1fc74e3bd70c6f7a7a99d3d836879b80d5`.
These records are retained outside ignored `results/` so that reviewers can
inspect both the original run and the repaired implementation.

## Changes and validation

- Stop the entire study on either an exception or a failed consistency check.
  Save the partial CSV and return exit code 1 instead of continuing refinement.
- Guard summary analysis against missing or failed cases.
- Keep heavy Problem instances only when `include_problem=True` is requested.
- Measure relaxed construction and four-solve time separately.
- Recover strains using `sym(H)` consistently with the constitutive law.
- Weight the binary-reference fraction with JxW, as for the smooth fraction.
- Replace the roundoff-dependent failure test with an intentionally inconsistent
  macro gradient, and count actual Problem constructions in the relaxed test.
- Add injected failure coverage at the beginning, middle, last coupling case,
  and E_min verification case, for both `check_failed` and exception statuses.

The full suite passed **62 tests** in the WSL project. The complete numerical
script exited successfully with **10/10 cases status=ok**. Compared with the
preserved original CSV, the maximum relative difference was approximately
`1.12e-8` for Fz and `2.84e-12` for internal energy. Timing is not compared as a
physical output. The original maximum reduced residual is `3.92e-12`, correcting
the earlier report that claimed all reduced residuals were below `2.3e-13`.

## Evidence

- `original_a71c3b1.csv`, `original_a71c3b1.txt`: supplied original run.
- `fixed.csv`, `run.txt`: complete repaired run in the original WSL project.
- `tests.txt`: full-suite output.
- `environment.json`: installed package versions, CUDA device, lock and source
  file hashes. It records the parent commit because capture preceded commit;
  tested file hashes identify the modifications exactly.
- `comparison.csv`, `summary.json`: keyed physical-output comparisons and hashes.

The tests and numerical run use float64 (`jax_enable_x64=True` in the code).
This run loaded PyPI `jax-fem==0.0.12` from the project's Pixi environment;
it did not import the separate reference checkout under `~/projects/jax-fem`.

## Reproduce from the repository root

```sh
pixi run python -m pytest tests/ -q
pixi run python scripts/m4_numerical_study.py
cp results/m4_numerical_study.csv validation/m4_review/fixed.csv
pixi run python scripts/capture_m4_evidence.py
```

To refresh saved logs, redirect the first two commands to `tests.txt` and
`run.txt`. Refreshing records modifies tracked evidence; retain the original
baseline files. The numerical checks are internal consistency checks, not an
independent proof against Abaqus or the binary Gyroid geometry. The 10-case
matrix remains fixed lateral strain; it contains no beta-relaxation study.

## Remaining scientific questions

Mesh change rates are indicators, not physical error bounds. Beta changes the
material field and its integrated fraction. A pair of E_min values does not
establish convergence to a traction-free binary geometry. This is an XY-periodic
flat-face compression problem, not full XYZ periodic homogenization. Standard
Abaqus C3D8 has selective volumetric integration and is not assumed identical to
the JAX eight-point displacement element; a separate element comparison must
precede any same-discretization claim.
