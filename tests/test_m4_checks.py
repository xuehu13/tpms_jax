"""M4-A tests: consistency checks and status semantics of the study script."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "m4_numerical_study",
    Path(__file__).resolve().parents[1] / "scripts" / "m4_numerical_study.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_evaluate_case_ok_with_consistency_checks():
    row = MOD.evaluate_case(8, 20.0, 1e-3, "fixed", include_problem=True)
    assert row["status"] == "ok"
    assert all(row["checks"].values())
    # macro work identity uses sym(H_used), not averaged integral strains
    assert row["work_identity_err"] <= 1e-8
    assert row["U_internal"] == pytest.approx(row["U_macro"], abs=1e-8)
    # reaction through the loaded face matches the volume-average sigma_zz
    assert abs(row["Fz_top"] - row["sigma_zz"]) <= 1e-6
    assert abs(row["balance"]) <= 1e-8
    # density evaluated on the problem's OWN Gauss points
    from geometry import density
    expected = density(row["_problem"].physical_quad_points, 0.541062, 20.0, 1.0)
    import numpy as onp
    import jax
    jax.config.update("jax_enable_x64", True)
    assert onp.allclose(onp.asarray(row["_problem"].rho),
                        onp.asarray(expected))


def test_status_distinguishes_check_failure_from_success(monkeypatch):
    # Deliberately misreport the imposed H; this breaks macro work without
    # depending on platform-specific floating-point roundoff.
    import numpy as onp
    original = MOD.solve_case
    def wrong_macro(*args):
        values = list(original(*args))
        values[2] = onp.zeros((3, 3))
        return tuple(values)
    monkeypatch.setattr(MOD, "solve_case", wrong_macro)
    row = MOD.evaluate_case(8, 20.0, 1e-3, "fixed")
    assert row["status"] == "check_failed"
    assert row["work_identity_err"] > 1e-8
    assert not row["checks"]["work_identity<=1e-08"]


def test_relaxed_path_has_no_temporary_problem(monkeypatch):
    # the relaxed branch must evaluate the density callable on the single
    # problem it builds (no duplicate mesh/P_mat construction)
    import density_fem
    original = density_fem.make_density_problem
    builds = []
    def counted_build(*args, **kwargs):
        builds.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(density_fem, "make_density_problem", counted_build)
    row = MOD.evaluate_case(8, 20.0, 1e-3, "relaxed", include_problem=True)
    assert len(builds) == 1
    assert row["status"] == "ok"
    from geometry import density
    import numpy as onp
    expected = density(row["_problem"].physical_quad_points, 0.541062, 20.0, 1.0)
    assert onp.allclose(onp.asarray(row["_problem"].rho),
                        onp.asarray(expected))
    assert abs(row["sigma_xx"]) <= 1e-8 and abs(row["sigma_yy"]) <= 1e-8
    assert row["t_build"] > 0 and row["t_solve"] > 0


@pytest.mark.parametrize("failure_status", ["check_failed", "FAILED: injected error"])
@pytest.mark.parametrize("failure_index", [0, 1, 8, 9])
def test_main_preserves_failure_and_stops(tmp_path, monkeypatch,
                                         failure_status, failure_index):
    import csv
    calls = []
    def fake_case(N, beta, emin_ratio, lateral):
        index = len(calls)
        calls.append((N, beta, emin_ratio, lateral))
        row = dict(N=N, beta=beta, emin_ratio=emin_ratio, lateral=lateral,
                   status=failure_status if index == failure_index else "ok")
        if index != failure_index:
            row.update(Fz_top=-0.02, U_internal=0.0001)
        return row
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MOD, "evaluate_case", fake_case)
    assert MOD.main() is False
    assert len(calls) == failure_index + 1
    with open(MOD.CSV_PATH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == failure_index + 1
    assert rows[-1]["status"] == failure_status


def test_main_complete_batch(tmp_path, monkeypatch):
    calls = []
    def fake_case(N, beta, emin_ratio, lateral):
        calls.append((N, beta, emin_ratio, lateral))
        return dict(N=N, beta=beta, emin_ratio=emin_ratio, lateral=lateral,
                    status="ok", Fz_top=-0.02, U_internal=0.0001)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MOD, "evaluate_case", fake_case)
    assert MOD.main() is True
    assert len(calls) == 10


def test_petsc_option_preserves_projected_fixed_and_free_response():
    from petsc4py import PETSc
    settings = {"ksp_rtol": 1e-11, "ksp_atol": 1e-13, "ksp_max_it": 5000,
                "ksp_error_if_not_converged": True}
    options = PETSc.Options()
    previous = {key: options.getAll().get(key) for key in settings}
    try:
        for key, value in settings.items():
            options[key] = value
        solver_options = {"petsc_solver": {"ksp_type": "cg", "pc_type": "gamg"}}
        for lateral in ("fixed", "relaxed_free"):
            default = MOD.evaluate_case(8, 20., .001, lateral)
            petsc = MOD.evaluate_case(8, 20., .001, lateral, solver_options=solver_options)
            assert default["status"] == petsc["status"] == "ok"
            for key in ("Fz_top", "U_internal", "eps_x", "eps_y"):
                assert petsc[key] == pytest.approx(default[key], rel=1e-6, abs=1e-10)
            if lateral == "relaxed_free":
                assert max(abs(petsc["sigma_xx"]), abs(petsc["sigma_yy"])) <= 1e-8
    finally:
        for key, value in previous.items():
            if value is None:
                del options[key]
            else:
                options[key] = value
