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
    row = MOD.evaluate_case(8, 20.0, 1e-3, "fixed")
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


def test_status_distinguishes_check_failure_from_success():
    # same computation with a machine-precision work-identity gate must
    # be reported as check_failed, not ok (status semantics)
    row = MOD.evaluate_case(8, 20.0, 1e-3, "fixed", tol_work=1e-16)
    assert row["status"] == "check_failed"
    assert row["work_identity_err"] > 1e-16
    assert not row["checks"]["work_identity<=1e-16"]


def test_relaxed_path_has_no_temporary_problem():
    # the relaxed branch must evaluate the density callable on the single
    # problem it builds (no duplicate mesh/P_mat construction)
    row = MOD.evaluate_case(8, 20.0, 1e-3, "relaxed")
    assert row["status"] == "ok"
    from geometry import density
    import numpy as onp
    expected = density(row["_problem"].physical_quad_points, 0.541062, 20.0, 1.0)
    assert onp.allclose(onp.asarray(row["_problem"].rho),
                        onp.asarray(expected))
    assert abs(row["sigma_xx"]) <= 1e-8 and abs(row["sigma_yy"]) <= 1e-8
