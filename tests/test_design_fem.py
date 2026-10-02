"""Numerical derivative checks exercise the sparse adjoint and periodic field."""
import numpy as np
import pytest
from design_fem import GyroidDesign, width_basis, SOLVER_OPTIONS
from geometry import density
from scripts.m4_numerical_study import evaluate_case


@pytest.mark.parametrize('axis', [0,1,2])
def test_width_and_density_are_periodic(axis):
    points = np.array([[.13,.29,.41],[.31,.59,.79]])
    shifted = points.copy()
    shifted[:,axis] += 1
    theta = np.array([.541062,.025,-.02,.015])
    widths = width_basis(points) @ theta
    shifted_widths = width_basis(shifted) @ theta
    np.testing.assert_allclose(widths,shifted_widths,atol=1e-14,rtol=0)
    np.testing.assert_allclose(density(points,widths,20),density(shifted,shifted_widths,20),atol=1e-14,rtol=0)


@pytest.mark.parametrize('theta', [[.01,.02,0,0],[float('nan'),0,0,0],[.5,0,0]])
def test_invalid_design_rejected(theta):
    with pytest.raises(ValueError):
        GyroidDesign.validate_theta(theta)


def test_design_forward_matches_existing_m4():
    model = GyroidDesign(4)
    actual = model.forward([.541062,0,0,0])
    reference = evaluate_case(4,20,.001,'fixed',solver_options=SOLVER_OPTIONS)
    assert actual['status'] == reference['status'] == 'ok'
    for name in ('Fz_top','U_internal'):
        assert actual[name] == pytest.approx(reference[name],rel=1e-8,abs=1e-10)


def test_complete_adjoint_and_concrete_state_restore():
    model = GyroidDesign(4)
    theta = np.array([.541062,.025,-.02,.015])
    derivatives = model.derivatives(theta)
    jac = derivatives['jacobian']
    np.testing.assert_allclose(jac[0],derivatives['energy_envelope_gradient'],atol=1e-8,rtol=1e-8)
    assert np.linalg.norm(jac[2]) > 1e-7  # Qw needs du/dtheta; frozen-w derivative is zero.
    directions = (np.array([1,0,0,0]), np.array([.5,-.4,.6,-.3])/np.sqrt(.86))
    for direction in directions:
        h = 3e-5
        plus,minus = model.forward(theta+h*direction),model.forward(theta-h*direction)
        assert plus['status'] == minus['status'] == 'ok'
        fd = (np.array(plus['values'])-np.array(minus['values']))/(2*h)
        np.testing.assert_allclose(jac @ direction,fd,atol=1e-8,rtol=1e-4)
    restored = model.forward(theta)
    assert restored['status'] == 'ok'
    np.testing.assert_allclose(restored['values'],derivatives['values'],atol=1e-10,rtol=1e-8)
