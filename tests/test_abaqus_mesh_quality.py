import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from abaqus_mesh_quality import von_mises, weighted_quantiles


def test_mises_hydrostatic_uniaxial_and_shear_exact_values():
    stress = np.array([[7,7,7,0,0,0], [-5,0,0,0,0,0], [0,0,0,2,0,0]], float)
    np.testing.assert_allclose(von_mises(stress), [0,5,2*np.sqrt(3)], atol=1e-14)


def test_volume_quantiles_are_not_element_count_quantiles():
    # A high-count low-volume tail must not dictate a bulk quantile.
    values = np.r_[np.ones(100)*100, [2,3]]
    weights = np.r_[np.ones(100)*.0001, [.49,.5]]
    assert weighted_quantiles(values, weights, [.5,.95,.99]) == [3,3,3]
    assert weighted_quantiles(values, weights, [.995]) == [100]
    reverse = np.arange(len(values))[::-1]
    assert weighted_quantiles(values[reverse], weights[reverse], [.5,.99]) == [3,3]


@pytest.mark.parametrize("weights", [[1,0],[1,-1],[1,np.nan],[1,np.inf]])
def test_invalid_integration_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="positive finite"):
        weighted_quantiles([1,2], weights)
