import torch
from test_oec_certificate import _model
from splart.oec_endpoint_adapter import fit_oec_endpoints

def test_oec_cea_has_scalar_gradient_and_moves_from_initialization():
    model=_model(); result=fit_oec_endpoints(model,1.,iterations=8)
    assert torch.isfinite(result.loss)
    assert result.lower.item()!=-.5 or result.upper.item()!=1.5

def test_oec_cea_endpoint_recovery_stays_extrapolative():
    result=fit_oec_endpoints(_model(anisotropic=True),1.,iterations=40)
    assert result.lower < 0 and result.upper > 1

def test_oec_cea_never_populates_base_gradients():
    model=_model(); model.means=torch.nn.Parameter(model.means)
    fit_oec_endpoints(model,1.,iterations=3)
    assert model.means.grad is None
