import torch
import pytest
from core.ai.rabinet_grok import RabiNet, GrokResonantLayer

def test_rabinet_init():
    rabi_earth = RabiNet(omega=1.0, environment="earth")
    rabi_mars = RabiNet(omega=1.0, environment="mars")
    assert rabi_earth.omega.item() == 1.0
    assert rabi_mars.omega.item() == 1.2  # 1.0 * 1.2
    assert rabi_earth.phi.item() == pytest.approx(math.pi / 3.12)
    assert rabi_mars.phi.item() == pytest.approx((math.pi / 3.12) * 1.1)

def test_rabinet_forward():
    rabi = RabiNet(omega=1.0)
    x = torch.tensor([0.0, math.pi / 2, math.pi])
    y = rabi(x)
    assert y[0].item() == pytest.approx(1.0)  # cos(0)
    assert y[1].item() == pytest.approx(0.0)  # cos(pi/2)
    assert y[2].item() == pytest.approx(-1.0)  # cos(pi)

def test_grokresonantlayer_forward_shape():
    layer = GrokResonantLayer(d_model=768, d_ff=3072, omega=0.75)
    x = torch.randn(1, 10, 768)
    y = layer(x)
    assert y.shape == (1, 10, 768)

def test_grokresonantlayer_environment_effect():
    earth_layer = GrokResonantLayer(omega=1.0, environment="earth")
    mars_layer = GrokResonantLayer(omega=1.0, environment="mars")
    x = torch.tensor([0.0], dtype=torch.float32).repeat(1, 1, 768)
    y_earth = earth_layer(x)
    y_mars = mars_layer(x)
    # Check if environment affects output (approximate due to network weights)
    assert not torch.allclose(y_earth, y_mars, atol=1e-4)  # Different due to RabiNet params
