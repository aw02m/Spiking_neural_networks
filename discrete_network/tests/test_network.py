import torch
from ..network import KNNetParameters, KNNet, KNNetState

def test_crete_network():
    input_size = 10
    hidden_size = 1000
    output_size = 15
    net = KNNet(input_size, hidden_size, output_size)
    assert net.input_weights.shape == (hidden_size, input_size)
    assert net.output_weights.shape == (hidden_size, output_size)
    assert net.hidden_weights.shape == (hidden_size, hidden_size)

def test_self_parameters():
    params = KNNetParameters(x_th = torch.as_tensor(1.0))
    net = KNNet(1, 1, 1, p=params)
    assert net.params == params

def test_network_one_step_without_data():
    input_size = 0
    hidden_size = 1000
    output_size = 15
    net = KNNet(input_size, hidden_size, output_size)
    net.step()

def test_network_one_step_with_data():
    data = torch.tensor([[1]]).type(torch.float)
    input_size = 1
    hidden_size = 1000
    output_size = 15
    net = KNNet(input_size, hidden_size, output_size)
    net.step(data = data)
