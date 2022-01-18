import torch
from ..network import cubicFunction

def test_cubic_function_zero_input_zero_parameters():
    data = torch.zeros(1).type(torch.float)
    parameter = torch.as_tensor(0.).type(torch.float)
    output = cubicFunction(data, parameter)
    assert torch.allclose(output, torch.tensor([0]).type(torch.float))

def test_cubic_function_zero_input():
    data = torch.tensor([1, 2, 3]).type(torch.float)
    parameter = torch.as_tensor(0.).type(torch.float)
    output = cubicFunction(data, parameter)
    target_output = data * (data - parameter) * (1.0 - data)

    assert torch.allclose(output,target_output)

def test_cubic_function():
    data = torch.tensor([1, 2, 3]).type(torch.float)
    parameter = torch.as_tensor(0.).type(torch.float)
    output = cubicFunction(data, parameter)
    target_output = data * (data - parameter) * (1.0 - data)
    assert torch.allclose(output, target_output)

