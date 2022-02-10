import numpy as np
import torch
from discrete_network.network import KNNet, KNNetParameters, KNNetState
from discrete_network.method.force_method import ForceParameters, ForceLearn
from discrete_network.device import device
import matplotlib.pyplot as plt

print(f"Device = {device.type}")

# params_spiking = KNNetParameters(eps = 0.015, beta = 0.0, d = 0.26, a = 0.25, J = 0.1081 + 0.1)
# params_spiking = KNNetParameters(eps = 0.015, beta = 0.03, d = 0.26, a = 0.25, J = 0.1081 + 0.1)
# params_spiking = KNNetParameters(eps = 0.015, beta = 0.05, d = 0.26, a = 0.25, J = 0.15)

# normal spike
# params_spiking = KNNetParameters(eps = 0.02, beta = 0.0, d = 0.26, a = 0.25, J = 0.1081 + 0.1)
# params_spiking = KNNetParameters(eps = 0.03, beta = 0.035, d = 0.26, a = 0.25, J = 0.1081 + 0.1)

def one_neuron(x0, y0, iteration, p: KNNetParameters):
    """The dynamics of one neuron. Return x, y."""
    x, y = np.zeros(iteration), np.zeros(iteration)
    x[0], y[0] = x0, y0
    for i in range(iteration - 1):
        x[i + 1] = (
            x[i]
            + x[i] * (x[i] - p.a) * (1 - x[i])
            - p.beta * (x[i] > p.d)
            - y[i]
        )
        y[i + 1] = y[i] + p.eps * (x[i] - p.J)
    return x, y

imin = 0; icrit = 20000; nt = 21000

input_size = 0
hidden_size = 2000
output_size = 2

eps_start = 0.01
eps_stop = 0.1
eps = eps_start + (eps_stop - eps_start) * torch.rand(hidden_size, 1).to(device)

a = 0.25

J = (1 + a - torch.sqrt(1 + a * a - a + 3 * eps)) / 3
J = J.to(device)

p = KNNetParameters(
    eps=eps, a=torch.as_tensor(a), J=J, q=1.1, g=0.1, x_th=torch.as_tensor(0.65),
    beta=torch.as_tensor(0.0)
)

bifparams = []
bifparams_second = []

for i in np.arange(0.02, 0.03, 0.001):
    for j in np.arange(0.025, 0.1, 0.0025):
        params_spiking = KNNetParameters(eps = j, beta = i, d = 0.26, a = 0.25, J = 0.1081 + 0.1)
        f_out_x, f_out_y = one_neuron(0.3, 0, nt, params_spiking)
        f_out = np.concatenate([[f_out_x], [f_out_y]], 0).T

        x_initial = 0.9 * torch.rand(hidden_size, 1).to(device)
        y_initial = torch.zeros(hidden_size, 1).to(device)
        z_initial = torch.zeros(hidden_size, 1).to(device)
        ISPC_initial = torch.zeros(hidden_size, 1).to(device)
        initial_state = KNNetState(x=x_initial, y=y_initial, z=z_initial, ISPC=ISPC_initial)
        net = KNNet(input_size, hidden_size, output_size, p=p)
        net.to_device(device)
        lp = ForceParameters(stop_learning=icrit, start_learning=imin)
        fl = ForceLearn(net=net, lp=lp, save_states=True)

        train_logs, states = fl.train(target_outputs=f_out, state=initial_state)

        L2 = torch.linalg.norm(train_logs[-1000:, 0, 0] - f_out[-1000:, 0])
        L2_second = torch.linalg.norm(train_logs[-1000:, 1, 0] - f_out[-1000:, 1])

        print(torch.log(L2))
        bifparams.append([i, j, torch.log(L2).item()])
        bifparams_second.append([i, j, torch.log(L2_second).item()])
        print(f'1dim: {bifparams[-1]}, 2dim: {bifparams_second[-1]}')


bifparams = np.array(bifparams)
bifparams_second = np.array(bifparams_second)
np.save('./betaeps_2', bifparams)
np.save('./betaeps_second_2', bifparams_second)