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
params_spiking = KNNetParameters(eps = 0.03, beta = 0.035, d = 0.26, a = 0.25, J = 0.1081 + 0.1)

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

imin = 0; icrit = 15000; nt = 16000
# imin = 0; icrit = 20000; nt = 21000
f_out, _ = one_neuron(.5, 0, nt, params_spiking)
# %matplotlib inline
# plt.figure()
# plt.plot(f_out[0:500])
# plt.show()
# plt.close()

f_out = f_out.reshape(f_out.shape[0], 1)
print(f_out.shape)

input_size = 0
hidden_size = 2000
output_size = 1
eps_m = 0.025
delta_eps = 0.005
a = 0.25
# a = 0.3
eps = -delta_eps + 2 * delta_eps * torch.rand(hidden_size, 1).to(device) + eps_m
#eps = torch.as_tensor(eps_m).to(device)
# J = (1 + a - torch.sqrt(1 + a * a - a + 3 * eps)) / 3 + 0.01 # Slightly more bifurcation value
J = (1 + a - torch.sqrt(1 + a * a - a + 3 * eps)) / 3 + 0.05
J = J.to(device)
# p = KNNetParameters(eps=eps, J=J, q=2.0)
bifparams = []
# for i in np.arange(0.01, 0.75, 0.01):
for i in np.arange(0.0, 0.01, 0.01):
    for j in np.arange(0.0, 0.5, 0.001):
        p = KNNetParameters(eps=eps, J=J, q=0.7, g=0.05, beta=j)

        x_initial = 0.6 * torch.rand(hidden_size, 1).to(device)
        y_initial = torch.zeros(hidden_size, 1).to(device)
        z_initial = torch.zeros(hidden_size, 1).to(device)
        ISPC_initial = torch.zeros(hidden_size, 1).to(device)
        initial_state = KNNetState(x=x_initial, y=y_initial, z=z_initial, ISPC=ISPC_initial)
        net = KNNet(input_size, hidden_size, output_size, p=p)
        net.to_device(device)
        lp = ForceParameters(stop_learning=icrit, start_learning=imin)
        fl = ForceLearn(net=net, lp=lp, save_states=True)

        train_logs, states = fl.train(target_outputs=f_out, state=initial_state)

        # plt.plot((train_logs.numpy()[-1000:, 0, 0] - f_out[-1000:, 0]))
        L2 = torch.linalg.norm(train_logs[-1000:, 0, 0] - f_out[-1000:, 0])
        # print(L2)
        # print(torch.log(L2))
        bifparams.append([i, j, torch.log(L2).item()])
        print(bifparams[-1])
        # break
    # break

bifparams = np.array(bifparams)
np.save('./betalog', bifparams)