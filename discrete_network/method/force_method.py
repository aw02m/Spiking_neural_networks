import torch
import numpy as np

from tqdm import tqdm
from discrete_network.network import KNNet, KNNetState

from typing import NamedTuple, Optional


class ForceParameters(NamedTuple):
    lr: float = 1e-1
    start_learning: int = 0
    stop_learning: int = 0


def force_iteration(
    Pinv: torch.Tensor,
    r: torch.Tensor,
    error: torch.Tensor,
    output_weights: torch.Tensor,
):
    cd_out = torch.mm(Pinv, r).type(torch.float)
    Pinv_out = Pinv - torch.mm(cd_out, torch.t(cd_out)) / (
        1.0 + torch.mm(torch.t(r), cd_out)
    )
    cd_out = torch.mm(Pinv_out, r)
    phi_out = output_weights - torch.mm(cd_out, error.T)
    return Pinv_out, phi_out


class ForceLearn:
    def __init__(
        self,
        net: KNNet,
        lp: ForceParameters = ForceParameters(),
    ) -> None:
        self.lp = lp
        self.net = net
        # data.shape = (timesteps, dim_inout)
        # target_output.shape = (timesteps, dim_output)

    def train(self, target_outputs: np.array, data: Optional[np.array] = None, state: Optional[KNNetState] = None):
        target_outputs = torch.from_numpy(target_outputs).to(self.net.device)
        if not data is None:
            data = torch.from_numpy(data).to(self.net.device)
        T = len(target_outputs)
        s = state
        outputs = []
        out = torch.zeros(target_outputs.shape[1], 1)
        output_weights = self.net.output_weights
        Pinv = self.lp.lr * torch.eye(self.net.hidden_weights.shape[0]).type(
            torch.float
        ).to(self.net.device)
        for ts in tqdm(range(T)):
            out, r, s = self.net.step(state=s, prev_out=out)
            outputs.append(out.cpu())
            error = out - target_outputs[ts].reshape(out.shape)
            error = error.type(torch.float)
            if self.lp.start_learning < ts < self.lp.stop_learning:
                Pinv, output_weights = force_iteration(Pinv, r, error, output_weights)
                self.net.output_weights = output_weights
        return torch.stack(outputs)

    @property
    def params(self):
        return self.lp
