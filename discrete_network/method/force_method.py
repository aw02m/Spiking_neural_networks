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
            save_states: bool = False,
    ) -> None:
        self.lp = lp
        self.net = net
        self.save_states = save_states

    def train(
            self,
            target_outputs: np.array,
            data: Optional[np.array] = None,
            state: Optional[KNNetState] = None,
            split_data: int = 0
    ):

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
        if self.save_states:
            states = []
        if split_data == 0:
            target_outputs_torch = torch.from_numpy(target_outputs).to(self.net.device)
        else:
            j = 0
        for ts in tqdm(range(T)):

            out, r, s = self.net.step(state=s, prev_out=out)
            if self.save_states:
                states.append(KNNetState(s.x.cpu(), s.y.cpu(), s.z.cpu(), s.ISPC.cpu()))
            outputs.append(out.cpu())
            if split_data > 0:
                if ts % split_data == 0:
                    target_outputs_torch = torch.from_numpy(target_outputs[split_data * j: split_data * (j + 1)]).to(
                        self.net.device)
                    j += 1
                error = out - target_outputs_torch[ts - split_data * (j - 1)].reshape(out.shape)
            else:
                error = out - target_outputs_torch[ts].reshape(out.shape)
            error = error.type(torch.float)
            if self.lp.start_learning < ts < self.lp.stop_learning:
                Pinv, output_weights = force_iteration(Pinv, r, error, output_weights)
                self.net.output_weights = output_weights
        if self.save_states:
            return torch.stack(outputs), states
        return torch.stack(outputs)

    @property
    def params(self):
        return self.lp
