import torch
import numpy as np

from tqdm import tqdm
from typing import NamedTuple, Optional, Tuple


class KNNetParameters(NamedTuple):
    # shoud be tensor since these each neurons have each parameters
    eps: torch.Tensor = torch.as_tensor(0.015)
    beta: torch.Tensor = torch.as_tensor(0.0)
    d: torch.Tensor = torch.as_tensor(0.26)
    a: torch.Tensor = torch.as_tensor(0.25)
    x_th: torch.Tensor = torch.as_tensor(0.65)
    J: torch.Tensor = torch.as_tensor(0.1081)
    q: float = 0.6
    g: float = 0.04
    sparse_ratio: float = 0.1
    type_th: str = "tanh"
    # threshold of the Heviside function like H(x-x_th)


class KNNetState(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor  # z = Step_Function(x - x_th)
    ISPC: torch.Tensor
    # default_initial_state = KNNetState(
    #    x=torch.as_tensor(0.0), y=torch.as_tensor(0.0), z=torch.as_tensor(0.0)
    # )


@torch.jit.script
def _cubicFunction(
    input_tensor: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    return input_tensor * (input_tensor - parameter) * (1.0 - input_tensor)


def cubicFunction(
    input_tensor: torch.Tensor,
    parameter: torch.Tensor,
):
    return _cubicFunction(input_tensor, parameter)


def knNetStep(
    input_tensor: torch.Tensor,
    state: KNNetState = KNNetState(0, 0, 0, 0),
    p: KNNetParameters = KNNetParameters(),
) -> Tuple[torch.Tensor, KNNetState]:
    new_x = (
        state.x
        + cubicFunction(state.x, p.a)
        - p.beta
        * torch.heaviside(
            state.x - p.d, values=torch.as_tensor(0.0).to(input_tensor.device)
        )
        - state.y
    )
    new_y = state.y + p.eps * (state.x - (p.J + input_tensor))
    new_z = (new_x >= p.x_th).type(torch.float)
    return new_z, KNNetState(new_x, new_y, new_z, state.ISPC)


def readOut(
    state: KNNetState,
    p: KNNetParameters,
) -> torch.Tensor:
    if p.type_th == "tanh":
        return torch.tanh(state.x) * state.z


class KNNet:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        p: Optional[KNNetParameters] = None,
        input_weights: Optional[torch.Tensor] = None,
        hidden_weights: Optional[torch.Tensor] = None,
        output_weights: Optional[torch.Tensor] = None,
        eta: Optional[torch.Tensor] = None,
    ) -> None:
        if p is None:
            self.p = KNNetParameters()
        else:
            self.p = p
        if hidden_weights is None:
            self._hidden_weights = torch.randn(size=(hidden_size, hidden_size))
            self._hidden_weights *= self.p.g * (
                torch.rand(size=(hidden_size, hidden_size)) < self.p.sparse_ratio
            ).type(torch.float)
            self._hidden_weights /= torch.sqrt(
                torch.as_tensor(hidden_size).type(torch.float)) * self.p.sparse_ratio
            
        else:
            self._hidden_weights = hidden_weights

        if input_weights is None:
            self._input_weights = torch.rand(size=(hidden_size, input_size))
        else:
            self._input_weights = input_weights

        if output_weights is None:
            self._output_weights = torch.zeros(hidden_size, output_size)
        else:
            self._output_weights = output_weights

        if eta is None:
            self.eta = self.p.q * (
                -1.0 + 2.0 * torch.rand(size=(hidden_size, output_size))
            )  # q * U(-1, 1)

    def to_device(self, device: torch.device):
        self.input_weights = self.input_weights.to(device)
        self.output_weights = self.output_weights.to(device)
        self.hidden_weights = self.hidden_weights.to(device)
        self.eta = self.eta.to(device)

    @property
    def device(self):
        return self._hidden_weights.device

    @property
    def input_weights(self):
        return self._input_weights

    @input_weights.setter
    def input_weights(self, value: torch.Tensor):
        self._input_weights = value

    @property
    def output_weights(self):
        return self._output_weights

    @output_weights.setter
    def output_weights(self, value: torch.Tensor):
        self._output_weights = value

    @property
    def hidden_weights(self):
        return self._hidden_weights

    @hidden_weights.setter
    def hidden_weights(self, value: torch.Tensor):
        self._hidden_weights = value

    @property
    def params(self):
        return self.p

    def step(
        self,
        data: Optional[torch.tensor] = None,
        state: Optional[KNNetState] = None,
        prev_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.tensor, KNNetState]:
        if state is None:
            state = KNNetState(
                torch.zeros(size=(self.hidden_weights.shape[0], 1)).to(
                    self.hidden_weights.device
                ),
                torch.zeros(size=(self.hidden_weights.shape[0], 1)).to(
                    self.hidden_weights.device
                ),
                torch.zeros(size=(self.hidden_weights.shape[0], 1)).to(
                    self.hidden_weights.device
                ),
                torch.zeros(size=(self.hidden_weights.shape[0], 1)).to(
                    self.hidden_weights.device
                ),
            )
        if prev_out is None:
            prev_out = torch.zeros(self.output_weights.shape[1], 1).to(self.device)
        else:
            prev_out = prev_out.to(self.device)
        # between neurons hidden_weights * r, r = tanh(x)\
        JX = torch.mm(self.eta, prev_out) + state.ISPC
        
        if data is None:
            _, new_state = knNetStep(JX, state, self.p)
        else:
            _, new_state = knNetStep(
                JX + torch.mm(self.input_weights, data), state, self.p
            )
        r = readOut(state, self.p)
        r = r.reshape(r.shape, 1)
        new_ISPC = torch.mm(self.hidden_weights, r)
        out = torch.mm(self.output_weights.T, r)
        return out, r, KNNetState(new_state.x, new_state.y, new_state.z, new_ISPC)


# The first step:
# current: prev_current, prev_voltage, JX: ISPC + eta * output_weights.T, parameters
# voltage:  prev_current, prev_voltage, z, parameters
# The seond step:
# new r, new out,
