from re import T
import torch
from typing import NamedTuple, Optional, Tuple


class KNNetParameters(NamedTuple):
    # shoud be tensor since these each neurons have each parameters
    eps: torch.Tensor = torch.as_tensor(0.015)
    beta: torch.Tensor = torch.as_tensor(0.0)
    d: torch.Tensor = torch.as_tensor(0.26)
    a: torch.Tensor = torch.as_tensor(0.25)
    x_th: torch.Tensor = torch.as_tensor(0.65)
    q: float = 0.6
    g: float = 0.04
    sparse_ratio: float = 0.1
    type_th: str = "tanh"
    # threshold of the Heviside function like H(x-x_th)


class KNNetState(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor  # z = Step_Function(x - x_th)
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
    state: KNNetState = KNNetState(),
    p: KNNetParameters = KNNetParameters(),
) -> Tuple(torch.Tensor, KNNetState):
    new_x = (
        state.x
        + cubicFunction(state.x, p.a)
        - p.beta * torch.heaviside(state.x - p.d)
        - state.y
    )
    new_y = state.y + p.eps * (state.x - (p.J + input_tensor))
    new_z = (new_x >= p.x_th).type(torch.float)
    return new_z, KNNetState(new_x, new_y, new_z)


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
            self.hidden_weights = torch.randn(size=(hidden_size, hidden_size))
            self.hidden_weights *= p.g * (
                torch.random(size=(hidden_size, hidden_size)) < p.sparse_ratio
            ).type(torch.float)
            self.hidden_weights /= torch.sqrt(
                torch.as_tensor(hidden_size).type(torch.float) * p.sparse_ratio
            )
        if input_weights is None:
            self.input_weights = torch.random(size=(hidden_size, input_size))

        if output_weights is None:
            self.output_weights = torch.zeros(hidden_size, output_weights)

        if eta is None:
            self.eta = p.q * torch.zeros(output_size, hidden_size)

    def to_device(self, device: torch.device):
        self.input_weights = self.input_weights.to(device)
        self.output_weights = self.output_weights.to(device)
        self.hidden_weights = self.hidden_weights.to(device)
        self.eta = self.eta.to(device)

    def forward(
        self, data: torch.tensor, state: Optional[KNNetState] = None
    ) -> Tuple(torch.tensor, KNNetState):
        if state is None:
            state = KNNetState(
                torch.as_tensor(0.0).to(self.hidden_weights.device()),
                torch.as_tensor(0.0).to(self.hidden_weights.device()),
                torch.as_tensor(0.0).to(self.hidden_weights.device()),
            )
