import torch
from typing import NamedTuple, Optional, Tuple

class KNNetParameters(NamedTuple):
    # shoud be tensor since these each neurons have each parameters
    eps: torch.Tensor = torch.as_tensor(0.015)
    beta: torch.Tensor = torch.as_tensor(0.018)
    d: torch.Tensor = torch.as_tensor(0.26)
    a: torch.Tensor = torch.as_tensor(0.25)


#class KNNetState(NamedTuple):
    
