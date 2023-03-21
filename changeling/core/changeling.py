import torch
from torch import nn
from typing import Dict

from changeling.core.branch import Branch


class Changeling(nn.Module):
    def __init__(self, branches: nn.ModuleDict):
        super().__init__()
        self.branches = branches

    def activate_all(self):
        for branch in self.branches.values():
            branch.activate()

    def deactivate_all(self):
        for branch in self.branches.values():
            branch.deactivate()

    def activate_branch(self, branch_name: str):
        if branch_name in self.branches:
            self.branches[branch_name].activate()
        else:
            raise ValueError(f"Branch '{branch_name}' not found")

    def deactivate_branch(self, branch_name: str):
        if branch_name in self.branches:
            self.branches[branch_name].deactivate()
        else:
            raise ValueError(f"Branch '{branch_name}' not found")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}
        for name, branch in self.branches.items():
            if branch.active and name in inputs:
                outputs[name] = branch(inputs[name])
        return outputs
