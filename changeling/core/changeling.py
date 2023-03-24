from torch import nn, Tensor


class Changeling(nn.Module):
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def prep_lesson(self, name: str) -> None:
        raise NotImplementedError
