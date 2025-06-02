from enum import IntEnum
from pprint import pformat

import torch
from pytorch3d.transforms import axis_angle_to_matrix
from torch.nn import Module, Parameter
from torch.nn.functional import normalize

ArticulationType = IntEnum("ArticulationType", ("REVOLUTE", "PRISMATIC", "CYLINDRICAL", "UNDEFINED"))


class ArticulationParams(Module):
    def __init__(self, input_dict, trainable=True, requires_grad=True):
        super().__init__()
        d = self.get_empty_dict()
        d.update(input_dict)
        self.register_buffer("articulation_type", torch.as_tensor(d["type"]))
        self.register_buffer(
            "is_valid",
            torch.tensor(
                d["type"] == ArticulationType.REVOLUTE
                and {"pivot", "axis", "angle"}.issubset(input_dict)
                or d["type"] == ArticulationType.PRISMATIC
                and {"axis", "dist"}.issubset(input_dict)
                or d["type"] == ArticulationType.CYLINDRICAL
                and {"pivot", "axis", "angle", "dist"}.issubset(input_dict)
            ),
        )
        for k in {"pivot", "axis", "angle", "dist"}:
            t = torch.as_tensor(d[k])
            if trainable:
                self.register_parameter(k, Parameter(t, requires_grad=requires_grad))
            else:
                self.register_buffer(k, t)

    @staticmethod
    def get_empty(trainable=False):
        return ArticulationParams(ArticulationParams.get_empty_dict(), trainable=trainable)

    @staticmethod
    def get_empty_dict():
        return {
            "type": ArticulationType.UNDEFINED,
            "pivot": torch.empty(3),
            "axis": torch.empty(3),
            "angle": torch.empty(()),
            "dist": torch.empty(()),
        }

    def randomize(self):
        if self.articulation_type != ArticulationType.PRISMATIC:
            self.pivot.data.normal_(0, 3e-1)
        self.axis.data.normal_()
        self.axis.data /= self.axis.norm()
        self.angle.data.uniform_(-torch.pi, torch.pi)
        self.dist.data.normal_(0, 1e-1)
        self.is_valid.data.fill_(True)
        return self

    def update(self, other):
        for k in {"articulation_type", "is_valid", "pivot", "axis", "angle", "dist"}:
            getattr(self, k).data.copy_(getattr(other, k))
        return self

    def to_matrix(self):
        assert self.is_valid
        T = torch.eye(4, device=self.axis.device)
        axis = normalize(self.axis, dim=0)
        if self.articulation_type != ArticulationType.PRISMATIC:
            R = axis_angle_to_matrix(axis * self.angle)
            T[:3, :3] = R
            T[:3, 3] = (torch.eye(3, device=axis.device) - R) @ self.pivot
        if self.articulation_type != ArticulationType.REVOLUTE:
            T[:3, 3] += axis * self.dist
        return T

    def __str__(self):
        return pformat(
            {
                "type": ArticulationType(self.articulation_type.item()),
                "pivot": self.pivot.tolist(),
                "axis": self.axis.tolist(),
                "angle": self.angle.item(),
                "dist": self.dist.item(),
            },
            sort_dicts=False,
        )
