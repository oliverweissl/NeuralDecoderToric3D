"""A collection of pooling layers."""
from ._global_average_pooling import GlobalAveragePooling
from._translational_equivariant_pooling_2d import TranslationalEquivariantPooling2D
from ._translational_equivariant_pooling_3d import TranslationalEquivariantPooling3D

__all__ = [
    "GlobalAveragePooling",
    "TranslationalEquivariantPooling2D",
    "TranslationalEquivariantPooling3D"
]