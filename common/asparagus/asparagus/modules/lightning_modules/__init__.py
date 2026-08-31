from .clsreg_module import ClassificationModule, RegressionModule
from .dinov2 import DINOv2Module
from .fomo_classification_module import FomoClassificationModule
from .fomo_regression_module import FomoRegressionModule
from .linear_probe_module import LinearProbeModule
from .segmentation_module import SegmentationModule
from .self_supervised import SelfSupervisedModule

__all__ = [
    "SegmentationModule",
    "ClassificationModule",
    "FomoClassificationModule",
    "RegressionModule",
    "FomoRegressionModule",
    "SelfSupervisedModule",
    "DINOv2Module",
    "LinearProbeModule",
]
