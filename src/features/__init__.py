"""Feature extraction pipeline."""

from .pipeline import FeatureExtractionPipeline
from .frame_sampler import FrameSampler
from .cnn_extractor import CNNFeatureExtractor

__all__ = ["FeatureExtractionPipeline", "FrameSampler", "CNNFeatureExtractor"]
