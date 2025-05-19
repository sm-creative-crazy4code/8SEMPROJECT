"""This module contains functions for loading models."""

import logging
from os import path
from typing import Tuple

import torch

from network.anomaly_detector_model import AnomalyDetector
from network.c3d import C3D
from network.MFNET import MFNET_3D
from network.resnet import generate_model
from network.TorchUtils import TorchModel
from utils.types import Device, FeatureExtractor


def load_feature_extractor(
    features_method: str, feature_extractor_path: str, device: Device
) -> FeatureExtractor:
   
    if not path.exists(feature_extractor_path):
        raise FileNotFoundError(
            f"Couldn't find feature extractor {feature_extractor_path}.\n"
            + r"If you are using resnet, download it first from:\n"
            + r"r3d101: https://drive.google.com/file/d/1p80RJsghFIKBSLKgtRG94LE38OGY5h4y/view?usp=share_link"
            + "\n"
            + r"r3d152: https://drive.google.com/file/d/1irIdC_v7wa-sBpTiBlsMlS7BYNdj4Gr7/view?usp=share_link"
        )
    logging.info(f"Loading feature extractor from {feature_extractor_path}")

    model: FeatureExtractor

    if features_method == "c3d":
        model = C3D(pretrained=feature_extractor_path)
    elif features_method == "mfnet":
        model = MFNET_3D()
        model.load_state(state_dict=feature_extractor_path)
    elif features_method == "r3d101":
        model = generate_model(model_depth=101)
        param_dict = torch.load(feature_extractor_path)["state_dict"]
        param_dict.pop("fc.weight")
        param_dict.pop("fc.bias")
        model.load_state_dict(param_dict)
    elif features_method == "r3d152":
        model = generate_model(model_depth=152)
        param_dict = torch.load(feature_extractor_path)["state_dict"]
        param_dict.pop("fc.weight")
        param_dict.pop("fc.bias")
        model.load_state_dict(param_dict)
    else:
        raise NotImplementedError(
            f"Features extraction method {features_method} not implemented"
        )

    return model.to(device).eval()


def load_anomaly_detector(ad_model_path: str, device: Device) -> AnomalyDetector:
   
    if not path.exists(ad_model_path):
        raise FileNotFoundError(f"Couldn't find anomaly detector {ad_model_path}.")
    logging.info(f"Loading anomaly detector from {ad_model_path}")

    anomaly_detector = TorchModel.load_model(ad_model_path).to(device)
    return anomaly_detector.eval()


def load_models(
    feature_extractor_path: str,
    ad_model_path: str,
    features_method: str = "c3d",
    device: Device = "cuda",
) -> Tuple[AnomalyDetector, FeatureExtractor]:
   
    feature_extractor = load_feature_extractor(
        features_method, feature_extractor_path, device
    )
    anomaly_detector = load_anomaly_detector(ad_model_path, device)
    return anomaly_detector, feature_extractor
