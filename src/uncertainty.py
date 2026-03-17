import numpy as np


def compute_uncertainty(probabilities, threshold=0.55):

    confidence = probabilities.max(axis=1)

    uncertain_flag = (confidence < threshold).astype(int)

    return confidence, uncertain_flag