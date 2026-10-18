"""Population geometry: binning, swappable reducers, Procrustes and subspace metrics."""
from .core import (bin_trial, BaseReducer, PCAReducer, FAReducer, IsomapReducer, REDUCERS,
                   make_reducer, procrustes_disparity, procrustes_distance, principal_angles_deg,
                   alignment_index, projection_score, TrialGeometry)

__all__ = ["bin_trial", "BaseReducer", "PCAReducer", "FAReducer", "IsomapReducer", "REDUCERS",
           "make_reducer", "procrustes_disparity", "procrustes_distance", "principal_angles_deg",
           "alignment_index", "projection_score", "TrialGeometry"]
