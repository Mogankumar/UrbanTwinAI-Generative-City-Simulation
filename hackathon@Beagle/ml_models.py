# ml_models.py
import numpy as np
import joblib
from utils.google_dm import dist_matrix

TRAFFIC_FEATS = [
    "road_den","int_den","building_cov","impervious",
    "road_den_sq","int_den_sq","roadxint","log_road_den"
]

def _featurize_traffic(gdf):
    X = gdf[["road_den","int_den","building_cov","impervious"]].copy()
    X["road_den_sq"] = X["road_den"]**2
    X["int_den_sq"]  = X["int_den"]**2
    X["roadxint"]    = X["road_den"] * X["int_den"]
    X["log_road_den"]= np.log1p(X["road_den"])
    return X.reindex(columns=TRAFFIC_FEATS)

class TrafficML:
    """Predicts delay fraction 0..1 per grid cell."""
    def __init__(self, model_path="artifacts/traffic_lgbm.pkl"):
        self.model = joblib.load(model_path)
    def predict(self, grid_gdf):
        X = _featurize_traffic(grid_gdf)
        y = self.model.predict(X)
        return np.clip(y, 0.0, 1.0)