# train_dummy_traffic.py
import numpy as np, joblib, lightgbm as lgb
from pathlib import Path
from geo_utils import make_bbox, fetch_osm, grid_bbox, features

# Save into <repo_root>/artifacts/ regardless of where you run from
ART_DIR = Path(__file__).resolve().parent / "artifacts"
ART_DIR.mkdir(exist_ok=True)
ART_PATH = ART_DIR / "traffic_lgbm.pkl"

def featurize(df):
    X = df[["road_den","int_den","building_cov","impervious"]].copy()
    X["road_den_sq"] = X["road_den"]**2
    X["int_den_sq"]  = X["int_den"]**2
    X["roadxint"]    = X["road_den"] * X["int_den"]
    X["log_road_den"]= np.log1p(X["road_den"])
    return X

def build_sample(lat=43.0, lon=-78.79, km=1.0):
    bbox = make_bbox(lat, lon, km)
    bld, roads, green, poly_m = fetch_osm(bbox)
    grid = grid_bbox(poly_m, cell=50)
    return features(grid, bld, roads, green)

if __name__ == "__main__":
    print("[INFO] Building sample grid...")
    df = build_sample()
    print("[INFO] Grid rows:", len(df))

    X = featurize(df)
    # synthetic target in [0,1] just to produce a model file
    y = (0.5*df["road_den"] + 0.3*df["int_den"] + 0.2*df["building_cov"]).to_numpy()
    y = (y - y.min()) / (y.max() - y.min() + 1e-9)
    y = np.clip(y + 0.05*np.random.randn(len(y)), 0, 1)

    dtrain = lgb.Dataset(X, label=y)
    params = dict(objective="regression", metric="l2", learning_rate=0.1, num_leaves=31)
    print("[INFO] Training LightGBM...")
    model = lgb.train(params, dtrain, num_boost_round=120)

    joblib.dump(model, ART_PATH.as_posix())
    print(f"[OK] Saved {ART_PATH}")