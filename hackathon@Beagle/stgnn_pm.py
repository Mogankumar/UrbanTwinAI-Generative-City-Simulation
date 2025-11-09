# stgnn_pm.py
import os
import torch, torch.nn as nn
import numpy as np, joblib

FEATS = ["building_cov","green_cov","road_den","impervious","traffic_level","uhi_raw"]

class DenseGCN(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, 1)
        self.act  = nn.ReLU()

    @staticmethod
    def norm_adj(adj):
        if hasattr(adj, "toarray"):
            A = adj.toarray().astype("float32")
        else:
            A = np.asarray(adj, dtype="float32")
        A = A + np.eye(A.shape[0], dtype="float32")   # self-loops
        d = A.sum(1)
        d_inv_sqrt = np.power(d, -0.5, where=d > 0).astype("float32")
        D = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
        return torch.from_numpy(D)

    def forward(self, x, adj_norm):
        h = self.act(adj_norm @ self.lin1(x))
        y = self.lin2(adj_norm @ h)
        return y.squeeze(-1)

class PMSTGNN:
    def __init__(self, model_path=None, scaler_path=None):
        self.device = torch.device("cpu")
        self.model  = DenseGCN(len(FEATS)).to(self.device).eval()

        # optional scaler
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception:
                self.scaler = None

        # optional weights
        if model_path and os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
            except Exception:
                pass  # run with random init

    def _make_features(self, df):
        X = df.reindex(columns=FEATS, fill_value=0.0).values.astype("float32")
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception:
                pass
        return X

    @staticmethod
    def _dense_adj(adj):
        if hasattr(adj, "toarray") or isinstance(adj, np.ndarray):
            return DenseGCN.norm_adj(adj)
        N = len(adj)
        A = np.zeros((N, N), dtype="float32")
        for i, nbrs in enumerate(adj):
            A[i, nbrs] = 1.0
        return DenseGCN.norm_adj(A)

    def predict_map(self, base, scenario, adj, T=6, weather=None, traffic=None):
        xb = torch.from_numpy(self._make_features(base)).to(self.device)
        xs = torch.from_numpy(self._make_features(scenario)).to(self.device)
        A  = self._dense_adj(adj).to(self.device)

        with torch.no_grad():
            yb = self.model(xb, A).cpu().numpy()
            ys = self.model(xs, A).cpu().numpy()
        return ys.astype("float32"), float((ys - yb).mean())