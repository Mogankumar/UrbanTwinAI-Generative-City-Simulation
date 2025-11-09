# train_pm_stgnn.py
import os, numpy as np, torch, torch.nn as nn, joblib
from stgnn_pm import STGCN, FEATS
from geo_utils import grid_bbox
from shapely.geometry import Polygon

def rook_adjacency(h, w):
    N = h*w
    A = np.zeros((N, N), dtype=int)
    def idx(i,j): return i*w + j
    for i in range(h):
        for j in range(w):
            u = idx(i,j)
            for di, dj in [(1,0),(-1,0),(0,1),(0,-1)]:
                ni,nj = i+di, j+dj
                if 0 <= ni < h and 0 <= nj < w:
                    v = idx(ni,nj); A[u,v] = 1
    return A

def synth_sequence(h=14, w=14, T=24):
    N = h*w
    # random but realistic base
    building_cov = np.random.beta(2,5, N)
    green_cov    = np.random.beta(3,3, N)
    road_den     = np.random.gamma(1.5, 0.002, N)
    impervious   = np.clip(building_cov + 0.02*road_den, 0, 1)
    traffic      = np.clip(0.4*road_den/road_den.max() + 0.1*np.random.rand(N), 0, 1)
    uhi_raw      = 6*impervious - 4*green_cov
    temp_c       = 20 + 5*np.sin(np.linspace(0,2*np.pi,T)).mean()
    wind_ms      = 2.5
    rh           = 55

    X0 = np.column_stack([building_cov, green_cov, road_den, impervious, traffic, uhi_raw,
                          np.full(N,temp_c), np.full(N,wind_ms), np.full(N,rh)])
    seq = [X0]
    for t in range(1, T):
        jitter = np.random.normal(0, 0.01, X0.shape)
        seq.append(np.clip(seq[-1]*(1+jitter), 0, None))
    X = np.stack(seq, 0)  # [T, N, F]

    # teacher label (pm): c0 + c1*traffic + c2*(1-green) + c3*uhi + noise
    pm = 7.0 + 3.0*X[...,4].mean(1) + 2.0*(1-X[...,1]).mean(1) + 0.3*X[...,5].mean(1)
    y  = pm[-1]  # next-step/global
    return X, y

def main():
    os.makedirs("artifacts", exist_ok=True)
    # dataset
    Xs, ys = [], []
    for _ in range(400):
        X,y = synth_sequence()
        Xs.append(X); ys.append(y)
    Xs = np.stack(Xs,0)   # [B, T, N, F]
    ys = np.array(ys, dtype=np.float32)

    # scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    B,T,N,F = Xs.shape
    Xs2 = scaler.fit_transform(Xs.reshape(-1, F)).reshape(B,T,N,F)

    # model
    model = STGCN(in_dim=len(FEATS))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    # adjacency (14x14 grid)
    A = rook_adjacency(14,14)
    import torch as th
    edge_index = th.tensor(np.vstack(np.nonzero(A)), dtype=th.long)

    Xs_t = th.tensor(Xs2, dtype=th.float32)
    ys_t = th.tensor(ys, dtype=th.float32).unsqueeze(1)

    for epoch in range(80):
        optim.zero_grad()
        pred = model(Xs_t, edge_index)        # [B,1]
        loss = lossf(pred, ys_t)
        loss.backward()
        optim.step()
        if (epoch+1) % 10 == 0:
            print(f"[epoch {epoch+1}] loss={loss.item():.4f}")

    torch.save(model.state_dict(), "artifacts/pm_stgnn.pt")
    joblib.dump(scaler, "artifacts/pm_scaler.pkl")
    print("[OK] saved artifacts/pm_stgnn.pt and pm_scaler.pkl")

if __name__ == "__main__":
    main()