"""Reference values for test/models.test.js (NumPy / scikit-learn).
    python test/ref_models.py"""
import json, numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
rng = np.random.default_rng(3)
X = rng.normal(size=(300, 5)); X[:, 4] = (rng.random(300) < 0.4).astype(float)
y = 2.0 + X @ np.array([1.5, -0.7, 0.0, 3.2, -1.1]) + rng.normal(0, 0.5, 300)
A = np.c_[np.ones(len(X)), X]
w = np.linalg.lstsq(A, y, rcond=None)[0]
pred = A @ w
act = np.exp(y / 3) * 1000; prd = np.exp(pred / 3) * 1000
Xq = rng.normal(size=(40, 5))
knn = {f"{w}_{k}": KNeighborsRegressor(n_neighbors=k, weights=w).fit(X, y).predict(Xq).tolist()
       for w in ("uniform", "distance") for k in (1, 5, 12)}
json.dump({"Xq": Xq.tolist(), "knn": knn, "X": X.tolist(), "y": y.tolist(), "w": w.tolist(), "act": act.tolist(), "prd": prd.tolist(),
           "mae": mean_absolute_error(act, prd), "rmse": mean_squared_error(act, prd) ** 0.5,
           "r2": r2_score(act, prd), "mape": mean_absolute_percentage_error(act, prd)},
          open("test/models_reference.json", "w"))
print("ok", np.round(w, 4))
