import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

"""delays = np.arange(30000000, 35100000, 100000)
gates = np.arange(5000, 45000, 5000)
x1_grid, x2_grid = np.meshgrid(delays, gates)
X_pred = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])"""

delays = np.arange(30000000, 32000000, 100000)
gates = np.arange(5000, 45000, 5000)
x1_grid, x2_grid = np.meshgrid(delays, gates)
X_pred = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])

data = pd.read_csv("data/training/training.csv")
X = data[["Delay Time", "Gate Time"]].values
y = data["Coincidence Counts"].values

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

predictions = rf.predict(X_pred)
optimal_idx = np.argmax(predictions)
optimal_delay, optimal_gate = X_pred[optimal_idx]
max_coinc = predictions[optimal_idx]
print(
    f"Optimal Delay Time: {optimal_delay}, Optimal Gate Time: {optimal_gate}, Coincidence Count: {max_coinc}"
)

pred_grid = predictions.reshape(len(gates), len(delays))
plt.figure(figsize=(8, 6))
cp = plt.contourf(delays, gates, pred_grid, levels=20, cmap="viridis")
plt.colorbar(cp)
plt.title("Coincidence Count Predictions")
plt.xlabel("Delay Time")
plt.ylabel("Gate Time")
plt.show()
