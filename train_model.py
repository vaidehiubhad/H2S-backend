import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

np.random.seed(42)

n_samples = 400
true_dose = np.random.uniform(0.5, 50, n_samples)
noise = np.random.normal(0, 1.2, n_samples)

a, b = 2.1, 0.74
delta_e = a * (true_dose ** b) + noise
delta_e = np.clip(delta_e, 0, None)

X = delta_e.reshape(-1, 1)
y = true_dose

model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
model.fit(X, y)

joblib.dump(model, "dose_model.pkl")
print("Model saved as dose_model.pkl")
