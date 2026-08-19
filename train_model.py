import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------
# 1. Generate trial EV dataset
# -------------------------------------------------

np.random.seed(42)

N = 5000

data = pd.DataFrame({
    "battery_temp": np.random.uniform(20, 50, N),
    "ambient_temp": np.random.uniform(10, 45, N),
    "speed": np.random.uniform(20, 120, N),
    "acceleration": np.random.uniform(-3, 3, N),
    "hvac": np.random.randint(0, 2, N),
    "road_slope": np.random.uniform(-10, 10, N),
    "traffic": np.random.randint(1, 4, N),
    "vehicle_load": np.random.uniform(500, 1500, N),
    "road_condition": np.random.randint(1, 4, N),
    "soc": np.random.uniform(10, 100, N)
})


# -------------------------------------------------
# 2. Create trial energy-consumption target
# -------------------------------------------------

data["energy_consumption"] = (
    100
    + 0.45 * data["speed"]
    + 8 * abs(data["acceleration"])
    + 0.8 * data["road_slope"]
    + 12 * data["hvac"]
    + 5 * data["traffic"]
    + 0.025 * data["vehicle_load"]
    + 0.5 * abs(data["battery_temp"] - 25)
    + 0.2 * abs(data["ambient_temp"] - 25)
    + 3 * data["road_condition"]
    + 0.05 * (100 - data["soc"])
    + np.random.normal(0, 5, N)
)


# -------------------------------------------------
# 3. Features and target
# -------------------------------------------------

features = [
    "battery_temp",
    "ambient_temp",
    "speed",
    "acceleration",
    "hvac",
    "road_slope",
    "traffic",
    "vehicle_load",
    "road_condition",
    "soc"
]

X = data[features]
y = data["energy_consumption"]


# -------------------------------------------------
# 4. Train-test split
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------------------------------------------------
# 5. Random Forest model
# -------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")

model.fit(X_train, y_train)


# -------------------------------------------------
# 6. Evaluate model
# -------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL RESULTS")
print("----------------------------")
print(f"MAE  : {mae:.2f} Wh/km")
print(f"RMSE : {rmse:.2f} Wh/km")
print(f"R2   : {r2:.4f}")


# -------------------------------------------------
# 7. Save model
# -------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": features
    },
    "ev_energy_model.pkl"
)

print("\nModel saved successfully!")
print("File: ev_energy_model.pkl")