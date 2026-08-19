from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import random
import joblib
import pandas as pd
from pathlib import Path


# =================================================
# FASTAPI APPLICATION
# =================================================

app = FastAPI(title="EV AI Backend")


# =================================================
# CORS
# =================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =================================================
# LOAD ML MODEL
# =================================================

MODEL_PATH = Path(__file__).parent / "ev_energy_model.pkl"

model_data = joblib.load(MODEL_PATH)

ml_model = model_data["model"]
features = model_data["features"]

print("========================================")
print("ML MODEL LOADED SUCCESSFULLY")
print("Model:", type(ml_model).__name__)
print("Features:", features)
print("========================================")


# =================================================
# BASIC APIs
# =================================================

@app.get("/")
def root():
    return {
        "project": "EV AI Backend",
        "status": "Backend is running",
        "ml_model": "Random Forest",
        "features": features
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ml_model": "loaded"
    }


# =================================================
# THERMAL RISK FUNCTION
# =================================================

def calculate_thermal_risk(battery_temp):

    if battery_temp < 40:
        return "NORMAL"

    elif battery_temp < 45:
        return "WARNING"

    else:
        return "CRITICAL"


# =================================================
# ECO-SPEED RECOMMENDATION
# =================================================

def calculate_eco_speed(
    speed,
    battery_temp,
    road_slope,
    traffic,
    hvac
):

    recommended_speed = speed

    # High battery temperature
    if battery_temp >= 45:
        recommended_speed -= 15

    elif battery_temp >= 40:
        recommended_speed -= 10

    # Uphill road
    if road_slope >= 5:
        recommended_speed -= 10

    elif road_slope >= 2:
        recommended_speed -= 5

    # Heavy traffic
    if traffic == 3:
        recommended_speed -= 5

    # HVAC ON
    if hvac == 1:
        recommended_speed -= 3

    # Keep speed within reasonable limits
    recommended_speed = max(30, recommended_speed)

    recommended_speed = min(80, recommended_speed)

    return int(recommended_speed)


# =================================================
# LIVE WEBSOCKET
# =================================================

@app.websocket("/live")
async def live_data(websocket: WebSocket):

    await websocket.accept()

    print("WebSocket client connected")

    try:

        while True:

            # =========================================
            # SIMULATED EV SENSOR INPUTS
            # Later these will come from ESP32
            # =========================================

            battery_temp = round(
                random.uniform(32, 46), 1
            )

            ambient_temp = round(
                random.uniform(20, 38), 1
            )

            speed = random.randint(40, 80)

            acceleration = round(
                random.uniform(-1.5, 2.0), 2
            )

            hvac = random.randint(0, 1)

            road_slope = round(
                random.uniform(-5, 8), 2
            )

            traffic = random.randint(1, 3)

            vehicle_load = round(
                random.uniform(500, 1200), 1
            )

            road_condition = random.randint(1, 3)

            soc = random.randint(55, 70)


            # =========================================
            # PREPARE DATA FOR ML MODEL
            # =========================================

            input_data = pd.DataFrame(
                [[
                    battery_temp,
                    ambient_temp,
                    speed,
                    acceleration,
                    hvac,
                    road_slope,
                    traffic,
                    vehicle_load,
                    road_condition,
                    soc
                ]],
                columns=features
            )


            # =========================================
            # RANDOM FOREST ML PREDICTION
            # =========================================

            predicted_energy = ml_model.predict(
                input_data
            )[0]

            predicted_energy = round(
                float(predicted_energy),
                1
            )


            # =========================================
            # REMAINING RANGE CALCULATION
            #
            # Trial battery capacity = 60 kWh
            # =========================================

            battery_capacity_wh = 60000

            remaining_energy_wh = (
                battery_capacity_wh * soc / 100
            )

            # Prevent division by zero
            if predicted_energy > 0:

                remaining_range = round(
                    remaining_energy_wh / predicted_energy,
                    1
                )

            else:

                remaining_range = 0


            # =========================================
            # THERMAL RISK
            # =========================================

            thermal_risk = calculate_thermal_risk(
                battery_temp
            )


            # =========================================
            # ECO-SPEED RECOMMENDATION
            # =========================================

            eco_speed = calculate_eco_speed(
                speed,
                battery_temp,
                road_slope,
                traffic,
                hvac
            )


            # =========================================
            # HVAC IMPACT
            # =========================================

            if hvac == 1:

                hvac_status = "ON"

                # Simple prototype estimation
                hvac_impact = 2.0

            else:

                hvac_status = "OFF"

                hvac_impact = 0.0


            # =========================================
            # FINAL DATA
            # =========================================

            data = {

                "speed": speed,

                "soc": soc,

                "battery_temp": battery_temp,

                "ambient_temp": ambient_temp,

                "acceleration": acceleration,

                "hvac": hvac_status,

                "hvac_impact": hvac_impact,

                "road_slope": road_slope,

                "traffic": traffic,

                "vehicle_load": vehicle_load,

                "road_condition": road_condition,

                "energy_consumption": predicted_energy,

                "range": remaining_range,

                "thermal_risk": thermal_risk,

                "eco_speed": eco_speed,

                "ml_prediction": True
            }


            # =========================================
            # SEND DATA
            # =========================================

            await websocket.send_json(data)

            print("Sending:", data)


            # =========================================
            # UPDATE EVERY 2 SECONDS
            # =========================================

            await asyncio.sleep(2)


    except Exception as e:

        print(
            "WebSocket disconnected:",
            e
        )