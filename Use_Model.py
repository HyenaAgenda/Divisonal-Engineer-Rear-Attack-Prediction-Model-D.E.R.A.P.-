import pickle
import time
import torch
import numpy as np
from Train_Model import NN_Regression

#-------------------------------------------------------

print("Welcome to the Div Rear Prediction Model")

# Load scalers
with open("scaler_x.pkl", "rb") as f:
    scaler_x = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

# Number of input features = 12 numeric + 4 road + 4 terrain + 3 time + 2 route = 25
input_size = 25
model = NN_Regression(input_size)
model.load_state_dict(torch.load("horizon_attack_model.pt"))
model.eval()

def one_hot_encode(value, categories):
    return [1.0 if value == cat else 0.0 for cat in categories]

#-------------------------------------------------------

print("Please enter the following features:")

distance_to_frontline_km = float(input("Distance to frontline (km) [60-160]: "))
duration_hours = float(input("Duration (hours) [0.5-6]: "))
distance_km = float(input("Distance moved (km): "))
elevation_change = float(input("Elevation change (meters) [0-200]: "))
vegetation_density = float(input("Vegetation density (0-1): "))
enemy_artillery_density = float(input("Enemy artillery density: "))
enemy_uav_activity = float(input("Enemy UAV activity (count): "))
recent_attack_count = float(input("Recent attack count: "))
intel_alert_level = float(input("Intel alert level (0-1): "))
electronic_interference = float(input("Electronic interference (0-1): "))
traffic_density = float(input("Traffic density (count): "))
weather_visibility = float(input("Weather visibility (0.1-10): "))

road_types = ["paved", "gravel", "dirt", "forest track"]
terrain_types = ["open_plains", "forest", "urban", "hills"]
times_of_day = ["night", "day", "twilight"]
route_types = ["msr", "asr"]   # FIXED here, was incorrect before

road_input = input(f"Road type Options: {road_types}\nEnter Choice: ").lower()
terrain_input = input(f"Terrain type Options: {terrain_types}\nEnter Choice: ").lower()
time_of_day_input = input(f"Time Of Day Options: {times_of_day}\nEnter Choice: ").lower()
route_type_input = input(f"Route Type Options: {route_types}\nEnter Choice: ").lower()  # FIXED

# Validate inputs
if road_input not in road_types:
    raise ValueError(f"Invalid road type: {road_input}")
if terrain_input not in terrain_types:
    raise ValueError(f"Invalid terrain type: {terrain_input}")
if time_of_day_input not in times_of_day:
    raise ValueError(f"Invalid time Of Day option: {time_of_day_input}")
if route_type_input not in route_types:
    raise ValueError(f"Invalid route type: {route_type_input}")  # Added validation

#-------------------------------------------------------

road_encoded = one_hot_encode(road_input, road_types)
terrain_encoded = one_hot_encode(terrain_input, terrain_types)  # FIXED here
time_encoded = one_hot_encode(time_of_day_input, times_of_day)

if route_type_input == "msr":
    route_encoded = [1.0, 0.0]
else:
    route_encoded = [0.0, 1.0]

#-------------------------------------------------------

# Construct full input feature vector in correct order:
input_features = [
    distance_to_frontline_km,
    duration_hours,
    distance_km,
    elevation_change,
    vegetation_density,
    enemy_artillery_density,
    enemy_uav_activity,
    recent_attack_count,
    intel_alert_level,
    electronic_interference,
    traffic_density,
    weather_visibility
] + road_encoded + terrain_encoded + time_encoded + route_encoded

input_array = np.array(input_features).reshape(1, -1)

input_scaled = scaler_x.transform(input_array)

# Convert to tensor
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    prediction = model(input_tensor)
    prediction_unscaled = scaler_y.inverse_transform(prediction.numpy())

print("Processed ...")
print(f"\nPrediction: {prediction_unscaled.flatten()[0]}")
print("This is the predicted attack probability (or whatever target your model outputs).")
