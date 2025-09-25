import numpy as np
import pandas as pd

np.random.seed(42)
num_samples = 50_000

# === Distance to frontline between 60 and 160 km ===
distance_to_frontline_km = np.random.uniform(60, 160, size=num_samples)

# === Movement features ===
duration_hours = np.random.normal(loc=2.0, scale=0.5, size=num_samples).clip(0.5, 6)
distance_km = duration_hours * np.random.uniform(5, 20, size=num_samples)

# === Route type: MSR or ASR ===
route_types = ['MSR', 'ASR']
route_type = np.random.choice(route_types, size=num_samples, p=[0.7, 0.3])

# === Terrain, road, time-of-day as before ===
road_types = ['paved', 'gravel', 'dirt', 'forest']
terrain_types = ['open_plains', 'forest', 'urban', 'hills']
times_of_day = ['night', 'day', 'twilight']

road_type = np.random.choice(road_types, size=num_samples)
terrain_type = np.random.choice(terrain_types, size=num_samples)
time_of_day = np.random.choice(times_of_day, size=num_samples)

elevation_change = np.random.normal(loc=50, scale=40, size=num_samples).clip(0, 200)
vegetation_density = np.random.beta(2, 5, size=num_samples)

enemy_artillery_density = np.random.exponential(scale=0.5, size=num_samples)
enemy_uav_activity = np.random.poisson(1.5, size=num_samples)
recent_attack_count = np.random.poisson(0.8, size=num_samples)
intel_alert_level = np.random.beta(2, 2, size=num_samples)
electronic_interference = np.random.beta(1.5, 4, size=num_samples)
traffic_density = np.random.poisson(10, size=num_samples)
weather_visibility = np.random.normal(loc=6.0, scale=2.0, size=num_samples).clip(0.1, 10)

# === Encode categoricals except route_type ===
road_type_encoded = pd.get_dummies(road_type, prefix='road')
terrain_type_encoded = pd.get_dummies(terrain_type, prefix='terrain')
time_of_day_encoded = pd.get_dummies(time_of_day, prefix='tod')

# === Convert route_type to binary: MSR=1, ASR=0 ===
route_type_binary = (route_type == 'MSR').astype(int)

# === Build DataFrame of features ===
df = pd.DataFrame({
    'distance_to_frontline_km': distance_to_frontline_km,
    'duration_hours': duration_hours,
    'distance_km': distance_km,
    'elevation_change': elevation_change,
    'vegetation_density': vegetation_density,
    'enemy_artillery_density': enemy_artillery_density,
    'enemy_uav_activity': enemy_uav_activity,
    'recent_attack_count': recent_attack_count,
    'intel_alert_level': intel_alert_level,
    'electronic_interference': electronic_interference,
    'traffic_density': traffic_density,
    'weather_visibility': weather_visibility,
    'route_type_binary': route_type_binary
})

df = pd.concat([
    df,
    road_type_encoded,
    terrain_type_encoded,
    time_of_day_encoded
], axis=1)

# === Threat component risk formulas using binary route_type ===
artillery_risk = 0.3 * (enemy_artillery_density / (1 + distance_to_frontline_km)) \
                 * (1 + 0.2 * route_type_binary)

uav_risk = 0.2 * (enemy_uav_activity / 10) * (1 + 0.3 * route_type_binary) \
           * (1 - vegetation_density)

ambush_risk = 0.25 * (recent_attack_count / 5) * (1 + 0.5 * (1 - route_type_binary)) \
              * (terrain_type_encoded['terrain_forest'].values + terrain_type_encoded['terrain_urban'].values)

mine_risk = 0.15 * (1 / (1 + weather_visibility)) * (1 + 0.5 * (1 - route_type_binary))

base_risk = 0.1 * intel_alert_level + 0.05 * electronic_interference + 0.05 * (1 - weather_visibility/10)

combined_risk = artillery_risk + uav_risk + ambush_risk + mine_risk + base_risk

attack_probability = np.clip(combined_risk, 0, 1)

df['artillery_risk'] = artillery_risk
df['uav_risk'] = uav_risk
df['ambush_risk'] = ambush_risk
df['mine_risk'] = mine_risk
df['base_risk'] = base_risk
df['attack_probability'] = attack_probability

# Save out
csv_path = "synthetic_rear_area_attack_with_route_and_threats.csv"
df.to_csv(csv_path, index=False)

print("Saved synthetic dataset to:", csv_path)
