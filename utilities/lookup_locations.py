from meteostat import Stations

# Houston suburbs with coordinates
suburbs = {
    "The Woodlands, TX": (30.1658, -95.4613),
    "Katy, TX": (29.7858, -95.8245),
    "Friendswood, TX": (29.5294, -95.2010),
    "Baytown, TX": (29.7355, -94.9774)
}

for name, (lat, lon) in suburbs.items():
    stations = Stations()
    stations = stations.nearby(lat, lon)
    station = stations.fetch(1)  # fetch the closest station
    print(f"\n{name}")
    print(station)

import json

# Dictionary of Houston suburbs with coordinates
suburbs_coords = {
    "The Woodlands, TX": {"lat": 30.1658, "lon": -95.4613},
    "Katy, TX": {"lat": 29.7858, "lon": -95.8245},
    "Friendswood, TX": {"lat": 29.5294, "lon": -95.2010},
    "Baytown, TX": {"lat": 29.7355, "lon": -94.9774}
}

# Save to JSON file
json_path = "../pipeline/houston_suburbs_coords.json"
with open(json_path, "w") as f:
    json.dump(suburbs_coords, f, indent=4)

json_path
