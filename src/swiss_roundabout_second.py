import os
import requests
import pandas as pd
import numpy as np

# ==========================================
# Function to load trajectories from a CSV file (same as before)
# ==========================================
def load_swiss_trajectories(csv_path):
    """Load CSV with columns: track_id, x, y, time, ..."""
    df = pd.read_csv(csv_path)
    print(f"Columns: {df.columns.tolist()}")
    
    # Expect 'track_id', 'x', 'y' (or 'lon', 'lat')
    id_col = 'track_id' if 'track_id' in df.columns else 'id'
    x_col = 'x' if 'x' in df.columns else 'lon'
    y_col = 'y' if 'y' in df.columns else 'lat'
    time_col = 'time' if 'time' in df.columns else None
    
    if time_col:
        df = df.sort_values(time_col)
    
    trajectories = []
    for agent_id, group in df.groupby(id_col):
        positions = group[[x_col, y_col]].values.astype(np.float32)
        if len(positions) >= 10:
            trajectories.append({
                "agent_id": agent_id,
                "positions": positions
            })
    print(f"Loaded {len(trajectories)} agents from {csv_path}.")
    return trajectories

# ==========================================
# Download the second dataset if not already present
# ==========================================
url_swiss_2 = "https://zenodo.org/records/15077435/files/D2_PM1_L1.csv?download=1"
second_file = "D2_PM1_L1.csv"

if not os.path.exists(second_file):
    print("Downloading second Swiss Roundabout dataset sample...")
    response = requests.get(url_swiss_2, stream=True)
    with open(second_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
else:
    print("File already exists.")

# Load trajectories from the new file
trajectories_2 = load_swiss_trajectories(second_file)

# Now you can use trajectories_2 for further analysis
print(f"Total agents in second dataset: {len(trajectories_2)}")
