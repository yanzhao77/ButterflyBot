import os
import json
import joblib
from datetime import datetime

REGISTRY_DIR = "/home/ubuntu/ButterflyBot/models/registry"

def save_model_with_metadata(model, metadata):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v{timestamp}"
    model_path = os.path.join(REGISTRY_DIR, f"{version}.pkl")
    metadata_path = os.path.join(REGISTRY_DIR, f"{version}.json")

    joblib.dump(model, model_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return version

def find_best_model_by_auc():
    best_model = None
    best_auc = -1

    for f in os.listdir(REGISTRY_DIR):
        if f.endswith(".json"):
            with open(os.path.join(REGISTRY_DIR, f), "r") as meta_f:
                metadata = json.load(meta_f)
                if metadata.get("auc", -1) > best_auc:
                    best_auc = metadata["auc"]
                    best_model = f.replace(".json", "")
    return best_model

def update_latest_model(version):
    with open(os.path.join(REGISTRY_DIR, "latest_model.txt"), "w") as f:
        f.write(version)

def load_latest_model_path():
    latest_model_file = os.path.join(REGISTRY_DIR, "latest_model.txt")
    if not os.path.exists(latest_model_file):
        return None
    with open(latest_model_file, "r") as f:
        version = f.read().strip()
        return os.path.join(REGISTRY_DIR, f"{version}.pkl")
