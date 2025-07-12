import os
import pickle

def load_models():
    base_path = "models"

    def load_pickle(filename):
        with open(os.path.join(base_path, filename), "rb") as f:
            return pickle.load(f)

    model_low = load_pickle("low_custom.pkl")
    model_mid = load_pickle("mid_custom.pkl")
    model_high = load_pickle("high_custom.pkl")
    duration_model = load_pickle("duration_model.pkl")
    ohe = load_pickle("ohe.pkl")
    ohe_duration = load_pickle("ohe_duration.pkl")
    scaler = load_pickle("scaler.pkl")

    return model_low, model_mid, model_high, duration_model, ohe, ohe_duration, scaler