"""Feature engineering shared by training (train_v2.py / train_final.py) and the Flask app."""
import numpy as np

RAW = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
OVERSTRAIN = np.array([13000, 11000, 12000])  # Type encoded H=0, L=1, M=2


def add_derived(df):
    d = df[RAW].copy()
    d.columns = ['type', 'air', 'proc', 'rpm', 'torque', 'wear']
    d['temp_diff'] = d.proc - d.air
    d['power'] = d.rpm * d.torque                       # kept for the existing app.py
    d['power_w'] = d.torque * d.rpm * 2 * np.pi / 60
    d['wear_torque'] = d.wear * d.torque
    d['overstrain_margin'] = d.wear_torque - OVERSTRAIN[d.type.astype(int)]
    d['wear_ge_200'] = (d.wear >= 200).astype(int)
    d['wear_over_200'] = (d.wear - 200).clip(lower=0)
    d['heat_risk'] = ((d.temp_diff < 8.6) & (d.rpm < 1380)).astype(int)
    d['temp_diff_x_rpm'] = d.temp_diff * d.rpm
    d['power_dev'] = (d.power_w - 6000).abs()
    return d
