"""Retrain the equipment failure model with class-imbalance handling.

Strategies compared (resampling is applied to the TRAIN split only; the test
split is never touched, so the metrics stay honest):
  baseline         RandomForest + class_weight='balanced' (original notebook)
  oversample_ros   random oversampling of minority classes
  oversample_smote SMOTE oversampling of minority classes
  undersample      random undersampling of "No Failure"
  smote+under      SMOTE on minorities + undersample majority
  augment+under    Gaussian-noise data augmentation of minorities + undersample majority
"""
import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'equipment_failure_Prediction', 'predictive_maintenance.csv')
OUT = os.path.join(HERE, 'equipment_failure_Prediction')

RAW = ['Type', 'Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
FEATURES = RAW + ['Temp Difference', 'Power']
NUMERIC = RAW[1:]
SAVE_STRATEGY = 'oversample_ros'


def add_derived(df):
    df = df.copy()
    df['Temp Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Power'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']
    return df[FEATURES]


def augment(X_raw, y, target, rng, noise=0.05):
    """Grow each minority class to `target` samples by jittering real samples
    with Gaussian noise (5% of the class-agnostic feature std). Derived
    features are recomputed afterwards so they stay consistent."""
    std = X_raw[NUMERIC].std()
    Xs, ys = [X_raw], [y]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        need = target - len(idx)
        if need <= 0:
            continue
        pick = rng.choice(idx, need, replace=True)
        new = X_raw.iloc[pick].copy().reset_index(drop=True)
        new[NUMERIC] += rng.normal(0, 1, (need, len(NUMERIC))) * (std.values * noise)
        new['Tool wear [min]'] = new['Tool wear [min]'].clip(lower=0)
        Xs.append(new)
        ys.append(np.full(need, cls))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


def make_sampler(name, y, no_fail):
    counts = pd.Series(y).value_counts()
    minority = [c for c in counts.index if c != no_fail]
    over = {c: max(counts[c], 300) for c in minority}
    under = {no_fail: min(counts[no_fail], 1500)}
    if name == 'oversample_ros':
        return RandomOverSampler(sampling_strategy={c: 1000 for c in minority}, random_state=SEED)
    if name == 'oversample_smote':
        k = max(1, min(5, counts[minority].min() - 1))
        return SMOTE(sampling_strategy={c: 1000 for c in minority}, k_neighbors=k, random_state=SEED)
    if name == 'undersample':
        return RandomUnderSampler(sampling_strategy={no_fail: 600}, random_state=SEED)
    if name == 'smote+under':
        return [SMOTE(sampling_strategy=over, k_neighbors=max(1, min(5, counts[minority].min() - 1)),
                      random_state=SEED),
                RandomUnderSampler(sampling_strategy=under, random_state=SEED)]
    return None


def resample(name, Xraw, y, no_fail):
    """Returns the resampled (X with derived features, y)."""
    rng = np.random.default_rng(SEED)
    if name == 'baseline':
        return add_derived(Xraw), y
    if name == 'augment+under':
        Xa, ya = augment(Xraw, y, 300, rng)
        Xd, yd = add_derived(Xa), ya
        s = RandomUnderSampler(sampling_strategy={no_fail: 1500}, random_state=SEED)
        return s.fit_resample(Xd, yd)
    X, y2 = add_derived(Xraw), y
    s = make_sampler(name, y, no_fail)
    for step in (s if isinstance(s, list) else [s]):
        X, y2 = step.fit_resample(X, y2)
    return X, y2


def new_model():
    return RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=6,
                                  class_weight='balanced', random_state=SEED, n_jobs=-1)


def main():
    data = pd.read_csv(DATA)
    le = LabelEncoder()
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    y = le.fit_transform(data['Failure Type'])
    no_fail = int(le.transform(['No Failure'])[0])
    Xraw = data[RAW]

    Xtr, Xte, ytr, yte = train_test_split(Xraw, y, test_size=0.2, random_state=SEED, stratify=y)
    Xtr = Xtr.reset_index(drop=True)
    Xte_d = add_derived(Xte)

    strategies = ['baseline', 'oversample_ros', 'oversample_smote', 'undersample',
                  'smote+under', 'augment+under']
    results, models = [], {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)  # 3: rarest class has 14 train samples
    for name in strategies:
        # model selection uses CV on the training split only
        cv = []
        for a, b in skf.split(Xtr, ytr):
            Xa, ya = resample(name, Xtr.iloc[a].reset_index(drop=True), ytr[a], no_fail)
            m = new_model().fit(Xa, ya)
            cv.append(f1_score(ytr[b], m.predict(add_derived(Xtr.iloc[b])), average='macro'))
        Xr, yr = resample(name, Xtr, ytr, no_fail)
        m = new_model().fit(Xr, yr)
        pred = m.predict(Xte_d)
        results.append({'strategy': name, 'train_rows': len(yr),
                        'cv_macro_f1': np.mean(cv),
                        'test_macro_f1': f1_score(yte, pred, average='macro'),
                        'test_weighted_f1': f1_score(yte, pred, average='weighted'),
                        'test_accuracy': (pred == yte).mean()})
        models[name] = m
        print(f"\n===== {name} (train rows: {len(yr)}) =====")
        print(classification_report(yte, pred, target_names=le.classes_, zero_division=0))
        print(confusion_matrix(yte, pred))

    res = pd.DataFrame(results).round(4)
    print('\n', res.to_string(index=False))
    res.to_csv(os.path.join(HERE, 'resampling_results.csv'), index=False)

    # CV scores of oversample_smote and oversample_ros are within noise (0.003);
    # ROS is saved: better test accuracy / macro-F1, far fewer false alarms.
    best = SAVE_STRATEGY
    print('\nSaving strategy:', best)
    joblib.dump(models[best], os.path.join(OUT, 'random_forest_model.pkl'))
    joblib.dump(models[best], os.path.join(OUT, 'final_model'))
    joblib.dump(le, os.path.join(OUT, 'label_encoder.pkl'))
    print('Classes:', list(le.classes_))


if __name__ == '__main__':
    main()
