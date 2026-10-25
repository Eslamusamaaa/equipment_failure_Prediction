"""Stronger attempt: domain features + better models + per-class threshold tuning.
All tuning uses out-of-fold predictions on the TRAIN split; the test split is only used for the final report."""
import os, joblib, numpy as np, pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'equipment_failure_Prediction', 'predictive_maintenance.csv')
OUT = os.path.join(HERE, 'equipment_failure_Prediction')
from features import RAW, add_derived


def models():
    return {
        'rf': lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=1, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1),
        'et': lambda: ExtraTreesClassifier(n_estimators=400, class_weight='balanced_subsample', random_state=SEED, n_jobs=-1),
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6, class_weight='balanced', random_state=SEED),
    }


def fit(make, X, y):
    Xr, yr = RandomOverSampler(random_state=SEED).fit_resample(X, y)   # oversample every class to majority
    return make().fit(Xr, yr)


def tune_weights(P, y, n):
    w = np.ones(n)
    best = f1_score(y, (P * w).argmax(1), average='macro')
    for _ in range(3):
        for c in range(n):
            for v in [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20]:
                t = w.copy(); t[c] = v
                s = f1_score(y, (P * t).argmax(1), average='macro')
                if s > best + 1e-9: best, w = s, t
    return w, best


def main():
    data = pd.read_csv(DATA)
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    le = LabelEncoder(); y = le.fit_transform(data['Failure Type'])
    X = add_derived(data)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    Xtr = Xtr.reset_index(drop=True); n = len(le.classes_)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    oof = {}
    for name, mk in models().items():
        P = np.zeros((len(ytr), n))
        for a, b in skf.split(Xtr, ytr):
            P[b] = fit(mk, Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])
        oof[name] = P
        print(name, 'OOF macro-F1 (argmax):', round(f1_score(ytr, P.argmax(1), average='macro'), 4))
    oof['ens'] = sum(oof.values()) / len(oof)

    cand = {}
    for name, P in oof.items():
        w, s = tune_weights(P, ytr, n)
        cand[name] = (w, s)
        print(name, 'OOF macro-F1 (tuned weights):', round(s, 4), np.round(w, 2))
    best = max(cand, key=lambda k: cand[k][1])
    w = cand[best][0]
    print('\nSelected:', best)

    fitted = {k: fit(mk, Xtr, ytr) for k, mk in models().items()}
    def proba(Xs):
        ps = {k: m.predict_proba(Xs) for k, m in fitted.items()}
        return sum(ps.values()) / len(ps) if best == 'ens' else ps[best]
    pred = (proba(Xte) * w).argmax(1)
    print(classification_report(yte, pred, target_names=le.classes_, zero_division=0))
    print(confusion_matrix(yte, pred))
    print('test macro-F1', round(f1_score(yte, pred, average='macro'), 4), 'accuracy', round((pred == yte).mean(), 4))


if __name__ == '__main__':
    main()
