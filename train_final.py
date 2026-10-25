"""Final pipeline: 5 learnable classes (Random Failures excluded) + Tool Wear alert rule.

Random Failures (18 rows, all Target=0, ordinary sensor values) is random by construction
in this dataset, so it is dropped from training and evaluation and reported separately.
Tool Wear alert: if the model says "No Failure" but tool wear >= WEAR_ALERT, flag Tool Wear.
Test split is only used for the final report; weights/selection use train OOF predictions.
"""
import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import train_v2 as t

WEAR_ALERT = 198


def predict_labels(P, w, wear, classes):
    pred = (P * w).argmax(1)
    nf, tw = classes.index('No Failure'), classes.index('Tool Wear Failure')
    pred = np.where((pred == nf) & (wear >= WEAR_ALERT), tw, pred)
    return pred


def main():
    data = pd.read_csv(t.DATA)
    n_random = (data['Failure Type'] == 'Random Failures').sum()
    data = data[data['Failure Type'] != 'Random Failures'].reset_index(drop=True)
    data['Type'] = LabelEncoder().fit_transform(data['Type'])
    le = LabelEncoder(); y = le.fit_transform(data['Failure Type']); classes = list(le.classes_)
    X = t.add_derived(data); n = len(classes)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=t.SEED, stratify=y)
    Xtr = Xtr.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=t.SEED)

    oof = {}
    for name, mk in t.models().items():
        P = np.zeros((len(ytr), n))
        for a, b in skf.split(Xtr, ytr):
            P[b] = t.fit(mk, Xtr.iloc[a], ytr[a]).predict_proba(Xtr.iloc[b])
        oof[name] = P
    P_ens = sum(oof.values()) / len(oof)
    w, s = t.tune_weights(P_ens, ytr, n)
    oof_pred = predict_labels(P_ens, w, Xtr['wear'].values, classes)
    print('train OOF macro-F1 (5 classes, with alert):', round(f1_score(ytr, oof_pred, average='macro'), 4))

    fitted = {k: t.fit(mk, Xtr, ytr) for k, mk in t.models().items()}
    P = sum(m.predict_proba(Xte) for m in fitted.values()) / len(fitted)
    for label, pred in [('model only', (P * w).argmax(1)),
                        ('model + Tool Wear alert', predict_labels(P, w, Xte['wear'].values, classes))]:
        print(f'\n===== {label} =====')
        print(classification_report(yte, pred, target_names=classes, zero_division=0))
        print(confusion_matrix(yte, pred))
        print('TEST macro-F1 (5 classes):', round(f1_score(yte, pred, average='macro'), 4),
              '| accuracy:', round((pred == yte).mean(), 4))
    print(f'\n(Random Failures: {n_random} rows excluded as unpredictable)')

    final = {k: t.fit(mk, X, y) for k, mk in t.models().items()}
    joblib.dump({'models': final, 'weights': w, 'features': list(X.columns), 'classes': classes,
                 'wear_alert': WEAR_ALERT}, os.path.join(t.OUT, 'model_final.pkl'))
    joblib.dump(le, os.path.join(t.OUT, 'label_encoder_final.pkl'))
    print('saved model_final.pkl')


if __name__ == '__main__':
    main()
