"""Chronological HMDA baseline. Run from the repository root: python -m src.baseline."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import importlib.metadata
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['loan_amount_000s', 'applicant_income_000s']
CATEGORICAL = ['loan_type', 'loan_purpose']
FEATURES = NUMERIC + CATEGORICAL
SEED = 42
K_VALUES = [5, 15, 31, 61, 101]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_year(path, year):
    """Validate source schema and geography; exclude outcomes outside actions 1/2/3."""
    raw = pd.read_csv(path, low_memory=False)
    required = FEATURES + ['as_of_year', 'action_taken', 'state_code']
    if not set(required).issubset(raw.columns):
        raise ValueError(f'Missing columns: {set(required) - set(raw.columns)}')
    if not raw.as_of_year.eq(year).all() or not raw.state_code.eq(32).all():
        raise ValueError('Expected one Nevada year per input file')
    d = raw.loc[raw.action_taken.isin([1, 2, 3])].copy()
    d['approved'] = d.action_taken.isin([1, 2]).astype(int)
    if set(d.approved.unique()) != {0, 1}:
        raise ValueError('Each year must contain both classes')
    for col in FEATURES:
        d[col] = pd.to_numeric(d[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Loan/income values cannot be negative; treat invalid entries as missing.
    for col in NUMERIC:
        d.loc[d[col] < 0, col] = np.nan
    stats = dict(year=year, raw_rows=len(raw), retained_rows=len(d),
        excluded_rows=len(raw)-len(d), approved=int(d.approved.sum()),
        denied=int((d.approved == 0).sum()), approval_rate=float(d.approved.mean()),
        income_missing=int(d.applicant_income_000s.isna().sum()),
        income_missing_rate=float(d.applicant_income_000s.isna().mean()),
        median_loan=float(d.loan_amount_000s.median()),
        median_income=float(d.applicant_income_000s.median()),
        full_row_duplicates=int(raw.duplicated().sum()),
        feature_duplicates=int(d.duplicated(FEATURES).sum()))
    return d, stats


def preprocessor():
    # Fit every learned statistic on 2007 only. Indicator records missing income.
    num = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                    ('scale', StandardScaler())])
    cat = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric', num, NUMERIC), ('categorical', cat, CATEGORICAL)],
                             remainder='drop', sparse_threshold=0)


def evaluate(y, pred, probability):
    # AUC uses approval=1. Denial precision/recall/F1 explicitly use class 0.
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return dict(auc=float(roc_auc_score(y, probability)),
        accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        denial_precision=float(precision_score(y, pred, pos_label=0, zero_division=0)),
        denial_recall=float(recall_score(y, pred, pos_label=0, zero_division=0)),
        denial_f1=float(f1_score(y, pred, pos_label=0, zero_division=0)),
        approval_precision=float(precision_score(y, pred, zero_division=0)),
        approval_recall=float(recall_score(y, pred, zero_division=0)),
        approval_f1=float(f1_score(y, pred, zero_division=0)),
        denial_average_precision=float(average_precision_score(1-y, 1-probability)),
        denied_as_denied=int(cm[0,0]), denied_as_approved=int(cm[0,1]),
        approved_as_denied=int(cm[1,0]), approved_as_approved=int(cm[1,1]))


def run(dataset='random', output=None):
    out = Path(output) if output else ROOT / 'results' / dataset
    out.mkdir(parents=True, exist_ok=True)
    folder, stem = ('Random_Mortgage_Data', 'Random_Mortgage_Data') if dataset == 'random' else ('Cut_Mortgage_Data', 'Cut_Mortage_Data')
    manifest = json.loads((ROOT/'data_manifest.json').read_text())
    frames, audit, hashes = {}, [], {}
    for year in [2007,2008,2009]:
        rel = f'{folder}/{stem}{year%100:02}.csv'
        path = ROOT / rel
        digest = sha256(path)
        if digest != manifest['files'][rel]['sha256']:
            raise ValueError(f'Input checksum mismatch: {rel}')
        hashes[rel] = digest
        frames[year], stats = load_year(path, year)
        audit.append(stats)
    pd.DataFrame(audit).to_csv(out/'data_audit.csv', index=False)
    train, val, test = [frames[y] for y in [2007,2008,2009]]
    pp = preprocessor()
    xt = pp.fit_transform(train[FEATURES])
    xv = pp.transform(val[FEATURES])
    # Freeze k using 2008 AUC before transforming/evaluating the 2009 test set.
    candidates = []
    start = time.perf_counter()
    for k in K_VALUES:
        m = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski', p=2,
                                  algorithm='brute', n_jobs=1).fit(xt, train.approved)
        prob = m.predict_proba(xv)[:, list(m.classes_).index(1)]
        candidates.append({'k': k, 'validation_auc': float(roc_auc_score(val.approved, prob))})
    tuning = pd.DataFrame(candidates)
    tuning.to_csv(out/'validation_search.csv', index=False)
    best_k = int(tuning.sort_values(['validation_auc','k'], ascending=[False, True]).iloc[0].k)
    models = {
        'Majority': DummyClassifier(strategy='most_frequent', random_state=SEED).fit(xt, train.approved),
        f'kNN (k={best_k})': KNeighborsClassifier(n_neighbors=best_k, weights='uniform',
            metric='minkowski', p=2, algorithm='brute', n_jobs=1).fit(xt, train.approved)}
    xx = pp.transform(test[FEATURES])
    rows, predictions, curves = [], {}, {}
    for split, frame, x in [('validation_2008',val,xv), ('test_2009',test,xx)]:
        for name, m in models.items():
            prob = m.predict_proba(x)[:,list(m.classes_).index(1)]
            pred = m.predict(x)
            metrics = evaluate(frame.approved.to_numpy(), pred, prob)
            rows.append(dict(dataset=dataset,split=split,model=name,n=len(frame),**metrics))
            predictions[f'{split}_{name}'] = pd.DataFrame({'source_row':frame.index,
                 'approved':frame.approved.to_numpy(),'prediction':pred,'p_approved':prob})
            predictions[f'{split}_{name}'].to_csv(out/f'predictions_{split}_{"knn" if name.startswith("kNN") else "majority"}.csv.gz', index=False, compression={'method':'gzip','mtime':0})
            curves[(split,name)] = roc_curve(frame.approved,prob)
    results = pd.DataFrame(rows)
    results.to_csv(out/'baseline_table.csv', index=False)
    details = {'dataset':dataset,'best_k':best_k,'k_grid':K_VALUES,'seed':SEED,
        'features':FEATURES,'train_year':2007,'validation_year':2008,'test_year':2009,
        'refit_on_2008':False,'threshold':'sklearn predict: majority vote; odd k',
        'input_sha256':hashes,'python':platform.python_version(),
        'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn']},
        'numeric_imputation_medians':pp.named_transformers_['numeric'].named_steps['impute'].statistics_.tolist(),
        'feature_names':pp.get_feature_names_out().tolist(),
        'elapsed_seconds':round(time.perf_counter()-start,3)}
    (out/'run_metadata.json').write_text(json.dumps(details,indent=2)+'\n')
    # Keep plot generation optional for environments without a graphical display.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    for (split,name),(fpr,tpr,_) in curves.items():
        if split == 'test_2009':
            ax.plot(fpr,tpr,label=name)
    ax.set(xlabel='False positive rate (approval)',ylabel='True positive rate (approval)',title=f'2009 holdout ROC: {dataset} sample')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'test_roc.png',dpi=180); plt.close(fig)
    print(f'\n{dataset.upper()} | training rows={len(train):,} | selected k={best_k}')
    print(results[['split','model','auc','accuracy','denial_precision','denial_recall','denial_f1']].to_string(index=False))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['random','cut','both'], default='random')
    args = parser.parse_args()
    for dataset in (['random','cut'] if args.dataset == 'both' else [args.dataset]):
        run(dataset)
