"""Evaluasi ulang tanpa kebocoran data (data leakage).

Protokol yang benar:
  1. Pakai ulasan asli yang unik (dataset_unique_clean.csv).
  2. Split train/test DULU (stratified, random_state=42).
  3. Oversample kelas minoritas HANYA pada data latih.
  4. Fit TF-IDF hanya pada data latih.
  5. Uji pada data uji alami (distribusi kelas apa adanya).

Skrip juga memuat dua pemeriksaan pembanding:
  - "metode lama": oversample seluruh data lalu split (menghasilkan kebocoran), dari dataset_preprocessed.csv.
  - "uji ketat": membuang baris uji yang teks bersihnya identik dengan teks di data latih.

Jalankan:  python evaluate_no_leakage.py
Keluaran:  results_no_leakage.json dan tabel di terminal.
"""
import json
import os
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils import resample

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
TFIDF = dict(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95)
SCHEMES = [("SVM", 0.2), ("LR", 0.2), ("RF", 0.3)]  # sama dengan skema di notebook asli


def make_model(name):
    if name == "SVM":
        return SVC(kernel="linear", C=1.0, random_state=42, probability=True)
    if name == "LR":
        try:
            return LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs", multi_class="multinomial")
        except TypeError:  # scikit-learn terbaru menghapus parameter multi_class
            return LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
    return RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_split=2, random_state=42, n_jobs=-1)


def oversample(x, y):
    frame = pd.DataFrame({"x": list(x), "y": list(y)})
    n = frame["y"].value_counts().max()
    parts = [resample(frame[frame["y"] == c], replace=True, n_samples=n, random_state=42) for c in sorted(frame["y"].unique())]
    out = pd.concat(parts)
    return out["x"], out["y"].values


def fit_eval(name, x_train, y_train, x_test, y_test, classes):
    vec = TfidfVectorizer(**TFIDF)
    a, b = vec.fit_transform(x_train), vec.transform(x_test)
    model = make_model(name)
    model.fit(a, y_train)
    p_train, p_test = model.predict(a), model.predict(b)
    return {
        "train_acc": round(accuracy_score(y_train, p_train), 4),
        "test_acc": round(accuracy_score(y_test, p_test), 4),
        "macro_f1": round(f1_score(y_test, p_test, average="macro"), 4),
        "recall": {c: round(float(r), 3) for c, r in zip(classes, recall_score(y_test, p_test, average=None))},
    }


def main():
    df = pd.read_csv(os.path.join(HERE, "dataset_unique_clean.csv"))
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    x = df["review_clean"]
    out = {
        "n_reviews": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "majority_baseline_acc": round(float(df["label"].value_counts().max() / len(df)), 4),
        "correct_protocol": {},
        "strict_unseen_text": {},
        "old_method_with_leakage": {},
    }

    for name, ts in SCHEMES:
        tag = f"{name}_{round((1 - ts) * 100)}/{round(ts * 100)}"
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=ts, random_state=42, stratify=y)
        x_os, y_os = oversample(x_tr, y_tr)
        res = fit_eval(name, x_os, y_os, x_te, y_te, le.classes_)
        res.update(n_train_unique=len(x_tr), n_train_after_oversample=len(x_os), n_test=len(x_te))
        out["correct_protocol"][tag] = res
        print("benar ", tag, res, flush=True)

        keep = ~x_te.isin(set(x_tr)).values
        strict = fit_eval(name, x_os, y_os, x_te[keep], y_te[keep], le.classes_)
        out["strict_unseen_text"][tag] = {"n_test_kept": int(keep.sum()), "n_test": len(x_te), "test_acc": strict["test_acc"], "macro_f1": strict["macro_f1"]}
        print("ketat ", tag, out["strict_unseen_text"][tag], flush=True)

    old_path = os.path.join(HERE, "dataset_preprocessed.csv")
    if os.path.exists(old_path):
        old = pd.read_csv(old_path).dropna(subset=["review_clean"])
        old = old[old["review_clean"].str.strip() != ""]
        yo = LabelEncoder().fit_transform(old["label"])
        xo = old["review_clean"]
        for name, ts in SCHEMES:
            tag = f"{name}_{round((1 - ts) * 100)}/{round(ts * 100)}"
            x_tr, x_te, y_tr, y_te = train_test_split(xo, yo, test_size=ts, random_state=42, stratify=yo)
            res = fit_eval(name, x_tr, y_tr, x_te, y_te, le.classes_)
            res["test_rows_with_text_also_in_train"] = round(float(x_te.isin(set(x_tr)).mean()), 4)
            out["old_method_with_leakage"][tag] = res
            print("lama  ", tag, res, flush=True)

    with open(os.path.join(HERE, "results_no_leakage.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("Selesai: results_no_leakage.json")


if __name__ == "__main__":
    main()
