from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
MODELS_DIR = ROOT / "models"

RAW_FILE = RAW_DIR / "Autism-Adult-Data.arff"
TARGET = "class_asd"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def decode_arff_frame(df: pd.DataFrame) -> pd.DataFrame:
    decoded = df.copy()
    for column in decoded.columns:
        if decoded[column].dtype == object:
            decoded[column] = decoded[column].map(
                lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
            )
    return decoded


def load_raw_data() -> pd.DataFrame:
    data, _ = arff.loadarff(RAW_FILE)
    return decode_arff_frame(pd.DataFrame(data))


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    cleaned = raw.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace("/", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    cleaned = cleaned.rename(
        columns={
            "jundice": "jaundice",
            "austim": "family_asd",
            "contry_of_res": "country_of_residence",
            "class_asd": TARGET,
        }
    )

    cleaned = cleaned.replace("?", np.nan)
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].astype(object).where(cleaned[column].notna(), np.nan)
        cleaned[column] = cleaned[column].map(lambda value: value.strip() if isinstance(value, str) else value)

    score_columns = [f"a{i}_score" for i in range(1, 11)]
    for column in score_columns + ["age", "result"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned[(cleaned["age"].isna()) | (cleaned["age"].between(18, 100))]
    cleaned[TARGET] = cleaned[TARGET].map({"YES": 1, "NO": 0}).astype(int)
    cleaned["aq10_score_recomputed"] = cleaned[score_columns].sum(axis=1)
    cleaned["screening_outcome"] = cleaned[TARGET].map({1: "ASD", 0: "No ASD"})

    cleaned = cleaned.drop(columns=["age_desc"], errors="ignore").drop_duplicates()
    cleaned.to_csv(PROCESSED_DIR / "autism_adult_cleaned.csv", index=False)
    return cleaned


def build_visuals(cleaned: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    palette = ["#2563eb", "#dc2626"]

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=cleaned,
        x="screening_outcome",
        hue="screening_outcome",
        palette=palette,
        legend=False,
    )
    ax.set_title("ASD Screening Outcome Distribution")
    ax.set_xlabel("Screening Outcome")
    ax.set_ylabel("Records")
    save_plot(FIGURES_DIR / "target_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=cleaned,
        x="screening_outcome",
        y="age",
        hue="screening_outcome",
        palette=palette,
        legend=False,
    )
    plt.title("Age by Screening Outcome")
    plt.xlabel("Screening Outcome")
    plt.ylabel("Age")
    save_plot(FIGURES_DIR / "age_by_outcome.png")

    aq_rates = (
        cleaned.melt(
            id_vars=[TARGET],
            value_vars=[f"a{i}_score" for i in range(1, 11)],
            var_name="question",
            value_name="score",
        )
        .groupby(["question", TARGET])["score"]
        .mean()
        .mul(100)
        .reset_index(name="positive_response_rate")
    )
    aq_rates["screening_outcome"] = aq_rates[TARGET].map({1: "ASD", 0: "No ASD"})
    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=aq_rates,
        x="question",
        y="positive_response_rate",
        hue="screening_outcome",
        palette=palette,
    )
    plt.title("AQ-10 Positive Response Rate by Screening Outcome")
    plt.xlabel("AQ-10 Question")
    plt.ylabel("Positive Response Rate (%)")
    save_plot(FIGURES_DIR / "aq10_response_rates.png")

    country_summary = (
        cleaned.groupby("country_of_residence")
        .agg(records=(TARGET, "size"), asd_rate=(TARGET, "mean"))
        .query("records >= 8")
        .sort_values("records", ascending=False)
        .head(12)
        .reset_index()
    )
    country_summary["asd_rate"] *= 100
    country_summary.to_csv(TABLES_DIR / "country_screening_summary.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=country_summary.sort_values("asd_rate", ascending=False),
        y="country_of_residence",
        x="asd_rate",
        color="#0f766e",
    )
    plt.title("ASD Screening Rate by Country, Countries with 8+ Records")
    plt.xlabel("ASD Screening Rate (%)")
    plt.ylabel("Country")
    save_plot(FIGURES_DIR / "asd_rate_by_country.png")

    corr_cols = [f"a{i}_score" for i in range(1, 11)] + ["age", "aq10_score_recomputed", TARGET]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cleaned[corr_cols].corr(), cmap="RdBu_r", center=0, annot=False)
    plt.title("Correlation Heatmap")
    save_plot(FIGURES_DIR / "correlation_heatmap.png")


def get_features(cleaned: pd.DataFrame) -> list[str]:
    return [
        "a1_score",
        "a2_score",
        "a3_score",
        "a4_score",
        "a5_score",
        "a6_score",
        "a7_score",
        "a8_score",
        "a9_score",
        "a10_score",
        "age",
        "gender",
        "ethnicity",
        "jaundice",
        "family_asd",
        "country_of_residence",
        "used_app_before",
        "relation",
    ]


def make_preprocessors(X: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer]:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_scaled = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_tree = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
        ]
    )

    scaled_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_scaled, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_tree, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return scaled_preprocessor, tree_preprocessor


def train_models(cleaned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    features = get_features(cleaned)
    X = cleaned[features]
    y = cleaned[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.22,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    scaled_preprocessor, tree_preprocessor = make_preprocessors(X_train)

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", scaled_preprocessor),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=240,
                        max_depth=8,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", tree_preprocessor),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=130,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []
    fitted_models = {}
    for name, model in models.items():
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
                "cv_f1_mean": cv_scores["test_f1"].mean(),
                "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred, zero_division=0),
                "test_recall": recall_score(y_test, y_pred, zero_division=0),
                "test_f1": f1_score(y_test, y_pred, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, y_prob),
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["test_roc_auc", "test_f1"], ascending=False)
    best_model_name = metrics.iloc[0]["model"]
    best_model = fitted_models[best_model_name]

    metrics.to_csv(TABLES_DIR / "model_performance.csv", index=False)
    joblib.dump(best_model, MODELS_DIR / "best_autism_screening_model.joblib")

    best_pred = best_model.predict(X_test)
    best_prob = best_model.predict_proba(X_test)[:, 1]
    predictions = X_test.copy()
    predictions["actual_asd"] = y_test.values
    predictions["predicted_asd"] = best_pred
    predictions["asd_probability"] = best_prob.round(4)
    predictions.to_csv(TABLES_DIR / "sample_predictions.csv", index=False)

    cm = pd.DataFrame(
        confusion_matrix(y_test, best_pred),
        index=["actual_no_asd", "actual_asd"],
        columns=["predicted_no_asd", "predicted_asd"],
    )
    cm.to_csv(TABLES_DIR / "confusion_matrix.csv")

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=metrics.melt(id_vars="model", value_vars=["test_accuracy", "test_f1", "test_roc_auc"]),
        x="model",
        y="value",
        hue="variable",
        palette=["#2563eb", "#0f766e", "#dc2626"],
    )
    plt.title("Model Performance on Test Set")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    save_plot(FIGURES_DIR / "model_performance_comparison.png")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {best_model_name}")
    save_plot(FIGURES_DIR / "confusion_matrix.png")

    importance = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=12,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=1,
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": features,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(TABLES_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    top_features = feature_importance.head(12).sort_values("importance_mean", ascending=True)
    sns.barplot(data=top_features, y="feature", x="importance_mean", color="#7c3aed")
    plt.title(f"Top Features by Permutation Importance - {best_model_name}")
    plt.xlabel("ROC-AUC Importance")
    plt.ylabel("Feature")
    save_plot(FIGURES_DIR / "feature_importance.png")

    metadata = {
        "best_model": best_model_name,
        "features": features,
        "target": TARGET,
        "rows_after_cleaning": int(len(cleaned)),
        "positive_class_rate": float(cleaned[TARGET].mean()),
        "excluded_columns_note": "The aggregate result column was excluded to reduce target leakage risk.",
    }
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics, feature_importance, best_model_name


def write_reports(cleaned: pd.DataFrame, metrics: pd.DataFrame, feature_importance: pd.DataFrame, best_model_name: str) -> None:
    class_counts = cleaned["screening_outcome"].value_counts().to_dict()
    missing_summary = cleaned.isna().sum().sort_values(ascending=False)
    missing_summary = missing_summary[missing_summary > 0].reset_index()
    missing_summary.columns = ["column", "missing_values"]
    missing_summary.to_csv(TABLES_DIR / "missing_values_summary.csv", index=False)

    summary = {
        "records": int(len(cleaned)),
        "features_used": len(get_features(cleaned)),
        "asd_records": int(cleaned[TARGET].sum()),
        "no_asd_records": int((cleaned[TARGET] == 0).sum()),
        "asd_rate": round(float(cleaned[TARGET].mean()), 4),
        "best_model": best_model_name,
        "best_test_roc_auc": round(float(metrics.iloc[0]["test_roc_auc"]), 4),
        "best_test_f1": round(float(metrics.iloc[0]["test_f1"]), 4),
    }
    (REPORTS_DIR / "project_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    best = metrics.iloc[0]
    top_features = feature_importance.head(8)
    feature_rows = "\n".join(
        f"| {row.feature} | {row.importance_mean:.4f} | {row.importance_std:.4f} |"
        for row in top_features.itertuples(index=False)
    )
    feature_table = (
        "| feature | importance_mean | importance_std |\n"
        "|---|---:|---:|\n"
        f"{feature_rows}"
    )
    report = f"""# تقرير تنفيذي: التنبؤ بنتيجة فحص طيف التوحد للبالغين

## هدف المشروع

يهدف هذا المشروع إلى بناء نموذج Machine Learning يتنبأ باحتمالية ظهور نتيجة فحص إيجابية لطيف التوحد لدى البالغين، اعتمادًا على بيانات استبيان AQ-10 وبعض المؤشرات الديموغرافية والطبية المتاحة في مجموعة بيانات UCI.

> تنبيه مهني: هذا المشروع تعليمي وتحليلي لأغراض Portfolio، ولا يمثل أداة تشخيص طبي. التشخيص لا يتم إلا بواسطة مختصين سريريين.

## البيانات

- المصدر: UCI Machine Learning Repository - Autism Screening Adult.
- عدد السجلات بعد التنظيف: {summary["records"]}.
- عدد نتائج ASD: {summary["asd_records"]}.
- عدد نتائج No ASD: {summary["no_asd_records"]}.
- نسبة النتائج الإيجابية: {summary["asd_rate"]:.1%}.
- توزيع الهدف: {class_counts}.

## ما تم تنفيذه

1. تحميل ملف ARFF الخام من UCI.
2. تحويل البيانات إلى DataFrame وتنظيف أسماء الأعمدة والقيم المفقودة.
3. استبعاد العمر غير المنطقي والقيم المكررة.
4. استبعاد عمود `result` من التدريب لأنه مجموع مباشر للأسئلة وقد يسبب تسريبًا للمعلومة.
5. بناء رسوم EDA توضح توزيع الهدف، العمر، إجابات AQ-10، الارتباطات، والفروق حسب الدولة.
6. تدريب ثلاثة نماذج: Logistic Regression وRandom Forest وGradient Boosting.
7. تقييم النماذج باستخدام Accuracy وPrecision وRecall وF1 وROC-AUC مع Cross Validation.
8. حفظ أفضل نموذج ونتائج التنبؤات وأهمية الخصائص.

## أفضل نموذج

- النموذج المختار: {best_model_name}.
- Test Accuracy: {best["test_accuracy"]:.3f}.
- Test Precision: {best["test_precision"]:.3f}.
- Test Recall: {best["test_recall"]:.3f}.
- Test F1: {best["test_f1"]:.3f}.
- Test ROC-AUC: {best["test_roc_auc"]:.3f}.

ملاحظة تقييمية: الأرقام مرتفعة جدًا لأن الهدف مرتبط بقوة بإجابات AQ-10 نفسها. تم استبعاد عمود `result` لتقليل التسريب المباشر، لكن النتائج يجب تفسيرها كدليل على Workflow قابل للتكرار وليس كدليل على أداء تشخيصي سريري.

## أهم الخصائص

{feature_table}

## المخرجات

- Dataset نظيف: `data/processed/autism_adult_cleaned.csv`
- مقاييس النماذج: `reports/tables/model_performance.csv`
- مصفوفة الالتباس: `reports/tables/confusion_matrix.csv`
- أهمية الخصائص: `reports/tables/feature_importance.csv`
- تنبؤات عينة: `reports/tables/sample_predictions.csv`
- النموذج المحفوظ: `models/best_autism_screening_model.joblib`
"""
    (REPORTS_DIR / "executive_report_ar.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw = load_raw_data()
    cleaned = clean_data(raw)
    build_visuals(cleaned)
    metrics, feature_importance, best_model_name = train_models(cleaned)
    write_reports(cleaned, metrics, feature_importance, best_model_name)
    print(f"Completed autism screening ML pipeline. Best model: {best_model_name}")


if __name__ == "__main__":
    main()
