from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .data import build_customer_features, clean_transactions, load_raw

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RANDOM_STATE = 42
CORE_FEATURES = ["recency_days", "frequency_orders", "monetary_value"]
K_CANDIDATES = range(2, 9)
MIN_CLUSTER_SHARE = 0.05
PERSONA_LADDER = [
    "Champions",
    "Loyal High Value",
    "Promising",
    "Steady",
    "Occasional",
    "At Risk",
    "Dormant",
    "Low Engagement",
]


def make_preprocessor() -> Pipeline:
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])


def candidate_table(X_scaled: np.ndarray) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    rows = []
    models: dict[int, KMeans] = {}
    for k in K_CANDIDATES:
        model = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        counts = np.bincount(labels, minlength=k)
        min_share = float(counts.min() / len(labels))
        rows.append({
            "k": k,
            "silhouette": float(silhouette_score(X_scaled, labels)),
            "min_cluster_share": min_share,
            "eligible": bool(min_share >= MIN_CLUSTER_SHARE),
        })
        models[k] = model
    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]]
    if eligible.empty:
        raise RuntimeError("No clustering candidate passed the minimum cluster-share guard")
    return table, models


def select_k(table: pd.DataFrame) -> int:
    eligible = table.loc[table["eligible"]].copy()
    winner = eligible.sort_values(["silhouette", "min_cluster_share"], ascending=False).iloc[0]
    return int(winner["k"])


def stability_audit(X_scaled: np.ndarray, reference_labels: np.ndarray, k: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    sample_size = int(round(len(X_scaled) * 0.80))
    for run in range(10):
        idx = np.sort(rng.choice(len(X_scaled), size=sample_size, replace=False))
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE + run + 1)
        sample_labels = model.fit_predict(X_scaled[idx])
        ari = adjusted_rand_score(reference_labels[idx], sample_labels)
        rows.append({"run": run + 1, "sample_size": sample_size, "adjusted_rand_index": float(ari)})
    return pd.DataFrame(rows)


def assign_personas(profile: pd.DataFrame) -> pd.DataFrame:
    """Attach descriptive business labels after clustering without affecting the model.

    Clusters are ordered from stronger to weaker RFM behavior using percentile ranks.
    Persona names are then sampled across the full high-to-low ladder, so even a
    two-cluster solution receives one high-value and one low-engagement label rather
    than two positive-sounding names. These labels are descriptive, not ground truth.
    """
    out = profile.copy()
    recency_rank = out["recency_days_median"].rank(pct=True, ascending=False)
    freq_rank = out["frequency_orders_median"].rank(pct=True, ascending=True)
    money_rank = out["monetary_value_median"].rank(pct=True, ascending=True)
    out["value_score"] = recency_rank + freq_rank + money_rank

    ordered = out.sort_values("value_score", ascending=False).index.tolist()
    ladder_positions = np.rint(
        np.linspace(0, len(PERSONA_LADDER) - 1, num=len(ordered))
    ).astype(int)
    persona_map = {
        idx: PERSONA_LADDER[position]
        for idx, position in zip(ordered, ladder_positions, strict=True)
    }
    out["persona"] = out.index.map(persona_map)
    return out.drop(columns=["value_score"])


def profile_clusters(customers: pd.DataFrame) -> pd.DataFrame:
    profile = customers.groupby("cluster").agg(
        customers=("CustomerID", "count"),
        recency_days_median=("recency_days", "median"),
        frequency_orders_median=("frequency_orders", "median"),
        monetary_value_median=("monetary_value", "median"),
        monetary_value_mean=("monetary_value", "mean"),
        avg_order_value_median=("avg_order_value", "median"),
        items_purchased_median=("items_purchased", "median"),
    )
    profile["customer_share"] = profile["customers"] / profile["customers"].sum()
    return assign_personas(profile).reset_index()


def main() -> None:
    ART.mkdir(exist_ok=True)
    raw = load_raw()
    clean, audit = clean_transactions(raw)
    customers, snapshot_date = build_customer_features(clean)

    prep = make_preprocessor()
    X = customers[CORE_FEATURES].astype(float)
    X_scaled = prep.fit_transform(X)

    selection, models = candidate_table(X_scaled)
    selected_k = select_k(selection)
    selected_model = models[selected_k]
    labels = selected_model.labels_

    customers = customers.copy()
    customers["cluster"] = labels
    profile = profile_clusters(customers)
    persona_map = dict(zip(profile["cluster"], profile["persona"]))
    customers["persona"] = customers["cluster"].map(persona_map)

    stability = stability_audit(X_scaled, labels, selected_k)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        "CustomerID": customers["CustomerID"],
        "pc1": coords[:, 0],
        "pc2": coords[:, 1],
        "cluster": labels,
        "persona": customers["persona"],
    })

    selection.to_csv(ART / "cluster_selection.csv", index=False)
    stability.to_csv(ART / "stability_runs.csv", index=False)
    customers.to_csv(ART / "customer_segments.csv", index=False)
    profile.to_csv(ART / "cluster_profiles.csv", index=False)
    pca_df.to_csv(ART / "pca_coordinates.csv", index=False)

    bundle = {"preprocessor": prep, "kmeans": selected_model, "core_features": CORE_FEATURES}
    joblib.dump(bundle, ART / "model.joblib")

    selected_row = selection.loc[selection["k"] == selected_k].iloc[0]
    report = {
        "data_audit": audit,
        "snapshot_date": snapshot_date.isoformat(),
        "customer_count": int(len(customers)),
        "core_features": CORE_FEATURES,
        "selection_policy": {
            "k_candidates": list(K_CANDIDATES),
            "primary_metric": "silhouette_score",
            "minimum_cluster_share": MIN_CLUSTER_SHARE,
            "selected_k": selected_k,
            "selected_silhouette": float(selected_row["silhouette"]),
            "selected_min_cluster_share": float(selected_row["min_cluster_share"]),
        },
        "stability": {
            "method": "10 repeated 80% subsamples; ARI against full-data labels on matching customers",
            "mean_ari": float(stability["adjusted_rand_index"].mean()),
            "min_ari": float(stability["adjusted_rand_index"].min()),
        },
        "pca": {
            "role": "diagnostic visualization only; clustering uses scaled RFM space",
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        },
        "persona_policy": "post-hoc descriptive labels ordered by RFM profile and spread across a high-to-low engagement ladder; not ground-truth classes",
        "claim_boundary": "descriptive historical behavioral segmentation; clusters are not causal or universal customer types",
    }
    (ART / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
