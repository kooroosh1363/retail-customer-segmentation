from pathlib import Path
import json
import pandas as pd

from src.segment import main


def test_segmentation_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / "artifacts" / "metrics.json").read_text())

    assert 2 <= metrics["selection_policy"]["selected_k"] <= 8
    assert metrics["selection_policy"]["selected_silhouette"] > 0.20
    assert metrics["selection_policy"]["selected_min_cluster_share"] >= 0.05
    assert metrics["stability"]["mean_ari"] > 0.60
    assert metrics["customer_count"] > 4_000

    for name in [
        "customer_segments.csv",
        "cluster_profiles.csv",
        "cluster_selection.csv",
        "stability_runs.csv",
        "pca_coordinates.csv",
        "model.joblib",
    ]:
        assert (root / "artifacts" / name).exists()

    customers = pd.read_csv(root / "artifacts" / "customer_segments.csv")
    profiles = pd.read_csv(root / "artifacts" / "cluster_profiles.csv")
    assert customers["CustomerID"].nunique() == len(customers)
    assert customers["cluster"].nunique() == metrics["selection_policy"]["selected_k"]
    assert customers["persona"].notna().all()
    assert profiles["customer_share"].sum() > 0.999
    assert profiles["customer_share"].sum() < 1.001
