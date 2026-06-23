from __future__ import annotations

import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import FIG_OUT, RANDOM_STATE, TABLE_OUT, ensure_output_dirs, load_grid


FEATURES = [
    "population_pct",
    "housing_price_pct",
    "service_density_pct",
    "daily_life_buffer_1p0km_pct",
    "health_care_buffer_1p0km_pct",
    "transit_buffer_1p0km_pct",
    "elder_care_buffer_1p0km_pct",
    "education_child_buffer_1p0km_pct",
    "recreation_park_sport_buffer_1p0km_pct",
    "lon_center",
    "lat_center",
]

TARGET_LABELS = [
    "low_pet_serviceization",
    "mixed_or_midlevel",
    "retail_only_absorption",
    "medical_only_professionalization",
    "bundled_pet_service_maturity",
]

TARGET_CLASS = "bundled_pet_service_maturity"
SAMPLE_N = 3000

FEATURE_LABELS = {
    "population_pct": "Population",
    "housing_price_pct": "Housing price",
    "service_density_pct": "Service density",
    "daily_life_buffer_1p0km_pct": "Daily life",
    "health_care_buffer_1p0km_pct": "Health care",
    "transit_buffer_1p0km_pct": "Transit",
    "elder_care_buffer_1p0km_pct": "Elder care",
    "education_child_buffer_1p0km_pct": "Education",
    "recreation_park_sport_buffer_1p0km_pct": "Recreation",
    "lon_center": "Longitude",
    "lat_center": "Latitude",
}

FEATURE_GROUPS = {
    "population_pct": "Urban intensity",
    "housing_price_pct": "Urban intensity",
    "service_density_pct": "Urban intensity",
    "transit_buffer_1p0km_pct": "Urban intensity",
    "daily_life_buffer_1p0km_pct": "Daily-life care",
    "health_care_buffer_1p0km_pct": "All-age care",
    "elder_care_buffer_1p0km_pct": "All-age care",
    "education_child_buffer_1p0km_pct": "All-age care",
    "recreation_park_sport_buffer_1p0km_pct": "All-age care",
    "lon_center": "Spatial position",
    "lat_center": "Spatial position",
}

GROUP_COLORS = {
    "Daily-life care": "#2C7FB8",
    "All-age care": "#4FA18E",
    "Urban intensity": "#657381",
    "Spatial position": "#8B73B7",
}


def clean_png_metadata(path) -> None:
    with Image.open(path) as image:
        image.save(path, format="PNG")


def clean_svg_metadata(path) -> None:
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"\n <metadata>.*?</metadata>", "", svg, count=1, flags=re.DOTALL)
    svg = svg.replace("matplotlib.axis_", "axis_")
    path.write_text(svg, encoding="utf-8")


def classify_signatures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    retail_high = out["pet_retail_buffer_1p0km_pct"].ge(0.8)
    medical_high = out["pet_medical_buffer_1p0km_pct"].ge(0.8)
    retail_low = out["pet_retail_buffer_1p0km_pct"].le(0.25)
    medical_low = out["pet_medical_buffer_1p0km_pct"].le(0.25)
    out["signature_label"] = np.select(
        [
            retail_high & medical_low,
            medical_high & retail_low,
            retail_high & medical_high,
            retail_low & medical_low,
        ],
        [
            "retail_only_absorption",
            "medical_only_professionalization",
            "bundled_pet_service_maturity",
            "low_pet_serviceization",
        ],
        default="mixed_or_midlevel",
    )
    out["spatial_block"] = (
        pd.qcut(out["lon_center"].rank(method="first"), 5, labels=False, duplicates="drop").astype(str)
        + "_"
        + pd.qcut(out["lat_center"].rank(method="first"), 5, labels=False, duplicates="drop").astype(str)
    )
    return out


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    work = df[df["population_observed"].astype(bool)].copy()
    x = work[FEATURES].copy()
    x["housing_price_pct"] = x["housing_price_pct"].fillna(-1)
    for col in x.columns:
        x[col] = x[col].fillna(x[col].median())
    y = pd.Categorical(work["signature_label"], categories=TARGET_LABELS)
    groups = work["spatial_block"]
    return x, pd.Series(y, index=work.index, name="signature_label"), groups


def fit_best_model(x: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=260,
        max_depth=8,
        min_samples_leaf=15,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model.fit(x, y)


def evaluate_models(x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RandomForestClassifier]:
    models = {
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=260,
            max_depth=8,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.055,
            max_leaf_nodes=24,
            l2_regularization=0.10,
            random_state=RANDOM_STATE,
        ),
        "shallow_surrogate_tree": Pipeline(
            [
                ("scale", StandardScaler()),
                ("tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=60, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }
    rows = []
    predictions = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    group_cv = GroupKFold(n_splits=5)
    for name, model in models.items():
        for cv_name, splitter in [("stratified_5fold", cv), ("spatial_block_5fold", group_cv)]:
            if cv_name.startswith("spatial"):
                pred = cross_val_predict(model, x, y, cv=splitter, groups=groups, n_jobs=None)
            else:
                pred = cross_val_predict(model, x, y, cv=splitter, n_jobs=None)
            rows.append(
                {
                    "model": name,
                    "cv_scheme": cv_name,
                    "balanced_accuracy": balanced_accuracy_score(y, pred),
                    "macro_f1": f1_score(y, pred, average="macro"),
                    "weighted_f1": f1_score(y, pred, average="weighted"),
                }
            )
            rep = classification_report(y, pred, output_dict=True, zero_division=0)
            for label in TARGET_LABELS:
                predictions.append(
                    {
                        "model": name,
                        "cv_scheme": cv_name,
                        "label": label,
                        "precision": rep[label]["precision"],
                        "recall": rep[label]["recall"],
                        "f1": rep[label]["f1-score"],
                        "support": rep[label]["support"],
                    }
                )
    best_model = fit_best_model(x, y)
    cm_pred = cross_val_predict(best_model, x, y, cv=group_cv, groups=groups)
    cm = pd.DataFrame(
        confusion_matrix(y, cm_pred, labels=TARGET_LABELS),
        index=[f"true_{v}" for v in TARGET_LABELS],
        columns=[f"pred_{v}" for v in TARGET_LABELS],
    )
    return pd.DataFrame(rows), pd.DataFrame(predictions), cm, best_model


def feature_importance(x: pd.DataFrame, y: pd.Series, model: RandomForestClassifier) -> pd.DataFrame:
    perm = permutation_importance(model, x, y, n_repeats=12, random_state=RANDOM_STATE, scoring="balanced_accuracy", n_jobs=-1)
    rows = []
    model_importance = getattr(model, "feature_importances_", np.zeros(len(FEATURES)))
    for i, feature in enumerate(x.columns):
        rows.append(
            {
                "feature": feature,
                "rf_gini_importance": model_importance[i],
                "permutation_importance_mean": perm.importances_mean[i],
                "permutation_importance_std": perm.importances_std[i],
            }
        )
    return pd.DataFrame(rows).sort_values("permutation_importance_mean", ascending=False)


def class_shap_values(model: RandomForestClassifier, sample_x: pd.DataFrame) -> np.ndarray:
    values = shap.TreeExplainer(model).shap_values(sample_x, check_additivity=False)
    class_idx = TARGET_LABELS.index(TARGET_CLASS)
    if isinstance(values, list):
        return np.asarray(values[class_idx])
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[1] == sample_x.shape[1] and arr.shape[2] == len(TARGET_LABELS):
        return arr[:, :, class_idx]
    if arr.ndim == 3 and arr.shape[0] == len(TARGET_LABELS):
        return arr[class_idx, :, :]
    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")


def normalize_feature_values(sample_x: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=sample_x.index)
    for col in sample_x.columns:
        vals = sample_x[col].astype(float)
        lo = np.nanpercentile(vals, 5)
        hi = np.nanpercentile(vals, 95)
        out[col] = 0.5 if np.isclose(hi, lo) else ((vals - lo) / (hi - lo)).clip(0, 1)
    return out


def swarm_offsets(values: np.ndarray, base: float, max_width: float = 0.25, bins: int = 80) -> np.ndarray:
    vals = np.asarray(values)
    if len(vals) == 0:
        return np.array([])
    edges = np.linspace(vals.min(), vals.max(), bins + 1)
    if np.isclose(vals.min(), vals.max()):
        return np.full(len(vals), base)
    bin_id = np.clip(np.digitize(vals, edges) - 1, 0, bins - 1)
    offsets = np.zeros(len(vals))
    rng = np.random.default_rng(RANDOM_STATE + int(base * 1000))
    for b in np.unique(bin_id):
        idx = np.where(bin_id == b)[0]
        order = idx[np.argsort(vals[idx], kind="mergesort")]
        n = len(order)
        local = np.array([0.0]) if n == 1 else np.linspace(-min(max_width, 0.028 * n), min(max_width, 0.028 * n), n)
        offsets[order] = local + rng.normal(0, 0.006, size=n)
    return base + offsets


def render_shap_figure(sample_x: pd.DataFrame, shap_arr: np.ndarray, feature_order: list[str]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.titlesize": 8.8,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 6.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 0,
        }
    )
    cmap = LinearSegmentedColormap.from_list("feature_value", ["#2B83BA", "#F7F7F7", "#D7194A"])
    norm = Normalize(0, 1)
    feature_norm = normalize_feature_values(sample_x)

    fig = plt.figure(figsize=(7.85, 5.85), dpi=320, facecolor="white")
    ax = fig.add_axes([0.245, 0.155, 0.485, 0.675])
    ax_bar = fig.add_axes([0.765, 0.155, 0.135, 0.675], sharey=ax)
    color_ax = fig.add_axes([0.945, 0.205, 0.014, 0.545])

    y_positions = np.arange(len(feature_order))[::-1]
    mean_abs = np.array([np.mean(np.abs(shap_arr[:, FEATURES.index(f)])) for f in feature_order])
    abs_lim = float(np.nanpercentile(np.abs(shap_arr[:, [FEATURES.index(f) for f in feature_order]]), 99.5) * 1.16)
    abs_lim = max(abs_lim, 0.01)

    for y_pos, feature in zip(y_positions, feature_order):
        i = FEATURES.index(feature)
        vals = shap_arr[:, i]
        colors = feature_norm[feature].to_numpy()
        mapped_colors = cmap(norm(colors))
        mapped_colors[:, :3] = mapped_colors[:, :3] * 0.82 + 0.18
        mapped_colors[:, 3] = 1.0
        ax.scatter(
            np.clip(vals, -abs_lim, abs_lim),
            swarm_offsets(vals, float(y_pos) - 0.10),
            c=mapped_colors,
            s=5.8,
            alpha=1.0,
            linewidths=0,
            zorder=4,
        )

    ax.axvline(0, color="#5E6874", lw=0.75, alpha=0.70, zorder=2)
    ax.set_xlim(-abs_lim, abs_lim)
    ax.set_ylim(-0.7, len(feature_order) - 0.3)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([FEATURE_LABELS[f] for f in feature_order])
    ax.set_xlabel("SHAP value for bundled-maturity class")
    ax.set_ylabel("Features")
    ax.grid(axis="x", color="#E6EAF0", lw=0.75)
    ax.grid(axis="y", visible=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#B8C1CC")
    ax.spines["bottom"].set_color("#B8C1CC")

    bar_colors = [GROUP_COLORS[FEATURE_GROUPS[f]] for f in feature_order]
    ax_bar.barh(y_positions, mean_abs, height=0.54, color=bar_colors, edgecolor="white", linewidth=0.45, zorder=3)
    max_bar = float(mean_abs.max() * 1.16)
    for y_pos, value in zip(y_positions, mean_abs):
        ax_bar.text(value + max_bar * 0.025, y_pos, f"{value:.03f}", ha="left", va="center", fontsize=6.2, color="#252B32", fontweight="bold")
    ax_bar.set_xlim(0, max_bar)
    ax_bar.set_xlabel("Mean |SHAP|")
    ax_bar.xaxis.set_label_position("top")
    ax_bar.xaxis.tick_top()
    ax_bar.set_xticks([0, 0.02, 0.04, 0.06])
    ax_bar.set_xticklabels(["0", "0.02", "0.04", "0.06"])
    ax_bar.tick_params(axis="x", colors="#606B78", labelsize=6.6, pad=1)
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.grid(axis="x", color="#E6EAF0", lw=0.75, zorder=0)
    ax_bar.grid(axis="y", visible=False)
    ax_bar.set_facecolor("white")
    for side in ["top", "bottom", "left", "right"]:
        ax_bar.spines[side].set_color("#B8C1CC")
    ax_bar.spines["left"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)

    for j in range(80):
        y0 = j / 80
        color_ax.add_patch(Rectangle((0, y0), 1, 1 / 80, facecolor=cmap(norm(y0 + 0.5 / 80)), edgecolor="none"))
    color_ax.set_xlim(0, 1)
    color_ax.set_ylim(0, 1)
    color_ax.set_xticks([])
    color_ax.set_yticks([0, 1])
    color_ax.set_yticklabels(["Low", "High"])
    color_ax.tick_params(labelsize=6.6, length=0)
    for spine in color_ax.spines.values():
        spine.set_edgecolor("#D7DEE8")
    color_ax.set_title("Feature\nvalue", fontsize=6.9, pad=6)

    fig.text(0.075, 0.945, "(f) Model explanation of bundled pet-service maturity", ha="left", va="center", fontsize=9.4, fontweight="bold", color="#1E2329")
    handles = [Patch(facecolor=GROUP_COLORS[name], edgecolor="none", alpha=0.65, label=name) for name in ["Daily-life care", "All-age care", "Urban intensity", "Spatial position"]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.265, 0.906), ncol=4, frameon=False, handlelength=1.25, columnspacing=1.15, handletextpad=0.35, fontsize=6.7)

    png_path = FIG_OUT / "fig6_model.png"
    svg_path = FIG_OUT / "fig6_model.svg"
    pdf_path = FIG_OUT / "fig6_model.pdf"
    fig.savefig(png_path, dpi=420, facecolor="white")
    fig.savefig(svg_path, dpi=420, facecolor="white")
    plt.close(fig)
    clean_png_metadata(png_path)
    clean_svg_metadata(svg_path)
    with Image.open(png_path).convert("RGB") as image:
        image.save(
            pdf_path,
            "PDF",
            resolution=420.0,
            title="",
            author="",
            subject="",
            keywords="",
            creator="",
            producer="",
            creationDate="",
            modDate="",
        )


def write_shap_tables(sample_x: pd.DataFrame, sample_y: pd.Series, shap_arr: np.ndarray, feature_order: list[str]) -> None:
    summary = []
    sample_rows = []
    for feature in feature_order:
        i = FEATURES.index(feature)
        summary.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "feature_group": FEATURE_GROUPS[feature],
                "target_class": TARGET_CLASS,
                "sample_n": len(sample_x),
                "mean_abs_shap": float(np.mean(np.abs(shap_arr[:, i]))),
                "mean_shap": float(np.mean(shap_arr[:, i])),
                "q05_shap": float(np.quantile(shap_arr[:, i], 0.05)),
                "q50_shap": float(np.quantile(shap_arr[:, i], 0.50)),
                "q95_shap": float(np.quantile(shap_arr[:, i], 0.95)),
            }
        )
        sample_rows.append(
            pd.DataFrame(
                {
                    "grid_row_id": sample_x.index.astype(int),
                    "observed_label": sample_y.astype(str).to_numpy(),
                    "target_class": TARGET_CLASS,
                    "feature": feature,
                    "feature_value": sample_x[feature].to_numpy(float),
                    "shap_value": shap_arr[:, i],
                }
            )
        )
    pd.DataFrame(summary).to_csv(TABLE_OUT / "shap_summary.csv", index=False)
    pd.concat(sample_rows, ignore_index=True).to_csv(TABLE_OUT / "shap_sample.csv", index=False)


def main() -> None:
    ensure_output_dirs()
    df = classify_signatures(load_grid())
    x, y, groups = prepare_model_data(df)
    metrics, label_metrics, cm, model = evaluate_models(x, y, groups)
    importance = feature_importance(x, y, model)
    sample_index = x.sample(n=min(SAMPLE_N, len(x)), random_state=RANDOM_STATE).index
    sample_x = x.loc[sample_index].copy()
    sample_y = y.loc[sample_index].copy()
    shap_arr = class_shap_values(model, sample_x)
    mean_abs = {feature: float(np.mean(np.abs(shap_arr[:, i]))) for i, feature in enumerate(FEATURES)}
    feature_order = sorted(FEATURES, key=lambda f: mean_abs[f], reverse=True)

    metrics.to_csv(TABLE_OUT / "model.csv", index=False)
    label_metrics.to_csv(TABLE_OUT / "labels.csv", index=False)
    cm.to_csv(TABLE_OUT / "confusion.csv")
    importance.to_csv(TABLE_OUT / "importance.csv", index=False)
    write_shap_tables(sample_x, sample_y, shap_arr, feature_order)
    render_shap_figure(sample_x, shap_arr, feature_order)
    print("Signature model and explanation figure reproduced.")


if __name__ == "__main__":
    main()
