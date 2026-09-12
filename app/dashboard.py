import csv
import json
from pathlib import Path

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scene_inference"
MAP_DIR = PROJECT_ROOT / "outputs" / "damage_maps"


st.set_page_config(
    page_title="Aegis AI",
    page_icon="🛰️",
    layout="wide",
)


CLASS_ORDER = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


CLASS_LABELS = {
    "no-damage": "No Damage",
    "minor-damage": "Minor Damage",
    "major-damage": "Major Damage",
    "destroyed": "Destroyed",
}


@st.cache_data
def load_validation_rows():
    with open(VAL_FILE, newline="") as f:
        return list(csv.DictReader(f))


@st.cache_data
def load_scene_predictions(scene_id):
    path = OUTPUT_DIR / f"{scene_id}.json"

    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return (
        data.get("predictions")
        or data.get("results")
        or data.get("records")
        or []
    )


def get_scene_ids(rows):
    return sorted({row["scene_id"] for row in rows})


def get_scene_info(rows, scene_id):
    scene_rows = [
        row for row in rows
        if row["scene_id"] == scene_id
    ]

    if not scene_rows:
        return None

    return scene_rows[0]


def main():
    rows = load_validation_rows()
    scenes = get_scene_ids(rows)

    st.title("🛰️ Aegis AI")
    st.caption(
        "AI-powered building damage assessment from "
        "before/after satellite imagery"
    )

    st.divider()

    with st.sidebar:
        st.header("Scene")

        scene_id = st.selectbox(
            "Select a disaster scene",
            scenes,
        )

        st.divider()

        st.caption("Aegis AI")
        st.caption("Building-level damage assessment")

    predictions = load_scene_predictions(scene_id)
    scene_info = get_scene_info(rows, scene_id)

    if predictions is None:
        st.warning(
            "This scene has not been processed yet. "
            "Run scene inference first."
        )

        st.code(
            f"python -m src.inference.scene_predict "
            f"--split val --scene {scene_id}",
            language="bash",
        )

        return

    prediction_count = len(predictions)

    predicted_counts = {
        class_name: 0
        for class_name in CLASS_ORDER
    }

    correct = 0
    confidences = []

    for prediction in predictions:
        predicted_class = prediction.get("prediction")

        if predicted_class in predicted_counts:
            predicted_counts[predicted_class] += 1

        if (
            prediction.get("true_label")
            == prediction.get("prediction")
        ):
            correct += 1

        confidence = prediction.get("confidence")

        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    accuracy = correct / prediction_count if prediction_count else 0
    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    st.subheader(scene_id)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Buildings analyzed",
            f"{prediction_count:,}",
        )

    with c2:
        st.metric(
            "Scene accuracy",
            f"{accuracy * 100:.2f}%",
        )

    with c3:
        st.metric(
            "Predicted damaged",
            f"{prediction_count - predicted_counts['no-damage']:,}",
        )

    with c4:
        st.metric(
            "Average confidence",
            f"{avg_confidence * 100:.1f}%",
        )

    st.divider()

    # Damage distribution
    st.subheader("Damage assessment")

    cols = st.columns(4)

    for col, class_name in zip(cols, CLASS_ORDER):
        with col:
            count = predicted_counts[class_name]
            percentage = (
                count / prediction_count * 100
                if prediction_count
                else 0
            )

            st.metric(
                CLASS_LABELS[class_name],
                f"{count:,}",
                f"{percentage:.2f}%",
            )

    st.divider()

    # Map
    map_path = MAP_DIR / f"{scene_id}_damage_map.png"

    st.subheader("Building damage map")

    if map_path.exists():
        st.image(
            Image.open(map_path),
            width="stretch",
        )
    else:
        st.warning("Damage map not generated for this scene yet.")

    st.divider()

    # Building predictions
    st.subheader("Building-level predictions")

    table_rows = []

    for prediction in predictions:
        table_rows.append(
            {
                "Building ID": prediction.get("uid", "")[:12],
                "True damage": CLASS_LABELS.get(
                    prediction.get("true_label"),
                    prediction.get("true_label", ""),
                ),
                "Predicted damage": CLASS_LABELS.get(
                    prediction.get("prediction"),
                    prediction.get("prediction", ""),
                ),
                "Confidence": (
                    f"{prediction.get('confidence', 0) * 100:.1f}%"
                ),
            }
        )

    st.dataframe(
        table_rows,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.caption(
        "Aegis AI — disaster damage assessment prototype"
    )


if __name__ == "__main__":
    main()
