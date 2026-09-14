import csv
import json
from pathlib import Path

import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "val.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "scene_inference"
)

RISK_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "risk"
)

DAMAGE_MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "damage_maps"
)

PRIORITY_MAP_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "priority_maps"
)


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


PRIORITY_ORDER = [
    "critical",
    "high",
    "medium",
    "low",
]


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


@st.cache_data
def load_scene_risk(scene_id):
    path = RISK_DIR / f"{scene_id}.json"

    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    return (
        data.get("results")
        or data.get("predictions")
        or data.get("records")
        or []
    )


def get_scene_ids(rows):
    return sorted(
        {row["scene_id"] for row in rows}
    )


def get_scene_info(rows, scene_id):
    scene_rows = [
        row
        for row in rows
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

    # ---------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------

    with st.sidebar:
        st.header("Scene")

        scene_id = st.selectbox(
            "Select a disaster scene",
            scenes,
        )

        st.divider()

        st.caption("Aegis AI")
        st.caption(
            "Building-level damage assessment"
        )

    # ---------------------------------------------------------------
    # Load scene data
    # ---------------------------------------------------------------

    predictions = load_scene_predictions(
        scene_id
    )

    risk_results = load_scene_risk(
        scene_id
    )

    scene_info = get_scene_info(
        rows,
        scene_id
    )

    # ---------------------------------------------------------------
    # Inference check
    # ---------------------------------------------------------------

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

    # ---------------------------------------------------------------
    # Calculate inference statistics
    # ---------------------------------------------------------------

    predicted_counts = {
        class_name: 0
        for class_name in CLASS_ORDER
    }

    correct = 0
    confidences = []

    for prediction in predictions:

        predicted_class = prediction.get(
            "prediction"
        )

        if predicted_class in predicted_counts:
            predicted_counts[
                predicted_class
            ] += 1

        if (
            prediction.get("true_label")
            == prediction.get("prediction")
        ):
            correct += 1

        confidence = prediction.get(
            "confidence"
        )

        if isinstance(
            confidence,
            (int, float),
        ):
            confidences.append(
                float(confidence)
            )

    accuracy = (
        correct / prediction_count
        if prediction_count
        else 0
    )

    avg_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else 0
    )

    # ---------------------------------------------------------------
    # Scene title
    # ---------------------------------------------------------------

    st.subheader(scene_id)

    # ---------------------------------------------------------------
    # KPI row
    # ---------------------------------------------------------------

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
            (
                f"{prediction_count - predicted_counts['no-damage']:,}"
            ),
        )

    with c4:
        st.metric(
            "Average confidence",
            f"{avg_confidence * 100:.1f}%",
        )

    # ---------------------------------------------------------------
    # Damage distribution
    # ---------------------------------------------------------------

    st.divider()

    st.subheader("Damage assessment")

    cols = st.columns(4)

    for col, class_name in zip(
        cols,
        CLASS_ORDER,
    ):
        with col:

            count = predicted_counts[
                class_name
            ]

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

    # ---------------------------------------------------------------
    # Response priority / Risk Engine
    # ---------------------------------------------------------------

    st.divider()

    st.subheader("🚨 Response priority")

    if risk_results is None:

        st.info(
            "Response priority has not been calculated "
            "for this scene yet."
        )

        st.code(
            f"python -m src.risk.priority "
            f"--scene {scene_id}",
            language="bash",
        )

    else:

        priority_counts = {
            priority: 0
            for priority in PRIORITY_ORDER
        }

        for result in risk_results:

            priority = result.get(
                "response_priority"
            )

            if priority in priority_counts:
                priority_counts[
                    priority
                ] += 1

        p1, p2, p3, p4 = st.columns(4)

        with p1:
            st.metric(
                "🔴 Critical",
                f"{priority_counts['critical']:,}",
            )

        with p2:
            st.metric(
                "🟠 High",
                f"{priority_counts['high']:,}",
            )

        with p3:
            st.metric(
                "🟡 Medium",
                f"{priority_counts['medium']:,}",
            )

        with p4:
            st.metric(
                "🟢 Low",
                f"{priority_counts['low']:,}",
            )

        st.caption(
            "Priority is a model-based triage score "
            "combining predicted damage severity and "
            "model confidence. It is not a calibrated "
            "real-world emergency risk score."
        )

        # -----------------------------------------------------------
        # Top priority buildings
        # -----------------------------------------------------------

        st.subheader(
            "🔥 Top priority buildings"
        )

        top_results = sorted(
            risk_results,
            key=lambda x: x.get(
                "risk_score",
                0,
            ),
            reverse=True,
        )[:10]

        priority_table = []

        for result in top_results:

            priority_table.append(
                {
                    "Building ID": result.get(
                        "uid",
                        "",
                    )[:12],

                    "Predicted damage": (
                        CLASS_LABELS.get(
                            result.get(
                                "prediction"
                            ),
                            result.get(
                                "prediction",
                                "",
                            ),
                        )
                    ),

                    "Confidence": (
                        f"{result.get('confidence', 0) * 100:.1f}%"
                    ),

                    "Risk score": (
                        f"{result.get('risk_score', 0):.2f}"
                    ),

                    "Priority": (
                        result.get(
                            "response_priority",
                            "",
                        ).title()
                    ),
                }
            )

        st.dataframe(
            priority_table,
            width="stretch",
            hide_index=True,
        )

    # ---------------------------------------------------------------
    # Building damage map
    # ---------------------------------------------------------------

    st.divider()

    st.subheader(
        "🗺️ Building damage map"
    )

    damage_map_path = (
        DAMAGE_MAP_DIR
        / f"{scene_id}_damage_map.png"
    )

    if damage_map_path.exists():

        st.image(
            Image.open(damage_map_path),
            width="stretch",
        )

    else:

        st.warning(
            "Damage map not generated for this scene yet."
        )

    # ---------------------------------------------------------------
    # Response priority map
    # ---------------------------------------------------------------

    st.divider()

    st.subheader(
        "🚨 Response priority map"
    )

    priority_map_path = (
        PRIORITY_MAP_DIR
        / f"{scene_id}_priority_map.png"
    )

    if risk_results is None:

        st.info(
            "Generate response priority for this scene "
            "before creating the priority map."
        )

    elif priority_map_path.exists():

        st.image(
            Image.open(priority_map_path),
            width="stretch",
        )

        st.caption(
            "Map colors represent model-based response "
            "priority: Critical, High, Medium, and Low."
        )

    else:

        st.warning(
            "Response priority exists, but the priority "
            "map has not been generated for this scene yet."
        )

        st.code(
            f"python -m src.visualization.priority_map",
            language="bash",
        )

    # ---------------------------------------------------------------
    # Building-level predictions
    # ---------------------------------------------------------------

    st.divider()

    st.subheader(
        "Building-level predictions"
    )

    table_rows = []

    # Create a lookup table for risk information.
    risk_by_uid = {}

    if risk_results is not None:

        for result in risk_results:

            uid = result.get("uid")

            if uid:
                risk_by_uid[uid] = result

    for prediction in predictions:

        uid = prediction.get(
            "uid",
            "",
        )

        risk_result = risk_by_uid.get(
            uid,
            {},
        )

        table_rows.append(
            {
                "Building ID": uid[:12],

                "True damage": (
                    CLASS_LABELS.get(
                        prediction.get(
                            "true_label"
                        ),
                        prediction.get(
                            "true_label",
                            "",
                        ),
                    )
                ),

                "Predicted damage": (
                    CLASS_LABELS.get(
                        prediction.get(
                            "prediction"
                        ),
                        prediction.get(
                            "prediction",
                            "",
                        ),
                    )
                ),

                "Confidence": (
                    f"{prediction.get('confidence', 0) * 100:.1f}%"
                ),

                "Risk score": (
                    f"{risk_result.get('risk_score', 0):.2f}"
                    if risk_result
                    else "—"
                ),

                "Priority": (
                    risk_result.get(
                        "response_priority",
                        "",
                    ).title()
                    if risk_result
                    else "—"
                ),
            }
        )

    st.dataframe(
        table_rows,
        width="stretch",
        hide_index=True,
    )

    # ---------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------

    st.divider()

    st.caption(
        "Aegis AI — disaster damage assessment and "
        "response prioritization prototype"
    )


if __name__ == "__main__":
    main()