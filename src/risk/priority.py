import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "outputs" / "scene_inference"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "risk"


DAMAGE_SEVERITY = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


PRIORITY_LEVELS = {
    3: "critical",
    2: "high",
    1: "medium",
    0: "low",
}


def calculate_priority(damage_class, confidence):
    """
    Calculate a response-priority score from predicted damage
    and model confidence.

    This is a model-based triage score, not a real-world
    emergency decision.
    """
    severity = DAMAGE_SEVERITY.get(damage_class)

    if severity is None:
        raise ValueError(f"Unknown damage class: {damage_class}")

    confidence = max(0.0, min(1.0, float(confidence)))

    if severity == 0:
        score = 0.0
    else:
        score = (severity / 3.0) * confidence * 100.0

    if score >= 75:
        priority = "critical"
    elif score >= 45:
        priority = "high"
    elif score > 0:
        priority = "medium"
    else:
        priority = "low"

    return round(score, 2), priority


def load_predictions(path):
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


def process_scene(scene_id):
    INPUT_FILE = INPUT_DIR / f"{scene_id}.json"

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Inference result not found: {INPUT_FILE}"
        )

    predictions = load_predictions(INPUT_FILE)

    results = []

    for item in predictions:
        damage_class = item.get("prediction")
        confidence = item.get("confidence")

        if damage_class is None or confidence is None:
            continue

        score, priority = calculate_priority(
            damage_class,
            confidence,
        )

        result = dict(item)
        result["risk_score"] = score
        result["response_priority"] = priority

        results.append(result)

    results.sort(
        key=lambda x: x["risk_score"],
        reverse=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{scene_id}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "scene_id": scene_id,
                "buildings": len(results),
                "results": results,
            },
            f,
            indent=2,
        )

    counts = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    for item in results:
        counts[item["response_priority"]] += 1

    print("Aegis AI - Response Priority")
    print("=" * 50)
    print(f"Scene:     {scene_id}")
    print(f"Buildings: {len(results)}")
    print()
    print("Priority Summary")
    print("-" * 50)

    for priority in ["critical", "high", "medium", "low"]:
        print(f"{priority:<10}: {counts[priority]}")

    print()
    print("Top Priority Buildings")
    print("-" * 50)

    for i, item in enumerate(results[:10], 1):
        print(
            f"{i:2}. "
            f"{item.get('uid', '')[:12]}  "
            f"{item.get('prediction', ''):<15} "
            f"confidence={item.get('confidence', 0):.4f}  "
            f"score={item['risk_score']:>6.2f}  "
            f"priority={item['response_priority']}"
        )

    print()
    print(f"Results saved: {output_file}")


if __name__ == "__main__":
    process_scene("palu-tsunami_00000087")
