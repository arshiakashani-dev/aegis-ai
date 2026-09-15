# Aegis AI — Error Analysis

## Overview

The Aegis AI damage classification model was evaluated on the scene-level validation split of the xView2 building-damage dataset.

The final ResNet18 change-aware model uses:

- Pre-disaster RGB imagery
- Post-disaster RGB imagery
- Absolute pixel difference between pre- and post-disaster imagery

The model predicts four damage classes:

1. no-damage
2. minor-damage
3. major-damage
4. destroyed

## Validation Performance

| Metric | Score |
|---|---:|
| Accuracy | 80.04% |
| Macro F1 | 66.84% |
| Weighted F1 | 81.38% |

### Per-class performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| no-damage | 0.9553 | 0.8483 | 0.8986 |
| minor-damage | 0.4232 | 0.5776 | 0.4885 |
| major-damage | 0.5013 | 0.6344 | 0.5601 |
| destroyed | 0.6407 | 0.8389 | 0.7265 |

The strongest performance is obtained for `no-damage` and `destroyed`, while `minor-damage` is the most difficult class.

## Main Error Patterns

The confusion matrix shows that the most important errors occur between neighboring damage-severity classes.

### Minor damage vs. major damage

The model frequently confuses:

- `minor-damage → major-damage`
- `major-damage → minor-damage`

This is expected to be one of the more difficult distinctions because the visual boundary between these severity levels can be subtle at the building-crop resolution used by the model.

### Destroyed vs. major damage

Several high-confidence examples show:

- `destroyed → major-damage`
- `major-damage → destroyed`

These errors occur in scenes where the post-disaster imagery contains substantial structural or appearance changes, but the available visual evidence does not always clearly distinguish complete destruction from severe damage.

### Minor damage vs. no damage

Examples of:

- `minor-damage → no-damage`

show that relatively subtle damage can be difficult to distinguish from normal appearance changes between the pre- and post-disaster images.

### No damage vs. destroyed

A smaller but important class of errors is:

- `no-damage → destroyed`

These are particularly important because they represent high-severity false positives. They can be caused by strong appearance changes, image artifacts, vegetation, shadows, or other changes that resemble severe damage.

## Visual Error Analysis

Building-aware visualizations were generated for representative high-confidence errors.

Each visualization contains:

1. Pre-disaster context
2. Post-disaster context
3. Absolute difference image
4. Building-level crop before the disaster
5. Building-level crop after the disaster
6. Building-level difference image

The labeled building polygon is overlaid in red.

### Observed examples

Representative examples included:

- `major-damage → destroyed` in Santa Rosa wildfire imagery
- `minor-damage → major-damage` in Hurricane Harvey imagery
- `minor-damage → no-damage` in Mexico earthquake imagery
- `no-damage → destroyed` in Palu tsunami imagery
- `destroyed → major-damage` in Hurricane Harvey and Midwest flooding imagery

These examples demonstrate that model confidence alone does not guarantee correctness.

Several incorrect predictions have confidence above 0.98, indicating that the model can be confidently wrong when visual evidence is ambiguous or affected by strong appearance changes.

## Main Failure Modes

### 1. Appearance shift

The pre- and post-disaster images can differ substantially in color, brightness, texture, atmospheric conditions, and image quality even when the underlying building change is limited.

### 2. Low-resolution building evidence

The model operates on resized 128×128 building crops. Fine structural details that could distinguish minor, major, and destroyed damage may therefore be lost.

### 3. Vegetation and surrounding context

Vegetation can partially obscure buildings and introduce strong visual differences between the two images. These changes can be interpreted as building damage.

### 4. Blur and low-information imagery

Some buildings are difficult to identify clearly in either the pre- or post-disaster image. In these cases, the model must make a prediction from weak visual evidence.

### 5. Ambiguous damage severity

The distinction between adjacent severity levels is not always visually sharp. In particular, `minor-damage` and `major-damage` can have overlapping visual characteristics.

## Important Interpretation

The validation accuracy of 80.04% should not be interpreted as evidence that the system can independently determine real-world emergency risk.

The current model is a building-level damage classification model.

The response-priority component is a downstream triage mechanism based on model predictions and confidence. It is not a calibrated measure of actual human safety risk, structural stability, or emergency priority.

Therefore, predictions should be treated as decision-support signals rather than authoritative assessments.

## Limitations

The current system has several limitations:

- Damage classification is based only on the available pre/post imagery.
- The model does not use additional geographic, structural, population, infrastructure, or socioeconomic information.
- Confidence scores are not calibrated emergency-risk probabilities.
- The validation split is scene-level, but the dataset contains strong class imbalance.
- Visual quality varies substantially across disaster types and scenes.
- The model can produce high-confidence incorrect predictions.

## Next Improvements

Potential improvements should focus on improving robustness and reliability rather than simply increasing overall accuracy.

Possible directions include:

1. Better handling of difficult `minor-damage` vs. `major-damage` cases.
2. Stronger augmentation for appearance and acquisition differences.
3. Improved multi-scale building representations.
4. Confidence calibration and uncertainty estimation.
5. More detailed error analysis across individual disaster types.
6. Evaluation using metrics that better reflect minority-class performance.
7. Additional geographic and contextual information for downstream response prioritization.

## Conclusion

The current Aegis AI model demonstrates that paired pre- and post-disaster satellite imagery can be used to classify building damage across four severity levels.

The main remaining challenge is not simply recognizing whether an image changed, but reliably distinguishing the severity of damage under substantial variation in imagery quality, appearance, vegetation, and disaster context.

The error analysis provides a foundation for improving the model while keeping the system's limitations explicit.
