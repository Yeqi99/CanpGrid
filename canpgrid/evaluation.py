from __future__ import annotations

from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from typing import Any


def evaluate_interactions(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.35,
    label_threshold: float = 0.55,
) -> dict[str, Any]:
    """Compare predicted interactive elements against annotated clickable regions."""

    targets = [
        _coerce_item(item, f"ground_truth[{index}]")
        for index, item in enumerate(ground_truth)
    ]
    preds = [
        _coerce_item(item, f"predictions[{index}]", allow_missing_bbox=True)
        for index, item in enumerate(predictions)
    ]
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    target_rows: list[dict[str, Any]] = []

    for target_index, target in enumerate(targets):
        best_index = None
        best_score = -1.0
        best_reason = ""
        for pred_index, pred in enumerate(preds):
            if pred_index in used_predictions:
                continue
            score, reason = _geometry_score(target, pred, iou_threshold)
            if score > best_score:
                best_index = pred_index
                best_score = score
                best_reason = reason
        if best_index is None or best_score <= 0:
            continue
        pred = preds[best_index]
        used_predictions.add(best_index)
        used_targets.add(target_index)
        label_score = _label_score(target, pred)
        category = "correct"
        problem_type = "position_ok"
        if _has_label(target) and _has_label(pred) and label_score < label_threshold:
            category = "semantic_mismatch"
            problem_type = "understanding_semantic"
        target_rows.append(
            _target_row(
                category,
                problem_type,
                target,
                pred,
                geometry_reason=best_reason,
            )
        )

    for target_index, target in enumerate(targets):
        if target_index in used_targets:
            continue
        best_index = None
        best_label = -1.0
        for pred_index, pred in enumerate(preds):
            if pred_index in used_predictions:
                continue
            score = _label_score(target, pred)
            if score > best_label:
                best_index = pred_index
                best_label = score
        if best_index is None or best_label < label_threshold:
            continue
        pred = preds[best_index]
        used_predictions.add(best_index)
        used_targets.add(target_index)
        target_rows.append(
            _target_row(
                "localization_error",
                "position_problem",
                target,
                pred,
                geometry_reason="label_match_but_position_miss",
            )
        )

    for target_index, target in enumerate(targets):
        if target_index in used_targets:
            continue
        target_rows.append(
            {
                "category": "missed_interactive",
                "problem_type": "understanding_recall",
                "target": _public_item(target),
                "prediction": None,
                "label_score": None,
                "geometry": "not_detected",
            }
        )

    prediction_rows: list[dict[str, Any]] = []
    for pred_index, pred in enumerate(preds):
        if pred_index in used_predictions:
            continue
        duplicate_target = _duplicate_target(targets, pred, iou_threshold)
        if duplicate_target is not None:
            prediction_rows.append(
                {
                    "category": "duplicate_prediction",
                    "problem_type": "understanding_precision",
                    "target": _public_item(duplicate_target),
                    "prediction": _public_item(pred),
                    "label_score": _label_score(duplicate_target, pred),
                    "geometry": "duplicate_of_detected_target",
                }
            )
            continue
        prediction_rows.append(
            {
                "category": "false_positive",
                "problem_type": "understanding_precision",
                "target": None,
                "prediction": _public_item(pred),
                "label_score": None,
                "geometry": "not_in_annotation",
            }
        )

    rows = sorted(target_rows, key=_row_sort_key) + prediction_rows
    return {
        "summary": _summary(rows, len(targets), len(preds)),
        "rows": rows,
    }


def _summary(
    rows: list[dict[str, Any]], target_count: int, prediction_count: int
) -> dict[str, Any]:
    correct = _count(rows, "correct")
    semantic_mismatch = _count(rows, "semantic_mismatch")
    localization_errors = _count(rows, "localization_error")
    missed = _count(rows, "missed_interactive")
    false_positive = _count(rows, "false_positive")
    duplicate = _count(rows, "duplicate_prediction")
    recognized = correct + localization_errors
    precision_denominator = (
        correct + semantic_mismatch + localization_errors + false_positive + duplicate
    )
    recall = correct / target_count if target_count else 1.0
    recognition_recall = recognized / target_count if target_count else 1.0
    precision = correct / precision_denominator if precision_denominator else 1.0
    localization_accuracy = correct / recognized if recognized else 0.0
    overall = 100 * (0.45 * recall + 0.25 * precision + 0.30 * localization_accuracy)
    return {
        "target_count": target_count,
        "prediction_count": prediction_count,
        "correct_count": correct,
        "semantic_mismatch_count": semantic_mismatch,
        "localization_error_count": localization_errors,
        "missed_interactive_count": missed,
        "false_positive_count": false_positive,
        "duplicate_prediction_count": duplicate,
        "recall": round(recall, 3),
        "recognition_recall": round(recognition_recall, 3),
        "precision": round(precision, 3),
        "localization_accuracy": round(localization_accuracy, 3),
        "score_0_to_100": round(overall, 1),
    }


def _count(rows: list[dict[str, Any]], category: str) -> int:
    return sum(1 for row in rows if row["category"] == category)


def _row_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    target = row.get("target") or {}
    index = target.get("index")
    return int(index if index is not None else 10_000), row["category"]


def _target_row(
    category: str,
    problem_type: str,
    target: dict[str, Any],
    pred: dict[str, Any],
    *,
    geometry_reason: str,
) -> dict[str, Any]:
    return {
        "category": category,
        "problem_type": problem_type,
        "target": _public_item(target),
        "prediction": _public_item(pred),
        "label_score": round(_label_score(target, pred), 3),
        "geometry": geometry_reason,
    }


def _duplicate_target(
    targets: Sequence[dict[str, Any]], pred: dict[str, Any], iou_threshold: float
) -> dict[str, Any] | None:
    for target in targets:
        score, _ = _geometry_score(target, pred, iou_threshold)
        if score > 0:
            return target
    return None


def _geometry_score(
    target: Mapping[str, Any], pred: Mapping[str, Any], iou_threshold: float
) -> tuple[float, str]:
    target_bbox = target["bbox"]
    pred_point = pred.get("click_point")
    pred_bbox = pred.get("bbox")
    scores: list[tuple[float, str]] = []
    if pred_point is not None and point_in_bbox(pred_point, target_bbox):
        scores.append((1.0, "click_point_in_target_bbox"))
    if pred_bbox is not None:
        overlap = bbox_iou(target_bbox, pred_bbox)
        if overlap >= iou_threshold:
            scores.append((overlap, "bbox_iou"))
        if point_in_bbox(bbox_center(target_bbox), pred_bbox):
            scores.append((0.75, "target_center_inside_prediction_bbox"))
    if not scores:
        return 0.0, "position_miss"
    return max(scores, key=lambda item: item[0])


def _label_score(target: Mapping[str, Any], pred: Mapping[str, Any]) -> float:
    target_text = str(target.get("label", "")).strip().lower()
    pred_text = str(pred.get("label", "")).strip().lower()
    if not target_text or not pred_text:
        target_text = str(target.get("role", "")).strip().lower()
        pred_text = str(pred.get("role", "")).strip().lower()
    if not target_text or not pred_text:
        return 0.0
    if target_text == pred_text:
        return 1.0
    if target_text in pred_text or pred_text in target_text:
        return 0.85
    return SequenceMatcher(None, target_text, pred_text).ratio()


def _has_label(item: Mapping[str, Any]) -> bool:
    return bool(str(item.get("label", "")).strip())


def _public_item(item: Mapping[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": item.get("id"),
        "label": item.get("label", ""),
        "role": item.get("role", ""),
    }
    if "index" in item:
        data["index"] = item["index"]
    if item.get("bbox") is not None:
        data["bbox"] = bbox_to_dict(item["bbox"])
    if item.get("click_point") is not None:
        data["click_point"] = point_to_dict(item["click_point"])
    if item.get("confidence") is not None:
        data["confidence"] = item["confidence"]
    return data


def _coerce_item(
    item: Mapping[str, Any], field_name: str, *, allow_missing_bbox: bool = False
) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    bbox = _coerce_bbox(item.get("bbox"), f"{field_name}.bbox")
    if bbox is None and not allow_missing_bbox:
        raise ValueError(f"{field_name}.bbox is required")
    click_point = _coerce_point(
        item.get("click_point", item.get("point")), f"{field_name}.click_point"
    )
    if click_point is None and "x" in item and "y" in item:
        click_point = (_number(item["x"], f"{field_name}.x"), _number(item["y"], f"{field_name}.y"))
    if click_point is None and bbox is not None:
        click_point = bbox_center(bbox)
    return {
        "id": item.get("id"),
        "index": item.get("index"),
        "label": str(item.get("label", "")),
        "role": str(item.get("role", item.get("type", ""))),
        "bbox": bbox,
        "click_point": click_point,
        "confidence": item.get("confidence"),
    }


def _coerce_bbox(value: Any, field_name: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        x1 = _number(value.get("x1", value.get("left")), f"{field_name}.x1")
        y1 = _number(value.get("y1", value.get("top")), f"{field_name}.y1")
        if "x2" in value or "right" in value:
            x2 = _number(value.get("x2", value.get("right")), f"{field_name}.x2")
        else:
            x2 = x1 + _number(value.get("width"), f"{field_name}.width")
        if "y2" in value or "bottom" in value:
            y2 = _number(value.get("y2", value.get("bottom")), f"{field_name}.y2")
        else:
            y2 = y1 + _number(value.get("height"), f"{field_name}.height")
        return _ordered_bbox((x1, y1, x2, y2))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        return _ordered_bbox(tuple(_number(v, field_name) for v in value))
    raise ValueError(f"{field_name} must be bbox mapping or [x1,y1,x2,y2]")


def _coerce_point(value: Any, field_name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _number(value.get("x"), f"{field_name}.x"), _number(
            value.get("y"), f"{field_name}.y"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return _number(value[0], f"{field_name}.x"), _number(value[1], f"{field_name}.y")
    raise ValueError(f"{field_name} must be point mapping or [x,y]")


def _number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _ordered_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1 = max(lx1, rx1)
    iy1 = max(ly1, ry1)
    ix2 = min(lx2, rx2)
    iy2 = min(ly2, ry2)
    intersection = bbox_area((ix1, iy1, ix2, iy2))
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_to_dict(bbox: tuple[float, float, float, float]) -> dict[str, int | float]:
    x1, y1, x2, y2 = bbox
    return {
        "x1": _clean(x1),
        "y1": _clean(y1),
        "x2": _clean(x2),
        "y2": _clean(y2),
        "width": _clean(x2 - x1),
        "height": _clean(y2 - y1),
    }


def point_to_dict(point: tuple[float, float]) -> dict[str, int | float]:
    return {"x": _clean(point[0]), "y": _clean(point[1])}


def _clean(value: float) -> int | float:
    return int(round(value)) if abs(value - round(value)) < 1e-9 else round(value, 3)
