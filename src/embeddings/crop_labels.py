"""Read the owner's crop labels from a Label Studio JSON export (not the COCO export, which has no choices)."""
from collections import Counter


def latest_annotation(task: dict) -> dict | None:
    """The most recently updated annotation that was not skipped."""
    ok = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
    return max(ok, key=lambda a: (a.get("updated_at") or "", a.get("id") or 0)) if ok else None


def read_crop_labels(ls_export: list[dict], from_name: str, allowed: set, known_crops: set) -> tuple[dict, Counter]:
    """{crop id: {label, lead_time, updated_at}} and notes about what was skipped.
    A task with no label, a label outside `allowed`, or a crop id that is not in the sample is counted, not used."""
    labels, notes = {}, Counter()
    for task in ls_export:
        cid = (task.get("data") or {}).get("crop_id")
        if cid not in known_crops:
            notes["task_with_unknown_crop_id"] += 1
            continue
        ann = latest_annotation(task)
        if ann is None:
            notes["task_without_submitted_annotation"] += 1
            continue
        chosen = [c for r in ann.get("result", []) if r.get("from_name") == from_name and r.get("type") == "choices"
                  for c in r["value"]["choices"]]
        if len(chosen) != 1 or chosen[0] not in allowed:
            notes["annotation_without_exactly_one_valid_label"] += 1
            continue
        labels[cid] = {"label": chosen[0], "lead_time": ann.get("lead_time"), "updated_at": ann.get("updated_at")}
        if len(task.get("annotations", [])) > 1:
            notes["tasks_with_several_annotations_latest_used"] += 1
    return labels, notes


def audit_export(ls_export: list[dict], from_name: str, allowed: set, sample_ids: set) -> dict:
    """Crop ids (anonymous) that need a look, by problem. Nothing is dropped here: `read_crop_labels` decides what is used.
    - not_in_export: in the sample (tasks.json / mapping) but absent from the export
    - not_in_sample: in the export but not in the sample
    - duplicate_task: several export tasks with the same crop id
    - no_annotation: no submitted (non-skipped) annotation
    - several_annotations: more than one submitted annotation (the latest is used)
    - no_label / several_labels / invalid_label: the latest annotation has no choice, more than one, or a choice outside the label set
    """
    flags = {k: [] for k in ("not_in_export", "not_in_sample", "duplicate_task", "no_annotation", "several_annotations",
                             "no_label", "several_labels", "invalid_label")}
    seen = []
    for task in ls_export:
        cid = (task.get("data") or {}).get("crop_id")
        seen.append(cid)
        if cid not in sample_ids:
            flags["not_in_sample"].append(cid)
            continue
        ok = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not ok:
            flags["no_annotation"].append(cid)
            continue
        if len(ok) > 1:
            flags["several_annotations"].append(cid)
        ann = latest_annotation(task)
        chosen = [c for r in ann.get("result", []) if r.get("from_name") == from_name and r.get("type") == "choices" for c in r["value"]["choices"]]
        if not chosen:
            flags["no_label"].append(cid)
        elif len(chosen) > 1:
            flags["several_labels"].append(cid)
        elif chosen[0] not in allowed:
            flags["invalid_label"].append(cid)
    flags["not_in_export"] = sorted(sample_ids - set(seen))
    flags["duplicate_task"] = sorted({c for c in seen if seen.count(c) > 1})
    return flags
