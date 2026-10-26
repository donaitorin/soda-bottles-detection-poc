from src.embeddings.crop_labels import latest_annotation, read_crop_labels

ALLOWED = {"coca_classic", "unknown", "not_a_bottle", "cannot_tell"}
KNOWN = {"s01_aaaaaaaa_000", "s01_aaaaaaaa_001", "s02_bbbbbbbb_000"}   # synthetic ids


def _task(cid, choices, **ann):
    result = [{"from_name": "product", "to_name": "crop", "type": "choices", "value": {"choices": choices}}]
    return {"data": {"crop_id": cid}, "annotations": [{"id": 1, "result": result, "lead_time": 3.5,
                                                         "updated_at": "2026-01-01T00:00:00Z", **ann}]}


def test_labels_are_read_with_lead_time():
    labels, notes = read_crop_labels([_task("s01_aaaaaaaa_000", ["coca_classic"])], "product", ALLOWED, KNOWN)
    assert labels["s01_aaaaaaaa_000"] == {"label": "coca_classic", "lead_time": 3.5, "updated_at": "2026-01-01T00:00:00Z"}
    assert not notes


def test_invalid_missing_skipped_and_unknown_tasks_are_counted_not_used():
    tasks = [_task("s01_aaaaaaaa_000", ["not_in_the_set"]),
             _task("s01_aaaaaaaa_001", ["unknown", "coca_classic"]),                 # two labels: ambiguous
             {"data": {"crop_id": "s02_bbbbbbbb_000"}, "annotations": []},
             _task("s99_zzzzzzzz_000", ["unknown"]),
             _task("s01_aaaaaaaa_001", ["unknown"], was_cancelled=True)]
    labels, notes = read_crop_labels(tasks, "product", ALLOWED, KNOWN)
    assert labels == {}
    assert notes["annotation_without_exactly_one_valid_label"] == 2 and notes["task_without_submitted_annotation"] == 2
    assert notes["task_with_unknown_crop_id"] == 1


def test_latest_annotation_wins():
    task = {"annotations": [{"id": 1, "updated_at": "a"}, {"id": 2, "updated_at": "b"}, {"id": 3, "updated_at": "z", "was_cancelled": True}]}
    assert latest_annotation(task)["id"] == 2
