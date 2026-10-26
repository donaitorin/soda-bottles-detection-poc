"""Read the owner's crop labels from a Label Studio JSON export (Export -> JSON, NOT COCO) and write a clean CSV.

Also copies the export, untouched, with a UTC timestamp to the backup folder (the backup routine: one export after each
labeling session). Prints the counts per label, per store, the `not_a_bottle` share (a direct measure of the pseudo-label
PRECISION on val: the share of pseudo-label boxes that are not bottles; it says nothing about missed bottles) and warns if
there are too few `unknown` crops to calibrate the rejection rules (add a targeted batch then).

Usage: python scripts/export_crop_labels.py --ls-export FILE [--name val_sample]
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.paths import check_output_dir  # noqa: E402
from src.embeddings.crop_labels import audit_export, read_crop_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stage6.yaml")
    parser.add_argument("--ls-export", required=True, help="Label Studio JSON export of the crop project")
    parser.add_argument("--name", action="append", help="sample name(s) of mapping.csv to join (default: crops.sample.name); repeat for several batches")
    args = parser.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    s1 = yaml.safe_load((REPO_ROOT / "configs/stage1.yaml").read_text())
    le = cfg["labels_export"]
    protected = [REPO_ROOT / d for d in s1["protected_dirs"]]
    out_dir, backup_dir = REPO_ROOT / le["output_dir"], REPO_ROOT / le["backup_dir"]
    for d in (out_dir, backup_dir):
        check_output_dir(d, protected)
        d.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for name in args.name or [cfg["crops"]["sample"]["name"]]:
        with (REPO_ROOT / cfg["crops"]["mapping_dir"] / name / "mapping.csv").open() as f:
            for r in csv.DictReader(f):
                mapping[r["crop_id"]] = {**r, "sample": name}
    src = Path(args.ls_export)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    shutil.copyfile(src, backup_dir / f"crop_ls_export_{stamp}.json")
    allowed = set(cfg["products"]) | set(cfg["other_labels"])
    labels, notes = read_crop_labels(json.loads(src.read_text()), cfg["label_studio"]["from_name"], allowed, set(mapping))

    path = out_dir / "val_sample_labels.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["crop_id", "photo_id", "store_code", "sample", "label", "lead_time_s"])
        w.writeheader()
        for cid in sorted(labels):
            m = mapping[cid]
            w.writerow({"crop_id": cid, "photo_id": m["photo_id"], "store_code": m["store_code"], "sample": m["sample"],
                        "label": labels[cid]["label"], "lead_time_s": labels[cid]["lead_time"]})
    counts = Counter(v["label"] for v in labels.values())
    n = len(labels)
    export = json.loads(src.read_text())
    sample_ids = set(mapping)
    tasks_ids = set()
    for name in args.name or [cfg["crops"]["sample"]["name"]]:
        tasks_path = REPO_ROOT / cfg["crops"]["mapping_dir"] / name / "tasks.json"
        if tasks_path.exists():
            tasks_ids |= {t["data"]["crop_id"] for t in json.loads(tasks_path.read_text())}
    flags = audit_export(export, cfg["label_studio"]["from_name"], allowed, sample_ids)
    if tasks_ids and tasks_ids != sample_ids:
        print(f"WARNING: tasks.json and mapping.csv hold different crop ids ({len(tasks_ids ^ sample_ids)} differ)")
    bad = {k: v for k, v in flags.items() if v}
    print("audit of the export: " + ("clean (every sample crop has exactly one valid label, ids match tasks.json and mapping.csv)" if not bad else ""))
    for k, v in bad.items():
        print(f"  FLAG {k}: {len(v)} crop(s): {', '.join(v[:20])}")
    by_store = {}
    for cid, v in labels.items():
        by_store.setdefault(mapping[cid]["store_code"], Counter())[v["label"]] += 1
    order = cfg["products"] + cfg["other_labels"]
    print("labels per store:")
    print("| store | " + " | ".join(order) + " | total |")
    for store in sorted(by_store):
        print(f"| {store} | " + " | ".join(str(by_store[store][l]) for l in order) + f" | {sum(by_store[store].values())} |")
    few = [p for p in cfg["products"] if counts[p] < 5]
    if few:
        print("LOW COUNT: fewer than 5 val crops for " + ", ".join(f"{p} ({counts[p]})" for p in few) + ": limits what can be concluded for them.")
    print(f"{n} of {len(mapping)} crops labeled; notes {dict(notes)}; backup copy in {backup_dir.relative_to(REPO_ROOT)}/; labels in {path.relative_to(REPO_ROOT)}")
    for label in cfg["products"] + cfg["other_labels"]:
        print(f"  {label}: {counts[label]}")
    if n:
        times = sorted(v["lead_time"] for v in labels.values() if v["lead_time"] is not None)
        if times:
            print(f"time per crop (Label Studio lead_time): median {times[len(times) // 2]:.1f} s, total {sum(times) / 60:.1f} min")
        nb = counts["not_a_bottle"]
        print(f"not_a_bottle: {nb} of {n} ({100 * nb / n:.1f}%) -> pseudo-label precision on val about {100 * (n - nb) / n:.1f}% "
              f"(share of boxes that are bottles; a sample estimate, says nothing about missed bottles)")
        if counts["unknown"] < le["min_unknown_warning"]:
            print(f"WARNING: only {counts['unknown']} unknown crops (under {le['min_unknown_warning']}): add a targeted batch of crops from photos with other brands "
                  f"(scripts/build_crop_sample.py --name val_unknowns --only-photos FILE --exclude val_sample).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
