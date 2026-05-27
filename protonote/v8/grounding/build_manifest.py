"""Build unified manifest.csv from all V8 image datasets.

Output columns (one row per labeled instance — an image may have many
rows if multiple labels apply):

    image_path,label,entity_type,dataset,raw_label

Usage:
    python -m protonote.v8.grounding.build_manifest \
        --cache-root cache/image_library \
        --output cache/image_library/processed/manifest.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

from protonote.v8.grounding.dataset_mappers import (
    CHEMEQ25_MAP,
    PHYSICS27_MAP,
    VECTOR_LABPICS_MATERIAL_MAP,
    VECTOR_LABPICS_VESSEL_MAP,
    WIKIMEDIA_MAP,
)


# ============================================================
# ChemEq25 (YOLO format)
# ============================================================

def build_chemeq25_rows(dataset_root: Path,
                                 rows: list[dict]) -> dict:
    """Parse ChemEq25 (YOLO format) → rows.

    Each label file lists 0+ classes per image (one bbox per line);
    we add one row per UNIQUE class per image.
    """
    yaml_file = dataset_root / "data.yaml"
    if not yaml_file.exists():
        return {"images": 0, "rows": 0, "unmapped": 0}

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    n_imgs = 0
    n_rows = 0
    n_unmapped = 0
    seen_unmapped = set()

    for split in ("train", "valid", "test"):
        imgs_dir = dataset_root / split / "images"
        lbls_dir = dataset_root / split / "labels"
        if not imgs_dir.exists():
            continue
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbls_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            classes_in_image: set[str] = set()
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_idx = int(parts[0])
                    except ValueError:
                        continue
                    if 0 <= cls_idx < len(class_names):
                        classes_in_image.add(class_names[cls_idx])
            n_imgs += 1
            for raw in classes_in_image:
                mapping = CHEMEQ25_MAP.get(raw)
                if mapping is None:
                    n_unmapped += 1
                    seen_unmapped.add(raw)
                    continue
                label, etype = mapping
                rows.append({
                    "image_path": str(img_path.resolve()),
                    "label": label,
                    "entity_type": etype,
                    "dataset": "chemeq25",
                    "raw_label": raw,
                })
                n_rows += 1

    return {"images": n_imgs, "rows": n_rows,
              "unmapped": n_unmapped,
              "unmapped_keys": sorted(seen_unmapped)}


# ============================================================
# Vector-LabPics (Medical & Chemistry use the same schema)
# ============================================================

# Generic vessel/material class names we treat as "fallback" — pick a
# more-specific class if present in the same annotation.
_GENERIC_VESSEL_NAMES = {"Vessel"}


def _pick_specific_vessel(class_names: list[str]) -> str | None:
    """Among All_ClassNames for one vessel, choose the most specific
    one that we can map. Skip generic 'Vessel' if a subclass is present."""
    candidates = [c for c in class_names if c in VECTOR_LABPICS_VESSEL_MAP]
    if not candidates:
        return None
    specific = [c for c in candidates if c not in _GENERIC_VESSEL_NAMES]
    if specific:
        return specific[0]
    return candidates[0]


def _pick_specific_material(class_names: list[str]) -> str | None:
    candidates = [c for c in class_names if c in VECTOR_LABPICS_MATERIAL_MAP]
    return candidates[0] if candidates else None


def build_vector_labpics_rows(dataset_root: Path,
                                          variant: str,
                                          rows: list[dict]) -> dict:
    """Parse Vector-LabPics V2 (Medical or Chemistry).

    Each instance directory has:
      - Image.jpg / Image.png
      - Data.json — annotation with "Vessels" (and possibly "Parts" /
        materials) sections, each with All_ClassNames etc.

    We only need ONE entry per instance dir per detected (vessel OR
    material) class — for whole-image SigLIP2 embedding, we don't need
    per-mask precision.
    """
    n_imgs = 0
    n_rows = 0
    n_unmapped = 0
    seen_unmapped: set[str] = set()

    # Walk all dirs containing Data.json (excluding EvaluationScripts)
    for data_json in dataset_root.rglob("Data.json"):
        if "EvaluationScripts" in data_json.parts:
            continue
        if "Categories" in data_json.parts:
            continue
        instance_dir = data_json.parent
        # Image file in same dir
        img_file = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = instance_dir / f"Image{ext}"
            if p.exists():
                img_file = p
                break
        if img_file is None:
            continue
        try:
            with open(data_json) as f:
                d = json.load(f)
        except Exception:
            continue

        n_imgs += 1
        labels_added_for_this_img: set[str] = set()

        # Vessels section — iterate each detected vessel
        vessels = d.get("Vessels") or {}
        for vid, vinfo in (vessels.items() if isinstance(vessels, dict)
                              else enumerate(vessels)):
            class_names = vinfo.get("All_ClassNames", []) if isinstance(vinfo, dict) else []
            picked = _pick_specific_vessel(class_names)
            if picked is None:
                for c in class_names:
                    if c not in VECTOR_LABPICS_VESSEL_MAP and c:
                        n_unmapped += 1
                        seen_unmapped.add(c)
                continue
            label, etype = VECTOR_LABPICS_VESSEL_MAP[picked]
            if label in labels_added_for_this_img:
                continue
            labels_added_for_this_img.add(label)
            rows.append({
                "image_path": str(img_file.resolve()),
                "label": label,
                "entity_type": etype,
                "dataset": f"vector_labpics_{variant}",
                "raw_label": picked,
            })
            n_rows += 1

        # Materials section if present (key name varies)
        for mat_key in ("Materials", "MaterialsAndParts"):
            mats = d.get(mat_key)
            if not isinstance(mats, dict):
                continue
            for mid, minfo in mats.items():
                class_names = minfo.get("All_ClassNames", []) if isinstance(minfo, dict) else []
                picked = _pick_specific_material(class_names)
                if picked is None:
                    continue
                label, etype = VECTOR_LABPICS_MATERIAL_MAP[picked]
                if label in labels_added_for_this_img:
                    continue
                labels_added_for_this_img.add(label)
                rows.append({
                    "image_path": str(img_file.resolve()),
                    "label": label,
                    "entity_type": etype,
                    "dataset": f"vector_labpics_{variant}",
                    "raw_label": picked,
                })
                n_rows += 1

    return {"images": n_imgs, "rows": n_rows,
              "unmapped": n_unmapped,
              "unmapped_keys": sorted(seen_unmapped)[:30]}


# ============================================================
# Physics-27 (YOLO format, same shape as ChemEq25)
# ============================================================

def build_physics27_rows(dataset_root: Path,
                                  rows: list[dict]) -> dict:
    """Parse Physics-27 (YOLO format, 27 classes) → rows.

    Layout: dataset_root/{train,valid,test}/{images,labels}/. data.yaml
    holds the class-name list (already mapped in PHYSICS27_MAP).
    """
    yaml_file = dataset_root / "data.yaml"
    if not yaml_file.exists():
        return {"images": 0, "rows": 0, "skip": "no data.yaml"}

    with open(yaml_file) as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get("names", [])
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    n_imgs = 0
    n_rows = 0
    n_unmapped = 0
    seen_unmapped: set[str] = set()

    for split in ("train", "valid", "test"):
        imgs_dir = dataset_root / split / "images"
        lbls_dir = dataset_root / split / "labels"
        if not imgs_dir.exists():
            continue
        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl_path = lbls_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            classes_in_image: set[str] = set()
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_idx = int(parts[0])
                    except ValueError:
                        continue
                    if 0 <= cls_idx < len(class_names):
                        classes_in_image.add(class_names[cls_idx])
            n_imgs += 1
            for raw in classes_in_image:
                mapping = PHYSICS27_MAP.get(raw)
                if mapping is None:
                    n_unmapped += 1
                    seen_unmapped.add(raw)
                    continue
                label, etype = mapping
                rows.append({
                    "image_path": str(img_path.resolve()),
                    "label": label,
                    "entity_type": etype,
                    "dataset": "physics27",
                    "raw_label": raw,
                })
                n_rows += 1

    return {"images": n_imgs, "rows": n_rows,
              "unmapped": n_unmapped,
              "unmapped_keys": sorted(seen_unmapped)}


# ============================================================
# Wikimedia Commons crawl (per-category manifest.jsonl)
# ============================================================

def build_wikimedia_rows(wikimedia_root: Path,
                                   rows: list[dict]) -> dict:
    """Parse `cache/image_library/raw/wikimedia/<slug>/` directories.

    Each subdir from `scripts/v8_crawl_wikimedia.py` has manifest.jsonl
    (per-image metadata) plus the image files themselves. WIKIMEDIA_MAP
    is the source of truth for (label, entity_type).
    """
    n_imgs = 0
    n_rows = 0
    n_unmapped = 0
    seen_unmapped: set[str] = set()

    if not wikimedia_root.exists():
        return {"images": 0, "rows": 0, "skip": "missing"}

    for cat_dir in sorted(wikimedia_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        slug = cat_dir.name
        mapping = WIKIMEDIA_MAP.get(slug)
        if mapping is None:
            n_unmapped += 1
            seen_unmapped.add(slug)
            continue
        label, etype = mapping
        mf = cat_dir / "manifest.jsonl"
        if mf.exists():
            with mf.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    fname = rec.get("fname")
                    if not fname:
                        continue
                    p = cat_dir / fname
                    if not p.exists():
                        continue
                    n_imgs += 1
                    rows.append({
                        "image_path": str(p.resolve()),
                        "label": label,
                        "entity_type": etype,
                        "dataset": "wikimedia",
                        "raw_label": slug,
                    })
                    n_rows += 1
        else:
            # No manifest yet — fall back to scanning image files in dir
            for p in cat_dir.iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    n_imgs += 1
                    rows.append({
                        "image_path": str(p.resolve()),
                        "label": label,
                        "entity_type": etype,
                        "dataset": "wikimedia",
                        "raw_label": slug,
                    })
                    n_rows += 1

    return {"images": n_imgs, "rows": n_rows,
              "unmapped": n_unmapped,
              "unmapped_keys": sorted(seen_unmapped)}


# ============================================================
# Orchestrator
# ============================================================

def build_unified_manifest(cache_root: Path,
                                       output_path: Path) -> dict:
    rows: list[dict] = []
    stats: dict[str, dict] = {}

    # ChemEq25
    chemeq_root = cache_root / "raw" / "chemeq25" / "ChemistryLabApparatus-25"
    if chemeq_root.exists():
        print(f"\nProcessing ChemEq25 at {chemeq_root}")
        stats["chemeq25"] = build_chemeq25_rows(chemeq_root, rows)
        print(f"  → {stats['chemeq25']}")
    else:
        stats["chemeq25"] = {"images": 0, "rows": 0, "skip": "missing"}
        print(f"\nChemEq25 not found at {chemeq_root}")

    # Vector-LabPics Medical
    labpics_med_root = cache_root / "raw" / "vector_labpics" / "LabPics Medical"
    if labpics_med_root.exists():
        print(f"\nProcessing LabPics Medical at {labpics_med_root}")
        stats["labpics_medical"] = build_vector_labpics_rows(
            labpics_med_root, "medical", rows,
        )
        print(f"  → {stats['labpics_medical']}")
    else:
        stats["labpics_medical"] = {"images": 0, "rows": 0, "skip": "missing"}
        print(f"\nLabPics Medical not found")

    # Vector-LabPics Chemistry
    labpics_chem_root = cache_root / "raw" / "vector_labpics" / "LabPicsV2"
    if not labpics_chem_root.exists():
        labpics_chem_root = cache_root / "raw" / "vector_labpics" / "LabPics Chemistry"
    if labpics_chem_root.exists():
        print(f"\nProcessing LabPics Chemistry at {labpics_chem_root}")
        stats["labpics_chemistry"] = build_vector_labpics_rows(
            labpics_chem_root, "chemistry", rows,
        )
        print(f"  → {stats['labpics_chemistry']}")
    else:
        stats["labpics_chemistry"] = {"images": 0, "rows": 0, "skip": "missing"}
        print(f"\nLabPics Chemistry not found")

    # Physics-27 — try both extracted and raw layouts
    physics27_root = (
        cache_root / "raw" / "physics27" / "Physics lab equipment image dataset"
    )
    if physics27_root.exists():
        print(f"\nProcessing Physics-27 at {physics27_root}")
        stats["physics27"] = build_physics27_rows(physics27_root, rows)
        print(f"  → {stats['physics27']}")
    else:
        stats["physics27"] = {"images": 0, "rows": 0, "skip": "missing"}
        print(f"\nPhysics-27 not found at {physics27_root}")

    # Wikimedia Commons targeted crawl
    wikimedia_root = cache_root / "raw" / "wikimedia"
    if wikimedia_root.exists():
        print(f"\nProcessing Wikimedia crawl at {wikimedia_root}")
        stats["wikimedia"] = build_wikimedia_rows(wikimedia_root, rows)
        print(f"  → {stats['wikimedia']}")
    else:
        stats["wikimedia"] = {"images": 0, "rows": 0, "skip": "missing"}
        print(f"\nWikimedia crawl not found at {wikimedia_root}")

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image_path", "label", "entity_type", "dataset", "raw_label",
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote manifest: {output_path}  ({len(rows)} rows)")
    return stats


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                     default=Path("cache/image_library"))
    ap.add_argument("--output", type=Path,
                     default=Path("cache/image_library/processed/manifest.csv"))
    args = ap.parse_args()
    sys.path.insert(0,
                       str(Path(__file__).resolve().parents[3]))
    stats = build_unified_manifest(args.cache_root, args.output)
    print("\n=== Summary ===")
    total = 0
    for dataset, st in stats.items():
        print(f"  {dataset:25s}  images={st.get('images',0):>5d}  "
              f"rows={st.get('rows',0):>5d}  "
              f"unmapped={st.get('unmapped',0):>3d}")
        total += st.get("rows", 0)
    print(f"  {'TOTAL':25s}                rows={total}")
