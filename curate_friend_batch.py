#!/usr/bin/env python3
"""
Curate friend-provided images for Stage2.
- Extracts zips (already extracted), collects images, converts to RGB jpg,
  deduplicates, writes manifest, contact sheets, and report.
"""

import os, sys, hashlib, csv, shutil, math
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path("/root/meteorite_stage2/meteorite_convnext_repro")
EXTRACT = Path("/tmp/friend_curated_extract")

NEG_OUT = BASE / "data/curated_stage2/raw_neg/friend_batch_all"
POS_OUT = BASE / "data/curated_stage2/raw_pos/friend_batch_all"
AUDIT   = BASE / "outputs/curated_stage2_audit/friend_batch"
CONTACT = AUDIT / "contact_sheets"
MANIFEST = AUDIT / "friend_batch_manifest.csv"
REPORT  = AUDIT / "friend_batch_report.md"

# Supported image extensions (lowercase)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

OUT_QUALITY = 95

# ─── Source definitions ──────────────────────────────────────────────────────
# Each source: (extract_roots_to_search, source_zip_name, inferred_label or None)
# label=None means we derive from the folder name (0→neg, 1→pos)
SOURCES = [
    # 0.zip → label 0 (neg). Files are directly under the 0/ folder inside.
    (EXTRACT / "0_zip", "0.zip", 0),
    # 1.zip → label 1 (pos).
    (EXTRACT / "1_zip", "1.zip", 1),
    # 扩充数据.zip — files under folders named 0 or 1.
    (EXTRACT / "kuo_data/扩充数据", "扩充数据.zip", None),
]

os.makedirs(NEG_OUT, exist_ok=True)
os.makedirs(POS_OUT, exist_ok=True)
os.makedirs(CONTACT, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def pil_quality_flag(img: Image.Image, w: int, h: int) -> str:
    short = min(w, h)
    long_ = max(w, h)
    aspect = long_ / short if short > 0 else 999
    if aspect > 4:  # extreme_aspect_ratio
        return "extreme_aspect_ratio"
    if short < 180:
        return "low_res_short_side_lt180"
    if short < 240:
        return "borderline_res_short_side_180_239"
    return "ok_res_short_side_ge240"


# ─── Phase 1: collect all image records ──────────────────────────────────────
print("=" * 60)
print("Phase 1: Collecting image records from all sources")
print("=" * 60)

records = []  # list of dicts
sha1_set = set()  # dedup
dup_count = 0
unreadable_count = 0
ext_counter = Counter()
zip_counts = defaultdict(int)

# Per-source counters for report
source_found = defaultdict(int)

for source_root, zip_name, fixed_label in SOURCES:
    print(f"\nScanning source: {zip_name} from {source_root}")
    if not source_root.exists():
        print(f"  WARNING: source root {source_root} does not exist, skipping")
        continue

    # Gather image files
    all_files = []
    for fpath in source_root.rglob("*"):
        if fpath.is_file() and fpath.suffix.lower() in IMG_EXTS:
            all_files.append(fpath)

    for fpath in all_files:
        rel_path = fpath.relative_to(source_root)
        source_found[zip_name] += 1

        # Determine label
        if fixed_label is not None:
            label = fixed_label
        else:
            # Derive from parent folder name: '0' → 0 (neg), '1' → 1 (pos)
            parent_name = fpath.parent.name
            if parent_name == "0":
                label = 0
            elif parent_name == "1":
                label = 1
            else:
                print(f"  SKIP: {rel_path} — parent folder '{parent_name}' not 0 or 1, can't determine label")
                continue

        label_name = "neg" if label == 0 else "pos"

        # Try to open image
        try:
            img = Image.open(fpath)
            img.verify()  # check integrity
        except Exception as e:
            print(f"  UNREADABLE: {rel_path} — {e}")
            unreadable_count += 1
            records.append({
                "path": "",
                "label": label,
                "label_name": label_name,
                "source_zip": zip_name,
                "source_path": str(rel_path),
                "width": 0,
                "height": 0,
                "short_side": 0,
                "long_side": 0,
                "aspect_ratio": 0,
                "sha1": "",
                "status": "unreadable",
                "quality_flag": "unreadable",
                "notes": f"Could not open: {e}",
            })
            continue

        ext_counter[fpath.suffix.lower()] += 1

        # Re-open after verify (verify closes the file)
        try:
            img = Image.open(fpath)
            w, h = img.size
        except Exception as e:
            print(f"  SIZE FAIL: {rel_path} — {e}")
            unreadable_count += 1
            records.append({
                "path": "",
                "label": label,
                "label_name": label_name,
                "source_zip": zip_name,
                "source_path": str(rel_path),
                "width": 0,
                "height": 0,
                "short_side": 0,
                "long_side": 0,
                "aspect_ratio": 0,
                "sha1": "",
                "status": "unreadable",
                "quality_flag": "unreadable",
                "notes": f"Could not read size: {e}",
            })
            continue

        s = sha1_file(fpath)

        # Dedup check
        if s in sha1_set:
            dup_count += 1
            quality = "duplicate_exact"
            records.append({
                "path": "",
                "label": label,
                "label_name": label_name,
                "source_zip": zip_name,
                "source_path": str(rel_path),
                "width": w,
                "height": h,
                "short_side": min(w, h),
                "long_side": max(w, h),
                "aspect_ratio": round(max(w, h) / min(w, h), 3) if min(w, h) > 0 else 999,
                "sha1": s,
                "status": "duplicate",
                "quality_flag": "duplicate_exact",
                "notes": f"Duplicate of an earlier image (first occurrence already copied)",
            })
            continue

        sha1_set.add(s)
        quality = pil_quality_flag(img, w, h)

        short = min(w, h)
        long_ = max(w, h)
        aspect = round(long_ / short, 3) if short > 0 else 999

        records.append({
            "path": "(to be filled)",
            "label": label,
            "label_name": label_name,
            "source_zip": zip_name,
            "source_path": str(rel_path),
            "width": w,
            "height": h,
            "short_side": short,
            "long_side": long_,
            "aspect_ratio": aspect,
            "sha1": s,
            "status": "copied",
            "quality_flag": quality,
            "notes": "",
        })

print(f"\nTotal records collected: {len(records)}")
print(f"  Unreadable: {unreadable_count}")
print(f"  Duplicates (exact): {dup_count}")
print(f"  Unique images to copy: {len(records) - unreadable_count - dup_count}")


# ─── Phase 2: Copy (convert) unique images ──────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2: Converting and copying unique images")
print("=" * 60)

name_counters = {"neg": 0, "pos": 0}

# Re-key by sha1 for easy lookup during copy
unique_records = [r for r in records if r["status"] == "copied"]

for rec in unique_records:
    label_name = rec["label_name"]
    source_zip = rec["source_zip"]
    sha1_short = rec["sha1"][:10]
    w = rec["width"]
    h = rec["height"]
    source_path = rec["source_path"]

    name_counters[label_name] += 1
    idx = name_counters[label_name]

    fname = f"friend_{label_name}_{source_zip.replace('.zip','')}_{idx:05d}_{sha1_short}_{w}x{h}.jpg"
    if label_name == "neg":
        out_path = NEG_OUT / fname
    else:
        out_path = POS_OUT / fname

    # Find the source file — reconstruct from extraction tree
    # We need to locate it. Search in the same source root it came from.
    found_src = None
    for source_root, zip_name, _ in SOURCES:
        if zip_name == source_zip:
            candidate = source_root / source_path
            if candidate.exists():
                found_src = candidate
                break

    if found_src is None:
        print(f"  ERROR: Could not locate source file {source_path} for copying")
        rec["status"] = "missing"
        continue

    try:
        img = Image.open(found_src)
        # Convert to RGB (handles RGBA, P, CMYK, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(out_path, quality=OUT_QUALITY)
        rec["path"] = str(out_path.relative_to(BASE))
    except Exception as e:
        print(f"  CONVERT FAIL: {source_path} → {e}")
        rec["status"] = "convert_failed"
        rec["notes"] = f"Convert/save failed: {e}"
        continue

    if name_counters[label_name] % 50 == 0:
        print(f"  ... copied {name_counters[label_name]} {label_name} images")

copied_neg = name_counters["neg"]
copied_pos = name_counters["pos"]
print(f"\nCopied: {copied_neg} neg + {copied_pos} pos = {copied_neg + copied_pos} total")


# ─── Phase 3: Write manifest CSV ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 3: Writing manifest CSV")
print("=" * 60)

MANIFEST.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "path", "label", "label_name", "source_zip", "source_path",
    "width", "height", "short_side", "long_side", "aspect_ratio",
    "sha1", "status", "quality_flag", "notes"
]

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for rec in records:
        writer.writerow(rec)

print(f"Manifest written: {MANIFEST}")


# ─── Phase 4: Resolution distribution ────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 4: Computing distributions")
print("=" * 60)

res_buckets = {"<180": 0, "180-239": 0, ">=240": 0}
copied_records = [r for r in records if r["status"] == "copied"]
for rec in copied_records:
    ss = rec["short_side"]
    if ss < 180:
        res_buckets["<180"] += 1
    elif ss < 240:
        res_buckets["180-239"] += 1
    else:
        res_buckets[">=240"] += 1

# Count quality flags for copied records
quality_counts = Counter(r["quality_flag"] for r in copied_records)
print(f"Resolution buckets (copied images): {res_buckets}")
print(f"Quality flags: {dict(quality_counts)}")

# Extension distribution on all found images (from ext_counter)
print(f"Extension distribution: {dict(ext_counter)}")


# ─── Phase 5: Generate contact sheets ────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 5: Generating contact sheets")
print("=" * 60)

TILE_SIZE = (200, 200)
TILES_PER_PAGE = 30  # 6 cols × 5 rows
COLS, ROWS = 6, 5
PAGE_TILES = COLS * ROWS
MARGIN = 10
FONT_SIZE_SMALL = 10

# Try to get a small font — fallback to default
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE_SMALL)
except Exception:
    font = ImageFont.load_default()


def make_contact_sheet(pages, label_name, filename_prefix):
    """Save all pages for a given label."""
    for page_idx, tiles in enumerate(pages):
        n_tiles = len(tiles)
        cols = min(COLS, n_tiles) if n_tiles >= COLS else n_tiles
        rows_actual = math.ceil(n_tiles / COLS)
        # Use 6 cols-width even if last row has fewer tiles
        actual_cols = COLS

        total_w = actual_cols * TILE_SIZE[0] + (actual_cols + 1) * MARGIN
        total_h = rows_actual * TILE_SIZE[1] + (rows_actual + 1) * MARGIN

        sheet = Image.new("RGB", (total_w, total_h), (30, 30, 30))
        draw = ImageDraw.Draw(sheet)

        for i, tile_data in enumerate(tiles):
            row = i // COLS
            col = i % COLS
            x = MARGIN + col * (TILE_SIZE[0] + MARGIN)
            y = MARGIN + row * (TILE_SIZE[1] + MARGIN)

            tile_img = tile_data["thumb"]
            sheet.paste(tile_img, (x, y))

            # Label text
            txt = f"{label_name}_{tile_data['idx']:05d} {tile_data['w']}x{tile_data['h']} {tile_data['basename']}"
            draw.text((x + 2, y + 2), txt, fill=(255, 255, 0), font=font)

        fpath = CONTACT / f"{filename_prefix}_page{page_idx + 1}.png"
        sheet.save(fpath)
        print(f"  Saved: {fpath}")


def build_thumb(rec, out_dir):
    """Build a 200x200 thumbnail for a record. Returns dict or None."""
    if rec["status"] != "copied":
        return None
    fpath = BASE / rec["path"]
    if not fpath.exists():
        return None
    try:
        img = Image.open(fpath)
        img.thumbnail(TILE_SIZE, Image.LANCZOS)
        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        base_name = os.path.basename(rec["path"])
        return {
            "thumb": img,
            "idx": int(rec["path"].split("_")[2]) if len(rec["path"].split("_")) > 2 else 0,
            "w": rec["width"],
            "h": rec["height"],
            "basename": base_name,
        }
    except Exception:
        return None


# Build thumbs only for copied records
neg_thumbs = []
pos_thumbs = []
low_res_thumbs = []
borderline_thumbs = []

for rec in copied_records:
    thumb = build_thumb(rec, None)
    if thumb is None:
        continue
    if rec["label_name"] == "neg":
        neg_thumbs.append(thumb)
    else:
        pos_thumbs.append(thumb)

    ss = rec["short_side"]
    if ss < 180:
        low_res_thumbs.append(thumb)
    elif ss < 240:
        borderline_thumbs.append(thumb)

# Split into pages
def split_pages(thumbs):
    pages = []
    for i in range(0, len(thumbs), PAGE_TILES):
        pages.append(thumbs[i:i + PAGE_TILES])
    return pages

if neg_thumbs:
    make_contact_sheet(split_pages(neg_thumbs), "neg", "neg_overview")
if pos_thumbs:
    make_contact_sheet(split_pages(pos_thumbs), "pos", "pos_overview")
if low_res_thumbs:
    make_contact_sheet(split_pages(low_res_thumbs), "low_res", "low_res_examples")
if borderline_thumbs:
    make_contact_sheet(split_pages(borderline_thumbs), "borderline", "borderline_res_examples")

print(f"  neg_overview: {len(neg_thumbs)} tiles across {len(split_pages(neg_thumbs))} pages")
print(f"  pos_overview: {len(pos_thumbs)} tiles across {len(split_pages(pos_thumbs))} pages")
print(f"  low_res_examples: {len(low_res_thumbs)} tiles across {len(split_pages(low_res_thumbs))} pages")
print(f"  borderline_examples: {len(borderline_thumbs)} tiles across {len(split_pages(borderline_thumbs))} pages")


# ─── Phase 6: Write report ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 6: Writing report")
print("=" * 60)

total_found = sum(source_found.values())
total_copied = copied_neg + copied_pos

# Count unique images per source (among copied, non-duplicate)
source_unique = defaultdict(int)
for rec in copied_records:
    source_unique[rec["source_zip"]] += 1

report_lines = [
    "# Friend Batch Curation Report",
    "",
    f"**Date:** 2026-05-29",
    f"**Project:** meteorite_convnext_repro — Stage2 curated friend data",
    "",
    "---",
    "",
    "## Summary",
    "",
    f"| Metric | Count |",
    f"|---|---|",
    f"| Total images found across all zips | {total_found} |",
    f"| Copied to neg (label=0) | {copied_neg} |",
    f"| Copied to pos (label=1) | {copied_pos} |",
    f"| **Total copied** | **{total_copied}** |",
    f"| Unreadable | {unreadable_count} |",
    f"| Duplicates (exact SHA1, not copied) | {dup_count} |",
    "",
    "---",
    "",
    "## Per-Source Breakdown",
    "",
    "| Source ZIP | Files Found | Unique Copied |",
    "|---|---|---|",
]

for src_name in ["0.zip", "1.zip", "扩充数据.zip"]:
    found = source_found.get(src_name, 0)
    unique = source_unique.get(src_name, 0)
    report_lines.append(f"| {src_name} | {found} | {unique} |")

report_lines += [
    "",
    "---",
    "",
    "## Image Extension Distribution",
    "",
    "| Extension | Count |",
    "|---|---|",
]
for ext, cnt in sorted(ext_counter.items(), key=lambda x: -x[1]):
    report_lines.append(f"| {ext} | {cnt} |")

report_lines += [
    "",
    "---",
    "",
    "## Resolution Distribution (copied images only)",
    "",
    "| Short Side Range | Count |",
    "|---|---|",
    f"| < 180 px | {res_buckets['<180']} |",
    f"| 180 - 239 px | {res_buckets['180-239']} |",
    f"| >= 240 px | {res_buckets['>=240']} |",
    "",
    "---",
    "",
    "## Quality Flag Distribution (copied images only)",
    "",
    "| Flag | Count |",
    "|---|---|",
]
for flag, cnt in sorted(quality_counts.items(), key=lambda x: -x[1]):
    report_lines.append(f"| {flag} | {cnt} |")

report_lines += [
    "",
    "---",
    "",
    "## Recommendation",
    "",
    "Many images in this batch are approximately 256×190 pixels and may appear blurry. "
    "Because of the generally lower resolution and quality of friend-provided images, "
    "the initial neg-only scouting pass should use a **low weight** (0.5 or 0.75, not 1.0) "
    "to avoid over-weighting potentially noisy negative samples. "
    "Higher weights can be tested later once baseline performance is established.",
    "",
    "### Next Steps",
    "",
    "1. Run a neg-only scout with weight=0.5 or 0.75",
    "2. Evaluate precision/recall on the friend batch held-out set",
    "3. Optionally filter out extreme-aspect-ratio or very blurry images before full training",
    "4. Consider augmenting with crops/resizes to handle the 256×190 resolution profile",
    "",
    "---",
    "",
    "## Files Produced",
    "",
    f"- **Negatives:** `data/curated_stage2/raw_neg/friend_batch_all/` ({copied_neg} files)",
    f"- **Positives:** `data/curated_stage2/raw_pos/friend_batch_all/` ({copied_pos} files)",
    f"- **Manifest:** `{MANIFEST.relative_to(BASE)}`",
    f"- **Contact sheets:** `{CONTACT.relative_to(BASE)}/`",
    "",
]

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Report written: {REPORT}")


# ─── Done ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
print(f"  Neg copied:  {copied_neg}")
print(f"  Pos copied:  {copied_pos}")
print(f"  Unreadable:  {unreadable_count}")
print(f"  Duplicates:  {dup_count}")
print(f"  Manifest:    {MANIFEST}")
print(f"  Report:      {REPORT}")
print(f"  Contact:     {CONTACT}/")
