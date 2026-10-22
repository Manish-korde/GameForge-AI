"""Create stratified human-review sheets and summarize caption-label reliability."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "evilsocket/alucard-sprites"
DEFAULT_INDICES = ROOT / "config" / "unique_indices.json"
VERDICTS = {
    "correct": "correct",
    "✅ correct": "correct",
    "incorrect": "incorrect",
    "❌ incorrect": "incorrect",
    "ambiguous": "ambiguous",
    "⚠️ ambiguous": "ambiguous",
    "⚠ ambiguous": "ambiguous",
}

# Captions are matched as phrases, not inferred from image content. These
# definitions select review strata; they do not establish that a label is true.
LABELS: dict[str, dict[str, tuple[str, ...]]] = {
    "object_type": {
        "character": ("character", "hero", "player character"),
        "warrior": ("warrior",),
        "wizard_mage": ("wizard", "mage", "magician", "spellcaster"),
        "monster": ("monster", "creature", "enemy"),
        "slime": ("slime",),
        "dragon": ("dragon", "wyvern"),
        "animal": ("animal", "beast", "wolf", "cat", "dog", "bird"),
        "headwear": ("helmet", "hat", "hood", "headwear", "crown"),
        "weapon": ("weapon",),
        "bow": ("bow", "crossbow"),
        "staff_wand": ("staff", "wand"),
        "potion": ("potion", "elixir"),
        "chest": ("chest", "treasure box"),
        "coin_gem": ("coin", "gem", "jewel", "crystal"),
        "key": ("key",),
        "tree_plant": ("tree", "plant", "flower", "bush", "shrub"),
        "heart_health_icon": ("heart", "health icon", "life icon"),
        "sword": ("sword", "blade"),
        "shield": ("shield",),
        "armor": ("armor", "armour"),
        "food": ("food", "fruit", "meat", "bread"),
        "tile_environment": ("tile", "terrain", "floor", "wall", "environment"),
        "effect_spell": ("effect", "spell", "magic effect", "projectile"),
    },
    "color": {
        "gray": ("gray", "grey"),
        "brown": ("brown",),
        "colorful": ("colorful", "colourful", "multicolored", "multi-colored"),
        "dark": ("dark",),
        "red": ("red",),
        "blue": ("blue",),
        "green": ("green",),
        "yellow_gold": ("yellow", "gold", "golden"),
        "purple": ("purple", "violet"),
        "black": ("black",),
        "white": ("white",),
        "orange": ("orange",),
        "pink": ("pink",),
    },
    "size": {
        "small": ("small", "tiny"),
        "medium": ("medium-sized", "medium sized", "medium"),
        "large": ("large", "big", "huge"),
    },
    "view_angle": {
        "front": ("front view", "front-facing", "frontal view", "facing front"),
        "back": ("back view", "rear view", "facing away"),
        "side": ("side view", "profile view", "side-facing"),
        "three_quarter": (
            "three-quarter view",
            "three quarter view",
            "3/4 view",
            "three-quarter",
        ),
    },
}


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = re.escape(phrase).replace(r"\ ", r"[\s-]+")
    return re.compile(rf"(?<!\w){words}(?!\w)", re.IGNORECASE)


PATTERNS = {
    dimension: {
        label: tuple(_phrase_pattern(phrase) for phrase in phrases)
        for label, phrases in labels.items()
    }
    for dimension, labels in LABELS.items()
}


def matching_labels(caption: str) -> list[tuple[str, str]]:
    """Return every configured (dimension, label) phrase found in a caption."""
    return [
        (dimension, label)
        for dimension, groups in PATTERNS.items()
        for label, patterns in groups.items()
        if any(pattern.search(caption) for pattern in patterns)
    ]


def wilson_interval(successes: int, trials: int) -> tuple[float, float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _get_dataset(dataset_name: str) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it in the active "
            "environment with: python -m pip install datasets"
        ) from exc
    return load_dataset(dataset_name)


def _reservoir_sample(
    dataset: Any, indices: list[int], per_label: int, seed: int
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    populations: Counter[str] = Counter(
        {
            f"{dimension}/{label}": 0
            for dimension, labels in LABELS.items()
            for label in labels
        }
    )
    for source_index in indices:
        item = dataset[source_index]
        caption_value = item.get("text")
        if not isinstance(caption_value, str):
            raise ValueError(
                f"Dataset row {source_index} has no string 'text' caption."
            )
        matches = matching_labels(caption_value)
        if not matches:
            continue
        image_value = item.get("image")
        if image_value is None:
            raise ValueError(f"Dataset row {source_index} has no 'image' value.")
        if not isinstance(image_value, Image.Image):
            image_value = Image.fromarray(image_value)
        image_value = image_value.convert("RGBA")
        if image_value.size != (128, 128):
            raise ValueError(
                f"Dataset row {source_index} is {image_value.size}, expected 128x128."
            )
        for dimension, label in matches:
            key = f"{dimension}/{label}"
            populations[key] += 1
            record = {
                "dataset_index": source_index,
                "caption": caption_value,
                "image": image_value.copy(),
            }
            sample = reservoirs[key]
            if len(sample) < per_label:
                sample.append(record)
            else:
                replacement = rng.randrange(populations[key])
                if replacement < per_label:
                    sample[replacement] = record
    return reservoirs, populations


def _draw_checkered_background(size: tuple[int, int]) -> Image.Image:
    background = Image.new("RGBA", size, (245, 245, 245, 255))
    draw = ImageDraw.Draw(background)
    square = 12
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            if (x // square + y // square) % 2:
                draw.rectangle(
                    (x, y, x + square - 1, y + square - 1),
                    fill=(225, 225, 225, 255),
                )
    return background


def _save_contact_sheets(
    rows: list[dict[str, Any]], sheet_dir: Path, page_size: int = 20
) -> list[str]:
    sheet_dir.mkdir(parents=True, exist_ok=True)
    columns = 4
    card_width, card_height = 360, 245
    pages: list[str] = []
    resampling = getattr(Image, "Resampling", Image).NEAREST
    for offset in range(0, len(rows), page_size):
        page_rows = rows[offset : offset + page_size]
        page_number = offset // page_size + 1
        page = Image.new(
            "RGB",
            (columns * card_width, math.ceil(len(page_rows) / columns) * card_height),
            "white",
        )
        draw = ImageDraw.Draw(page)
        for local_index, row in enumerate(page_rows):
            x = (local_index % columns) * card_width
            y = (local_index // columns) * card_height
            draw.rectangle(
                (x + 2, y + 2, x + card_width - 3, y + card_height - 3),
                outline=(130, 130, 130),
                width=1,
            )
            title = f"{row['sample_id']} | {row['focus_label']}"
            draw.text((x + 8, y + 8), title[:52], fill=(0, 0, 0))

            image = row.pop("_image")
            background = _draw_checkered_background((136, 136))
            thumbnail = image.copy()
            thumbnail.thumbnail((128, 128), resampling)
            image_x = x + 10 + (128 - thumbnail.width) // 2
            image_y = y + 42 + (128 - thumbnail.height) // 2
            background.alpha_composite(thumbnail, (image_x - (x + 10), image_y - (y + 42)))
            page.paste(background.convert("RGB"), (x + 10, y + 42))

            caption = row["caption"].encode("ascii", "replace").decode("ascii")
            lines = textwrap.wrap(caption, width=34)[:5]
            for line_number, line in enumerate(lines):
                draw.text(
                    (x + 160, y + 46 + line_number * 18),
                    line,
                    fill=(0, 0, 0),
                )
            draw.text((x + 10, y + 190), "Verdict: correct / incorrect / ambiguous", fill=(0, 0, 0))
            draw.text((x + 10, y + 211), f"Dataset row: {row['dataset_index']}", fill=(70, 70, 70))
            row["_image"] = image

        dimension = page_rows[0]["dimension"]
        label = page_rows[0]["label"]
        filename = f"{dimension}_{label}_page_{page_number:02d}.png"
        page.save(sheet_dir / filename)
        pages.append(filename)
    return pages


def create_review_sample(
    output_dir: Path,
    per_label: int = 100,
    seed: int = 42,
    dataset_name: str = DATASET_NAME,
    indices_path: Path = DEFAULT_INDICES,
) -> None:
    output_dir = output_dir.resolve()
    csv_path = output_dir / "review_samples.csv"
    if csv_path.exists():
        raise FileExistsError(
            f"{csv_path} already exists; choose a new output directory to "
            "avoid overwriting human verdicts."
        )
    if per_label < 1:
        raise ValueError("--per-label must be at least 1.")
    if not indices_path.is_file():
        raise FileNotFoundError(
            f"Unique-index file not found: {indices_path}. Use the same "
            "deduplicated indices as the AE/VAE experiments."
        )

    indices = json.loads(indices_path.read_text(encoding="utf-8"))
    if not isinstance(indices, list) or not indices or not all(
        isinstance(index, int) and index >= 0 for index in indices
    ):
        raise ValueError(f"Invalid unique-index list in {indices_path}.")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate indices found in {indices_path}.")

    dataset = _get_dataset(dataset_name)
    raw_train = dataset["train"]
    if max(indices) >= len(raw_train):
        raise ValueError(
            f"Index {max(indices)} exceeds dataset train split length "
            f"{len(raw_train)}."
        )

    reservoirs, populations = _reservoir_sample(
        raw_train, indices, per_label=per_label, seed=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = output_dir / "contact_sheets"
    rows: list[dict[str, Any]] = []
    strata = []
    for key in sorted(populations):
        dimension, label = key.split("/", 1)
        candidates = reservoirs.get(key, [])
        stratum_rows = []
        for ordinal, record in enumerate(candidates, start=1):
            sample_id = f"{dimension}-{label}-{ordinal:04d}"
            row = {
                "sample_id": sample_id,
                "dimension": dimension,
                "label": label,
                "focus_label": label.replace("_", " "),
                "dataset_index": record["dataset_index"],
                "caption": record["caption"],
                "verdict": "",
                "_image": record["image"],
            }
            stratum_rows.append(row)
            rows.append(row)
        pages = _save_contact_sheets(stratum_rows, sheet_dir) if stratum_rows else []
        strata.append(
            {
                "dimension": dimension,
                "label": label,
                "population": populations[key],
                "sampled": len(candidates),
                "pages": pages,
            }
        )

    with csv_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "sample_id",
                "dimension",
                "label",
                "dataset_index",
                "caption",
                "verdict",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    manifest = {
        "dataset": dataset_name,
        "unique_indices_file": str(indices_path),
        "unique_image_count": len(indices),
        "seed": seed,
        "per_label_target": per_label,
        "sampled_label_examples": len(rows),
        "unique_sprite_count_in_sample": len(
            {row["dataset_index"] for row in rows}
        ),
        "verdict_values": ["correct", "incorrect", "ambiguous"],
        "sampling_method": (
            "Independent deterministic reservoir sample within each caption "
            "label stratum; rare strata are fully reviewed when below target."
        ),
        "strata": strata,
    }
    (output_dir / "sample_design.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_index_html(output_dir, strata, len(rows))
    print(f"Created {len(rows)} label-example reviews across {len(strata)} strata.")
    print(f"Unique sprites represented: {manifest['unique_sprite_count_in_sample']}")
    print(f"Review index: {output_dir / 'index.html'}")
    print(f"Verdict CSV:  {csv_path}")


def _write_index_html(output_dir: Path, strata: list[dict[str, Any]], row_count: int) -> None:
    items = []
    for stratum in strata:
        links = ", ".join(
            f'<a href="contact_sheets/{page}">{page}</a>'
            for page in stratum["pages"]
        )
        items.append(
            "<tr>"
            f"<td>{stratum['dimension']}</td>"
            f"<td>{stratum['label'].replace('_', ' ')}</td>"
            f"<td>{stratum['population']}</td>"
            f"<td>{stratum['sampled']}</td>"
            f"<td>{links or 'No matching captions'}</td>"
            "</tr>"
        )
    html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Alucard caption review</title>
<h1>Alucard dataset-caption human review</h1>
<p>These are samples from the deduplicated Alucard sprite corpus. Each
contact-sheet card shows the actual sprite beside its exact dataset caption.
Review the caption claim for the card's focus label, then enter one value in
<code>review_samples.csv</code>: <code>correct</code>, <code>incorrect</code>,
or <code>ambiguous</code>. A sprite can appear in more than one stratum.</p>
<p>Sampled label-example rows: ROW_COUNT. See the CSV and sample_design.json
for row IDs, population counts, seed, and sampling method.</p>
<table border="1" cellpadding="6"><thead><tr><th>Dimension</th><th>Focus label</th>
<th>Matching unique captions</th><th>Sampled</th><th>Contact sheets</th></tr></thead>
<tbody>TABLE_ROWS</tbody></table>
<p>After every row has a human verdict, run
<code>python scripts/review_alucard_captions.py finalize --input-dir
evaluation/alucard_label_review</code>.</p></html>"""
    html = html.replace("ROW_COUNT", str(row_count)).replace(
        "TABLE_ROWS", "\n".join(items)
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def _normalized_verdict(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in VERDICTS:
        raise ValueError(
            f"Invalid verdict {value!r}; use correct, incorrect, or ambiguous."
        )
    return VERDICTS[normalized]


def finalize_review(input_dir: Path) -> Path:
    input_dir = input_dir.resolve()
    csv_path = input_dir / "review_samples.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Review CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The review CSV contains no sample rows.")

    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        verdict = _normalized_verdict(row.get("verdict", ""))
        grouped[(row["dimension"], row["label"])][verdict] += 1

    metrics = []
    for (dimension, label), counts in sorted(grouped.items()):
        total = sum(counts.values())
        correct = counts["correct"]
        accuracy = correct / total
        interval = wilson_interval(correct, total)
        if accuracy >= 0.90:
            qualitative = "Mostly correct"
        elif accuracy >= 0.75:
            qualitative = "Partially incorrect"
        else:
            qualitative = "Heavily mismatched"
        metrics.append(
            {
                "dimension": dimension,
                "label": label,
                "samples": total,
                "correct": correct,
                "incorrect": counts["incorrect"],
                "ambiguous": counts["ambiguous"],
                "accuracy": accuracy,
                "wilson_95": interval,
                "qualitative": qualitative,
            }
        )

    by_dimension: dict[str, Counter[str]] = defaultdict(Counter)
    for metric in metrics:
        by_dimension[metric["dimension"]].update(
            {
                "samples": metric["samples"],
                "correct": metric["correct"],
                "incorrect": metric["incorrect"],
                "ambiguous": metric["ambiguous"],
            }
        )
    total_samples = sum(row["samples"] for row in metrics)
    total_correct = sum(row["correct"] for row in metrics)
    total_ambiguous = sum(row["ambiguous"] for row in metrics)
    strict_accuracy = total_correct / total_samples if total_samples else 0.0
    min_lower_bound = min(
        (
            metric["wilson_95"][0]
            for metric in metrics
            if metric["samples"] >= 30
        ),
        default=0.0,
    )
    conditional_recommendation = (
        "Provisional pass under the predeclared screen (all strata with at least "
        "30 samples have a 95% Wilson lower bound >=80%, and ambiguity is <=10%). "
        "This does not validate caption combinations or text conditioning."
        if min_lower_bound >= 0.80 and total_ambiguous / total_samples <= 0.10
        else "Not reliable enough to approve conditional generation under the "
        "predeclared screen. Use unconditional training unless the failed "
        "strata are excluded and independently revalidated."
    )

    lines = [
        "# Alucard 280K Caption-Label Reliability — Human Review",
        "",
        "> This report is calculated only from verdicts entered by a human in "
        "`review_samples.csv`. It does not infer correctness from caption text "
        "or model predictions.",
        "",
        "## Per-label results",
        "",
        "| Dimension | Label | Samples | Correct | Incorrect | Ambiguous | Accuracy* | 95% Wilson CI | Assessment |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in metrics:
        lower, upper = metric["wilson_95"]
        lines.append(
            f"| {metric['dimension']} | {metric['label'].replace('_', ' ')} | "
            f"{metric['samples']} | {metric['correct']} | {metric['incorrect']} | "
            f"{metric['ambiguous']} | {metric['accuracy']:.1%} | "
            f"{lower:.1%}–{upper:.1%} | {metric['qualitative']} |"
        )
    lines.extend(
        [
            "",
            "* Accuracy is `correct / all reviewed examples`; ambiguous examples "
            "are not counted as confirmed-correct. Sample rows are label-example "
            "pairs, so a sprite present in multiple strata is counted once per "
            "focus label.",
            "",
            "## Dimension totals",
            "",
            "| Dimension | Label-example reviews | Correct | Incorrect | Ambiguous | Accuracy* |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dimension, counts in sorted(by_dimension.items()):
        n = counts["samples"]
        lines.append(
            f"| {dimension} | {n} | {counts['correct']} | {counts['incorrect']} | "
            f"{counts['ambiguous']} | {counts['correct'] / n:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Conditional-generation decision",
            "",
            f"- Overall strict accuracy: {strict_accuracy:.1%}.",
            f"- Most conservative 95% Wilson lower bound among strata with at "
            f"least 30 examples: {min_lower_bound:.1%}.",
            f"- Ambiguous share: {total_ambiguous / total_samples:.1%}.",
            f"- Screen: {conditional_recommendation}",
            "- This stratified label screen does not evaluate whether full "
            "multi-attribute captions are jointly correct; text-conditioned "
            "training needs a separate human review of complete captions.",
            "",
            "## Interpretation thresholds",
            "",
            "- At least 90% observed strict accuracy: mostly correct.",
            "- 75% to below 90%: partially incorrect.",
            "- Below 75%: heavily mismatched.",
            "- Conditional use is screened separately using the Wilson lower "
            "bound and ambiguity rate above; passing is not proof of broad "
            "generalization.",
            "",
        ]
    )
    output_path = input_dir / "label_reliability_report.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample_parser = subparsers.add_parser(
        "sample", help="create stratified image-caption contact sheets"
    )
    sample_parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation" / "alucard_label_review",
    )
    sample_parser.add_argument("--per-label", type=int, default=100)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.add_argument("--dataset", default=DATASET_NAME)
    sample_parser.add_argument("--unique-indices", type=Path, default=DEFAULT_INDICES)

    finalize_parser = subparsers.add_parser(
        "finalize", help="calculate label reliability from completed human verdicts"
    )
    finalize_parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "evaluation" / "alucard_label_review",
    )

    args = parser.parse_args()
    if args.command == "sample":
        create_review_sample(
            output_dir=args.output_dir,
            per_label=args.per_label,
            seed=args.seed,
            dataset_name=args.dataset,
            indices_path=args.unique_indices,
        )
    else:
        report = finalize_review(args.input_dir)
        print(f"Human-review report written to: {report}")


if __name__ == "__main__":
    main()
