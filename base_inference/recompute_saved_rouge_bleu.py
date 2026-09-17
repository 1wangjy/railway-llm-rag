"""Recompute character-level ROUGE/BLEU for saved second-round predictions."""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_PATTERN = "outputs/*_base_testdata799_system_max1024_predictions.json"
METRIC_PROTOCOL = "constrast_infer_char_level_v2_20260825"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute ROUGE/BLEU without regenerating model answers."
    )
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["INFER_METRICS_ONLY"] = "1"
    import infer_common as shared
    import infer_config as metric_config

    shared.cfg = metric_config
    paths = sorted(Path(".").glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No result files match: {args.pattern}")

    backup_root = Path("metric_recalc_backups") / datetime.now().strftime(
        "%Y%m%d_%H%M%S_char_level_v2"
    )
    if not args.dry_run:
        backup_root.mkdir(parents=True, exist_ok=False)

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        records = result.get("records") or []
        predictions = [str(record.get("prediction", "")) for record in records]
        references = [str(record.get("reference", "")) for record in records]
        if len(records) != 799:
            raise ValueError(f"{path}: expected 799 records, found {len(records)}")

        rouge = shared.compute_average_rouge(predictions, references)
        bleu = shared.compute_corpus_bleu(
            predictions,
            references,
            max_order=metric_config.bleu_max_order,
            smooth_value=metric_config.bleu_smooth_value,
        )
        updated = {
            "eval_rouge1": rouge["rouge1"],
            "eval_rouge2": rouge["rouge2"],
            "eval_rougeL": rouge["rougeL"],
            "eval_bleu4": bleu,
        }
        old = {key: result.get("metrics", {}).get(key) for key in updated}
        print(path)
        print("  old:", old)
        print("  new:", updated)

        if args.dry_run:
            continue
        backup_path = backup_root / path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        result.setdefault("metrics", {}).update(updated)
        result["metric_protocol"] = METRIC_PROTOCOL
        result["tokenization"] = (
            "character-level, ignore whitespace, preserve punctuation and case"
        )
        result["bleu_smoothing"] = metric_config.bleu_smooth_value
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)

    if args.dry_run:
        print("Dry run complete; no files were modified.")
    else:
        print(f"Backups saved under: {backup_root}")


if __name__ == "__main__":
    main()
