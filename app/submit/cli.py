"""回答CSVを検証し、問題がなければzipを作る。

  python -m app.submit.cli --run-id <id> --out submission.zip
  python -m app.submit.cli --validate <predictions.csv>
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

from app.config import CONFIG
from app.run.runner import load_questions, write_predictions
from app.submit.validate import validate

ZIP_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.zip$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--validate", default=None,
                    help="既存の predictions.csv を検証するだけ")
    args = ap.parse_args()

    if args.validate:
        r = validate(args.validate)
        print(r.render())
        return 0 if r.ok else 1

    if not args.run_id:
        print("--run-id か --validate が必要です")
        return 1

    run_dir = CONFIG.logs_dir / args.run_id
    if not run_dir.exists():
        print(f"ランが見つかりません: {run_dir}")
        return 1

    qcsv = Path(args.questions) if args.questions else None
    if qcsv is None:
        import json
        mf = run_dir / "manifest.json"
        if mf.exists():
            qcsv = Path(json.loads(mf.read_text(encoding="utf-8"))["questions_csv"])
    if qcsv is None or not qcsv.exists():
        print("質問CSVが特定できません（--questions で指定してください）")
        return 1

    all_q = load_questions(qcsv)
    pred = write_predictions(run_dir, all_q)
    r = validate(pred, expected_indices={q["index"] for q in all_q})
    print(r.render())
    if not r.ok:
        print("\n形式検証に失敗したため zip は作成しません。")
        return 1

    out = Path(args.out)
    if not ZIP_NAME_RE.match(out.name):
        print(f"zip名は英数字/_/- のみ: {out.name}")
        return 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred, arcname="predictions.csv")
    print(f"\nzip: {out.resolve()}  ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
