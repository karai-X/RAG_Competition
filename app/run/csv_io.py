"""質問CSVの読み取りとC列への回答書き戻し。

UTF-8 BOM、UTF-8、CP932を読み分け、元の文字コード・BOM・改行・追加列を
維持したまま、回答列（C列）だけを原子的に更新する。
"""
from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CsvTable:
    rows: list[list[str]]
    encoding: str
    bom: bool
    newline: str


def read_csv_table(path: Path) -> CsvTable:
    data = path.read_bytes()
    bom = data.startswith(b"\xef\xbb\xbf")
    body = data[3:] if bom else data
    try:
        text = body.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = body.decode("cp932")
        encoding = "cp932"
        bom = False
    newline = "\r\n" if "\r\n" in text else "\n"
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise ValueError(f"CSVが空です: {path}")
    return CsvTable(rows=rows, encoding=encoding, bom=bom, newline=newline)


def _column(header: list[str], name: str, fallback: int) -> int:
    normalized = [str(value).strip().casefold() for value in header]
    return normalized.index(name.casefold()) if name.casefold() in normalized else fallback


def load_question_rows(path: Path) -> list[dict]:
    """質問CSVを読み、A列=index・B列=question・C列=answerを既定にする。"""
    table = read_csv_table(path)
    header = table.rows[0]
    if len(header) < 2:
        raise ValueError("質問CSVには少なくともA列(index)とB列(question)が必要です。")
    index_col = _column(header, "index", 0)
    question_col = _column(header, "question", 1)
    answer_col = _column(header, "answer", 2) if len(header) >= 3 else None
    required = max(index_col, question_col)

    questions: list[dict] = []
    seen: set[int] = set()
    for row_no, row in enumerate(table.rows[1:], 2):
        if len(row) <= required or not str(row[index_col]).strip():
            continue
        try:
            index = int(str(row[index_col]).strip())
        except ValueError as exc:
            raise ValueError(f"CSV {row_no}行目のindexが整数ではありません。") from exc
        if index in seen:
            raise ValueError(f"CSV内でindex={index}が重複しています。")
        question = str(row[question_col]).strip()
        if not question:
            raise ValueError(f"CSV {row_no}行目のquestionが空です。")
        seen.add(index)
        gt = (row[answer_col].strip()
              if answer_col is not None and len(row) > answer_col
              and row[answer_col].strip() else None)
        questions.append({"index": index, "question": question, "gt": gt})
    if not questions:
        raise ValueError("CSVに回答対象の質問がありません。")
    return sorted(questions, key=lambda item: item["index"])


def write_answers_to_c_column(path: Path, answers: dict[int, str],
                              missing_answer: str,
                              output: Path | None = None) -> Path:
    """C列がなければ追加し、指定された回答だけを書いて原子的に更新する。"""
    table = read_csv_table(path)
    header = table.rows[0]
    if len(header) < 2:
        raise ValueError("質問CSVには少なくともA列(index)とB列(question)が必要です。")
    if len(header) < 3:
        header.extend([""] * (3 - len(header)))
        header[2] = "answer"
    index_col = _column(header, "index", 0)
    updated = 0
    for row_no, row in enumerate(table.rows[1:], 2):
        if len(row) <= index_col or not str(row[index_col]).strip():
            continue
        try:
            index = int(str(row[index_col]).strip())
        except ValueError as exc:
            raise ValueError(f"CSV {row_no}行目のindexが整数ではありません。") from exc
        while len(row) < 3:
            row.append("")
        if index not in answers:
            continue
        row[2] = str(answers.get(index) or missing_answer)
        updated += 1
    if not updated:
        raise ValueError("指定された回答に対応するCSV行がありません。")

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator=table.newline)
    writer.writerows(table.rows)
    encoded = buffer.getvalue().encode(table.encoding)
    if table.bom:
        encoded = b"\xef\xbb\xbf" + encoded

    target = output or path
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_bytes(encoded)
    os.replace(tmp, target)
    return target
