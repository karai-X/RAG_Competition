"""経路と回答の描画。質問画面（実行中）と履歴画面で共用する。"""
from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from webui.core import RESULT_PREVIEW_CHARS

TOOL_LABELS = {
    "search": "意味検索", "grep": "全文検索", "read": "ファイルを読む",
    "read_image": "画像を見る", "ask_image": "画像に質問", "run_python": "コード実行",
    "diff": "版比較", "decrypt": "復号", "submit_answer": "回答を確定",
}


def fmt_time(t: float | None) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else ""


def _arg_summary(args: dict) -> str:
    for key in ("relpath", "path", "query", "pattern", "question", "relpath_a", "code"):
        if args.get(key):
            s = " ".join(str(args[key]).split())
            return s if len(s) <= 70 else s[:69] + "…"
    return ""


def _clip(text: str) -> str:
    if len(text) <= RESULT_PREVIEW_CHARS:
        return text
    return text[:RESULT_PREVIEW_CHARS] + f"\n…（以降 {len(text) - RESULT_PREVIEW_CHARS:,} 文字省略）"


def render_steps(steps: list[dict]) -> None:
    """経路を上から順に。ツール呼び出しは折りたたみ、引数と戻り値を中に置く。"""
    n = 0
    for step in steps:
        k = step.get("k")
        if k == "think":
            if step.get("text"):
                with st.chat_message("assistant", avatar=":material/psychology:"):
                    st.markdown(step["text"])
                    st.caption(f"LLM {step.get('sec', 0)}秒")
        elif k == "tool":
            n += 1
            name = step.get("name") or "?"
            label = TOOL_LABELS.get(name, name)
            summary = _arg_summary(step.get("args") or {})
            if step.get("pending"):
                status = " — 実行中…"
            elif step.get("blocked"):
                status = " — 差し戻し"
            else:
                status = f" — {step.get('sec', 0)}秒"
            icon = (":material/hourglass_top:" if step.get("pending")
                    else ":material/block:" if step.get("blocked")
                    else ":material/build:")
            title = f"{n}. {label}（{name}）" + (f"：{summary}" if summary else "") + status
            with st.expander(title, icon=icon):
                args = step.get("args") or {}
                code = args.get("code")
                if code:
                    st.code(code, language="python", wrap_lines=True)
                    args = {a: v for a, v in args.items() if a != "code"}
                if args:
                    st.code(json.dumps(args, ensure_ascii=False, indent=1), language="json",
                            wrap_lines=True)
                if not step.get("pending") and (step.get("out") or name != "submit_answer"):
                    st.caption("戻り値")
                    st.code(_clip(step.get("out") or "（空）"), language=None, wrap_lines=True)
        elif k == "error":
            st.error(step.get("text") or "エラー", icon=":material/error:")


def render_answer(rec: dict) -> None:
    """最終回答と数値。rec は a / ev / turns / sec / tok を持つ。

    回答の正誤は判定していないので、正解に見える色や印は付けない。
    """
    answer = rec.get("a")
    if answer:
        with st.container(border=True):
            st.markdown(answer)
    else:
        st.warning("回答は得られませんでした。", icon=":material/help:")

    cols = st.columns(4)
    cols[0].metric("ターン", rec.get("turns") if rec.get("turns") is not None else "—")
    cols[1].metric("所要時間", f"{rec.get('sec') or 0:.0f}秒")

    if rec.get("ev"):
        st.markdown("**根拠にした資料**")
        st.markdown("\n".join(f"- `{e}`" for e in rec["ev"]))
    meta = [m for m in (rec.get("model") and f"モデル: {rec['model']}",
                        rec.get("tok") and f"トークン: {rec['tok']:,}",
                        rec.get("run_id") and f"記録: logs/{rec['run_id']}/") if m]
    if meta:
        st.caption(" ・ ".join(meta))


def question_browser(group, rows: list[dict], index_label: str) -> None:
    """質問の一覧表と、選んだ質問の回答・経路。回答経路と100問の記録の画面で共用する。

    index_label: 一覧の先頭列。"time" なら実行日時、"index" なら設問番号。
    """
    from webui.core import load_question

    keyword = st.text_input("絞り込み", placeholder="質問・回答に含まれる語",
                            label_visibility="collapsed")
    if keyword:
        rows = [r for r in rows
                if keyword in (r.get("q") or "") or keyword in (r.get("a") or "")]

    table = [{
        "#": fmt_time(r.get("t")) if index_label == "time" else r["i"],
        "質問": r.get("q") or "",
        "回答": r.get("a") or "—",
    } for r in rows]
    picked = st.dataframe(table, hide_index=True, width="stretch",
                          height=min(400, 38 + 35 * len(table)),
                          on_select="rerun", selection_mode="single-row",
                          # 行番号で選択を持つので、一覧が変わったら選択を捨てる
                          key=f"table-{group.key}-{keyword}",
                          column_config={"質問": st.column_config.TextColumn(width="large"),
                                         "回答": st.column_config.TextColumn(width="medium")})

    sel = [i for i in picked.selection.rows if i < len(rows)]
    if not sel:
        st.caption("行を選ぶと、その質問の回答経路を表示します。")
        return

    row = rows[sel[0]]
    rec = load_question(group, row["run_id"], row["i"])
    if rec is None:
        st.error("記録を読み込めませんでした。")
        return

    st.divider()
    st.markdown(f"#### {rec['q']}")
    if rec.get("t"):
        st.caption(fmt_time(rec["t"]))
    render_answer(rec)
    steps = rec.get("steps") or []
    st.markdown(f"**回答経路**（ツール呼び出し {sum(s.get('k') == 'tool' for s in steps)} 回）")
    if group.kind == "bundled":
        st.caption("同梱の記録は、引数と戻り値を先頭だけに切り詰めた要約版です。")
    render_steps(steps)
