/* Drive Trace — 静的データ(window.RAG_RUN / window.RAG_CORPUS)だけで動く。
   ビルド工程なし。index.html を直接開いても、http.server 越しでも同じ。 */
(function () {
  "use strict";

  var RUN = window.RAG_RUN || { questions: [], run: {} };
  var QS = RUN.questions || [];
  var FILES = (window.RAG_CORPUS || {}).files || [];
  var EXPLAIN = window.RAG_EXPLAIN || {};   // 設問ごとの解説（portfolio/explain/）

  var TOOLS = ["search", "grep", "read", "read_image", "ask_image",
               "run_python", "diff", "decrypt", "submit_answer"];

  var TOOL_INFO = {
    search:        ["ハイブリッド全文検索", "--t-search"],
    grep:          ["抽出済み全文への正規表現", "--t-grep"],
    read:          ["抽出済み内容を読む", "--t-read"],
    read_image:    ["画像を実際に見る", "--t-read_image"],
    ask_image:     ["画像へ質問して確かめる", "--t-ask_image"],
    run_python:    ["コードで走査・集計する", "--t-run_python"],
    diff:          ["2ファイルの差分", "--t-diff"],
    decrypt:       ["暗号化Officeを復号", "--t-decrypt"],
    submit_answer: ["最終回答の提出", "--t-submit"]
  };

  var ROUTE_ORDER = ["コード走査", "画像読取", "図版照会", "復号", "版比較",
                     "直読", "全文検索", "意味検索", "未探索"];

  var LIFECYCLE = [
    ["00.提案", "提案書・調査資料。見込み金額や当初のスコープの出どころ。"],
    ["01.契約", "契約書。単価・精算方法・検収条件。暗号化されているものがある。"],
    ["02.計画", "スケジュール。ガント期間はセルの塗り色で表現されている。"],
    ["03.データ", "学習データとカラム説明。行数が多く、コードで読む前提。"],
    ["04.分析", "分析プロジェクト一式。notebook・src・出力図・実験リーダーボード。"],
    ["05.会議", "会議録と報告資料。決定事項と申し送りの時系列。"],
    ["06.報告書", "最終報告。採用モデルと成果。old版と並んでいることがある。"]
  ];

  // ---------- helpers ----------

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function toolColor(name) {
    var info = TOOL_INFO[name];
    return info ? "var(" + info[1] + ")" : "var(--muted)";
  }

  function toolChip(name, count) {
    var c = el("span", "chip");
    c.style.color = toolColor(name);
    c.appendChild(el("span", "dot"));
    c.appendChild(document.createTextNode(count ? name + " ×" + count : name));
    return c;
  }

  function fmtSec(s) {
    if (s == null) return "";
    if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + "s";
    return Math.floor(s / 60) + "m" + String(Math.round(s % 60)).padStart(2, "0") + "s";
  }

  function n(x) { return (x || 0).toLocaleString("en-US"); }

  function statTile(value, unit, key) {
    var d = el("div", "stat");
    var v = el("span", "stat__n");
    v.appendChild(document.createTextNode(value));
    if (unit) {
      var u = el("small");
      u.textContent = " " + unit;
      v.appendChild(u);
    }
    d.appendChild(v);
    d.appendChild(el("span", "stat__k", key));
    return d;
  }

  function barRow(key, value, max, color, suffix) {
    var row = el("div", "bar");
    row.style.color = color || "var(--accent)";
    row.appendChild(el("span", "bar__k", key));
    var track = el("div", "bar__t");
    var fill = el("div", "bar__f");
    fill.style.width = (max ? (value / max) * 100 : 0).toFixed(1) + "%";
    track.appendChild(fill);
    row.appendChild(track);
    row.appendChild(el("span", "bar__v", n(value) + (suffix || "")));
    return row;
  }

  function fillBars(host, entries, colorOf, suffix) {
    host.textContent = "";
    var max = entries.reduce(function (m, e) { return Math.max(m, e[1]); }, 0);
    entries.forEach(function (e) {
      host.appendChild(barRow(e[0], e[1], max, colorOf ? colorOf(e[0]) : null, suffix));
    });
  }

  function countBy(list, fn) {
    var m = Object.create(null);
    list.forEach(function (x) {
      var k = fn(x);
      if (k == null) return;
      m[k] = (m[k] || 0) + 1;
    });
    return m;
  }

  // ---------- view switching ----------

  var views = {};
  ["competition", "impl", "corpus", "routes", "tech"].forEach(function (v) {
    views[v] = document.getElementById("view-" + v);
  });

  function show(name) {
    Object.keys(views).forEach(function (v) { views[v].hidden = v !== name; });
    document.querySelectorAll("#nav button").forEach(function (b) {
      b.setAttribute("aria-selected", String(b.dataset.view === name));
    });
    if (location.hash.slice(1).split("/")[0] !== name) {
      history.replaceState(null, "", name === "routes" ? "#routes/" + qid(selected) : "#" + name);
    }
  }

  document.getElementById("nav").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-view]");
    if (b) { show(b.dataset.view); window.scrollTo(0, 0); }
  });

  // ---------- theme ----------

  var themeBtn = document.getElementById("theme-btn");
  function readTheme() {
    try { return localStorage.getItem("dt-theme"); } catch (e) { return null; }
  }
  var stored = readTheme();
  if (stored === "dark" || stored === "light") {
    document.documentElement.setAttribute("data-theme", stored);
  }
  themeBtn.addEventListener("click", function () {
    var cur = document.documentElement.getAttribute("data-theme");
    var isDark = cur ? cur === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    var next = isDark ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("dt-theme", next); } catch (e) { /* 保存不可でも動く */ }
  });

  // ================= 集計（複数のページで使う） =================

  var toolCalls = Object.create(null);
  QS.forEach(function (q) {
    q.steps.forEach(function (s) {
      if (s.k === "tool") toolCalls[s.name] = (toolCalls[s.name] || 0) + 1;
    });
  });

  // ================= CORPUS =================

  (function corpus() {
    var projects = {};
    FILES.forEach(function (f) { if (f.project) projects[f.project] = 1; });
    var projectNames = Object.keys(projects).sort();
    var cats = LIFECYCLE.map(function (c) { return c[0]; });
    var withFlags = FILES.filter(function (f) { return f.flags.length; });
    var assets = FILES.reduce(function (a, f) { return a + f.assets; }, 0);
    var peers = FILES.filter(function (f) { return f.peers.length; });

    var host = document.getElementById("corpus-stats");
    [
      statTile(n(projectNames.length), "案件", "プロジェクトフォルダ"),
      statTile(n(FILES.length), "ファイル", "抽出対象"),
      statTile(String(new Set(FILES.map(function (f) { return f.ft; })).size), "種類", "拡張子"),
      statTile(n(assets), "点", "本文から切り出した画像"),
      statTile(n(peers.length), "ファイル", "版違いとして紐づいた")
    ].forEach(function (t) { host.appendChild(t); });

    // lifecycle spine
    var spine = document.getElementById("lifecycle");
    LIFECYCLE.forEach(function (c) {
      var count = FILES.filter(function (f) { return f.category === c[0]; }).length;
      var row = el("div", "spine__row");
      row.appendChild(el("span", "spine__n", c[0].slice(0, 2)));
      var mid = el("div");
      mid.appendChild(el("div", "spine__name", c[0].slice(3)));
      mid.appendChild(el("div", "spine__what", c[1]));
      row.appendChild(mid);
      row.appendChild(el("span", "spine__count", n(count) + " ファイル"));
      spine.appendChild(row);
    });

    // internal docs
    var internal = FILES.filter(function (f) { return !f.project; });
    var tbl = document.getElementById("internal-docs");
    var thead = el("thead");
    var htr = el("tr");
    ["文書", "形式", "エージェントにとっての役割"].forEach(function (h) {
      htr.appendChild(el("th", null, h));
    });
    thead.appendChild(htr);
    tbl.appendChild(thead);
    var ROLE = {
      "社内用語集": "質問文の略称を通常表現へ展開する。検索も条項の適用判断も、展開後の意味で行う。",
      "データアステル社内規定_パスワード導出規則": "暗号化された契約書のパスワード候補を組み立てる根拠。",
      "データアステル社内管理_決裁基準": "金額に応じた承認者を判定する。契約金額と突き合わせて使う。",
      "座席表": "氏名から内線番号・所属へ引く。図として描かれているので位置関係の読解が要る。"
    };
    var tb2 = el("tbody");
    internal.forEach(function (f) {
      var base = f.path.split("/").pop().replace(/\.[^.]+$/, "");
      var tr = el("tr");
      tr.appendChild(el("td", null, base));
      tr.appendChild(el("td", null, f.ft));
      var td = el("td", null, ROLE[base] || f.desc);
      td.style.cssText = "text-align:left;font-family:var(--sans);white-space:normal;min-width:260px";
      tr.appendChild(td);
      tb2.appendChild(tr);
    });
    tbl.appendChild(tb2);

    // matrix
    var mx = document.getElementById("corpus-matrix");
    var mh = el("thead"), mhr = el("tr");
    mhr.appendChild(el("th", null, "案件"));
    cats.forEach(function (c) { mhr.appendChild(el("th", null, c)); });
    mhr.appendChild(el("th", null, "計"));
    mh.appendChild(mhr);
    mx.appendChild(mh);
    var mb = el("tbody");
    var colTotals = cats.map(function () { return 0; });
    projectNames.forEach(function (p) {
      var tr = el("tr");
      tr.appendChild(el("td", null, p));
      var rowTotal = 0;
      cats.forEach(function (c, i) {
        var v = FILES.filter(function (f) {
          return f.project === p && f.category === c;
        }).length;
        rowTotal += v; colTotals[i] += v;
        tr.appendChild(el("td", v ? null : "zero", String(v)));
      });
      tr.appendChild(el("td", null, String(rowTotal)));
      mb.appendChild(tr);
    });
    mx.appendChild(mb);
    var mf = el("tfoot"), ftr = el("tr");
    ftr.appendChild(el("td", null, "計"));
    colTotals.forEach(function (v) { ftr.appendChild(el("td", null, String(v))); });
    ftr.appendChild(el("td", null, String(colTotals.reduce(function (a, b) { return a + b; }, 0))));
    mf.appendChild(ftr);
    mx.appendChild(mf);

    // flag buckets
    var BUCKET = [
      ["埋め込み画像がある", /^画像\d+$/, "画像を本文から切り出し、前処理でOCR済み。元の画像は read_image で見られる。"],
      ["スキャンページを含む", /^スキャン/, "文字情報を持たないPDFのページ。前処理でOCRし、本文に統合してある。"],
      ["行数が多い", /大規模/, "全部を会話に載せると溢れるため、run_python でコードから読む前提。"],
      ["グラフを持つ", /^グラフ\d+$/, "グラフに保存されている値から、元の表を復元してある。"],
      ["ピボットを持つ", /^ピボット\d+$/, "ピボットの行・列・フィルタ・集計方法を、定義として本文に書き出してある。"],
      ["EMF埋め込み表", /^EMF表\d+$/, "画像として貼り込まれた表。ファイルの中身を直接解析してセル値を取り出してある。"],
      ["暗号化されている", /暗号化/, "社内規定の規則からパスワードを組み立て、decrypt で開く。"]
    ];
    fillBars(document.getElementById("flag-bars"),
      BUCKET.map(function (b) {
        return [b[0], FILES.filter(function (f) {
          return f.flags.some(function (fl) { return b[1].test(fl); });
        }).length];
      }), null, " ファイル");
    // 注記ごとの意味と扱いは、1段落に続けず表にする
    var wrap = el("div", "scroll-x");
    var tbl = el("table", "matrix matrix--text");
    var th = el("thead"), htr = el("tr");
    ["注記", "意味と扱い"].forEach(function (h) { htr.appendChild(el("th", null, h)); });
    th.appendChild(htr);
    tbl.appendChild(th);
    var tb = el("tbody");
    BUCKET.forEach(function (b) {
      var tr = el("tr");
      tr.appendChild(el("td", null, b[0]));
      tr.appendChild(el("td", null, b[2]));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb);
    wrap.appendChild(tbl);
    wrap.style.marginTop = "18px";
    document.getElementById("flag-bars").after(wrap);

    // tree
    buildTree(FILES);
  }());

  function buildTree(files) {
    var root = { dirs: Object.create(null), files: [], n: 0 };
    files.forEach(function (f) {
      var parts = f.path.split("/");
      var node = root;
      for (var i = 0; i < parts.length - 1; i++) {
        node.n++;
        if (!node.dirs[parts[i]]) {
          node.dirs[parts[i]] = { dirs: Object.create(null), files: [], n: 0 };
        }
        node = node.dirs[parts[i]];
      }
      node.n++;
      node.files.push(f);
    });

    var host = document.getElementById("tree");

    function flagsOf(f) {
      var out = [];
      f.flags.forEach(function (fl) {
        out.push(["flag" + (/暗号化|EMF|スキャン/.test(fl) ? " hot" : ""), fl]);
      });
      if (f.peers.length) out.push(["flag ver", "版違い" + f.peers.length]);
      return out;
    }

    function render(node, depth) {
      var ul = el("ul");
      Object.keys(node.dirs).sort().forEach(function (name) {
        var child = node.dirs[name];
        var li = el("li");
        if (depth < 1) li.className = "open";
        var d = el("span", "dir");
        d.textContent = name + "/";
        var c = el("span", "n");
        c.textContent = child.n;
        d.appendChild(c);
        d.tabIndex = 0;
        li.appendChild(d);
        li.appendChild(render(child, depth + 1));
        ul.appendChild(li);
      });
      node.files.forEach(function (f) {
        var li = el("li");
        var s = el("span", "file");
        s.textContent = f.path.split("/").pop();
        var ft = el("span", "ft");
        ft.textContent = "  " + (f.desc || f.ft);
        s.appendChild(ft);
        flagsOf(f).forEach(function (fl) {
          s.appendChild(el("span", fl[0], fl[1]));
        });
        li.appendChild(s);
        li.dataset.path = f.path.toLowerCase();
        ul.appendChild(li);
      });
      return ul;
    }

    host.appendChild(render(root, 0));

    host.addEventListener("click", function (e) {
      var d = e.target.closest(".dir");
      if (d) d.parentElement.classList.toggle("open");
    });
    host.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" && e.key !== " ") return;
      var d = e.target.closest(".dir");
      if (d) { e.preventDefault(); d.parentElement.classList.toggle("open"); }
    });

    document.getElementById("tree-expand").addEventListener("click", function () {
      host.querySelectorAll("li").forEach(function (li) {
        if (li.querySelector(":scope > .dir")) li.classList.add("open");
      });
    });
    document.getElementById("tree-collapse").addEventListener("click", function () {
      host.querySelectorAll("li").forEach(function (li) { li.classList.remove("open"); });
      host.querySelectorAll(":scope > ul > li").forEach(function (li) {
        li.classList.add("open");
      });
    });

    var q = document.getElementById("tree-q");
    q.addEventListener("input", function () {
      var term = q.value.trim().toLowerCase();
      host.querySelectorAll("li[data-path]").forEach(function (li) {
        li.hidden = term !== "" && li.dataset.path.indexOf(term) === -1;
      });
      host.querySelectorAll("li").forEach(function (li) {
        if (!li.querySelector(":scope > .dir")) return;
        var anyVisible = li.querySelectorAll("li[data-path]:not([hidden])").length > 0;
        li.hidden = term !== "" && !anyVisible;
        if (term !== "" && anyVisible) li.classList.add("open");
      });
    });
  }

  // ================= ROUTES =================

  var state = { text: "", route: "", project: "", tool: "" };
  var selected = 0;              // 右に表示している設問の番号

  (function routes() {
    var routeCounts = countBy(QS, function (q) { return q.route; });
    fillBars(document.getElementById("route-bars"),
      ROUTE_ORDER.filter(function (r) { return routeCounts[r]; })
        .map(function (r) { return [r, routeCounts[r]]; })
        .sort(function (a, b) { return b[1] - a[1]; }),
      null, " 問");

    // filters
    var routeSel = document.getElementById("q-route");
    routeSel.appendChild(new Option("すべての経路", ""));
    ROUTE_ORDER.filter(function (r) { return routeCounts[r]; }).forEach(function (r) {
      routeSel.appendChild(new Option(r + "（" + routeCounts[r] + "）", r));
    });

    var projSel = document.getElementById("q-project");
    projSel.appendChild(new Option("すべての案件", ""));
    var projNames = {};
    FILES.forEach(function (f) { if (f.project) projNames[f.project] = 1; });
    Object.keys(projNames).sort().forEach(function (p) {
      projSel.appendChild(new Option(p, p));
    });

    var legend = document.getElementById("tool-legend-filter");
    TOOLS.filter(function (t) { return t !== "submit_answer"; }).forEach(function (t) {
      var chip = toolChip(t);
      chip.setAttribute("role", "button");
      chip.tabIndex = 0;
      chip.setAttribute("aria-pressed", "true");
      chip.dataset.tool = t;
      legend.appendChild(chip);
    });
    function toggleTool(c) {
      state.tool = state.tool === c.dataset.tool ? "" : c.dataset.tool;
      syncLegend();
      renderList();
    }
    legend.addEventListener("click", function (e) {
      var c = e.target.closest("[data-tool]");
      if (c) toggleTool(c);
    });
    legend.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" && e.key !== " ") return;
      var c = e.target.closest("[data-tool]");
      if (c) { e.preventDefault(); toggleTool(c); }
    });
    function syncLegend() {
      legend.querySelectorAll("[data-tool]").forEach(function (c) {
        c.setAttribute("aria-pressed",
          String(!state.tool || state.tool === c.dataset.tool));
      });
    }

    document.getElementById("q-search").addEventListener("input", function (e) {
      state.text = e.target.value.trim().toLowerCase();
      renderList();
    });
    routeSel.addEventListener("change", function (e) {
      state.route = e.target.value; renderList();
    });
    projSel.addEventListener("change", function (e) {
      state.project = e.target.value; renderList();
    });

    renderList();
  }());

  function matches(q) {
    if (state.route && q.route !== state.route) return false;
    if (state.tool && q.tools.indexOf(state.tool) < 0) return false;
    if (state.project) {
      var hit = q.ev.some(function (e) { return e.indexOf(state.project) >= 0; }) ||
        q.q.indexOf(state.project) >= 0 ||
        q.steps.some(function (s) {
          return s.k === "tool" && JSON.stringify(s.args || {}).indexOf(state.project) >= 0;
        });
      if (!hit) return false;
    }
    if (state.text) {
      var hay = (q.q + " " + q.a + " " + q.ev.join(" ")).toLowerCase();
      if (hay.indexOf(state.text) < 0) return false;
    }
    return true;
  }

  function qid(i) { return "Q" + String(i).padStart(2, "0"); }

  // 左の一覧を描き直す。選択中の設問が絞り込みで外れたら、先頭を選び直す
  function renderList() {
    var host = document.getElementById("qlist");
    host.textContent = "";
    var shown = QS.filter(matches);
    document.getElementById("q-count").textContent =
      shown.length + " / " + QS.length + " 問";
    if (!shown.length) {
      var p = el("p", "head-note", "条件に一致する設問はない。");
      p.style.padding = "12px 8px";
      host.appendChild(p);
      document.getElementById("qdetail").textContent = "";
      return;
    }
    if (!shown.some(function (q) { return q.i === selected; })) selected = shown[0].i;
    shown.forEach(function (q) { host.appendChild(navItem(q)); });
    renderDetail();
  }

  function navItem(q) {
    var b = el("button", "qnav");
    b.type = "button";
    b.dataset.i = q.i;
    b.setAttribute("aria-current", String(q.i === selected));
    b.appendChild(el("span", "qnav__i", qid(q.i)));
    b.appendChild(el("span", "qnav__t", q.q));
    b.appendChild(el("span", "qnav__m", q.route));
    b.addEventListener("click", function () { select(q.i); });
    return b;
  }

  function select(i, keepScroll) {
    selected = i;
    document.querySelectorAll("#qlist .qnav").forEach(function (b) {
      b.setAttribute("aria-current", String(Number(b.dataset.i) === i));
    });
    renderDetail();
    history.replaceState(null, "", "#routes/" + qid(i));
    if (!keepScroll) {
      var d = document.getElementById("qdetail");
      if (d.getBoundingClientRect().top < 60) d.scrollIntoView({ block: "start" });
    }
  }

  // 右の詳細。前後の移動は、いま一覧に出ている設問の中で行う
  function renderDetail() {
    var host = document.getElementById("qdetail");
    host.textContent = "";
    var q = QS.find(function (x) { return x.i === selected; });
    if (!q) return;
    var shown = QS.filter(matches);
    var pos = shown.findIndex(function (x) { return x.i === q.i; });

    var nav = el("div", "qd-nav");
    nav.appendChild(el("span", "qidx", qid(q.i)));
    [["← 前の設問", pos - 1], ["次の設問 →", pos + 1]].forEach(function (d) {
      var btn = el("button", "btn", d[0]);
      btn.type = "button";
      var target = shown[d[1]];
      btn.disabled = !target;
      if (target) btn.addEventListener("click", function () { select(target.i, true); });
      nav.appendChild(btn);
    });
    host.appendChild(nav);

    host.appendChild(el("h3", "qd-q", q.q));
    var ans = el("div", "qd-a");
    ans.appendChild(el("span", null, "回答"));
    ans.appendChild(document.createTextNode(q.a || "（中断）"));
    host.appendChild(ans);

    var meta = el("div", "qmeta");
    var routeChip = el("span", "chip", q.route);
    routeChip.style.color = q.stop === "submitted" ? "var(--ink)" : "var(--halt)";
    meta.appendChild(routeChip);
    var t = el("span", "chip", q.turns + "ターン / " + fmtSec(q.sec));
    t.style.color = "var(--faint)";
    meta.appendChild(t);
    host.appendChild(meta);

    if (EXPLAIN[q.i]) host.appendChild(explainBlock(EXPLAIN[q.i]));

    var body = el("div", "qbody");
    if (EXPLAIN[q.i]) body.appendChild(el("h3", "qx-part", "実行の記録"));
    buildBody(body, q);
    host.appendChild(body);
  }

  // 何を問うているか → 実際の資料 → どう判断したか
  function explainBlock(x) {
    var box = el("section", "qx");
    box.appendChild(el("h3", "qx-part", "解説"));

    box.appendChild(el("h4", null, "何を問うているか"));
    if (x.types && x.types.length) {
      var tags = el("div", "qx-types");
      x.types.forEach(function (t) { tags.appendChild(el("span", "qx-type", t)); });
      box.appendChild(tags);
    }
    box.appendChild(el("p", "qx-ask", x.ask));
    if (x.point) {
      var pt = el("p", "qx-point");
      pt.appendChild(el("span", "qx-point__k", "難しさ・意図"));
      pt.appendChild(document.createTextNode(x.point));
      box.appendChild(pt);
    }

    if (x.materials && x.materials.length) {
      box.appendChild(el("h4", null, "実際の資料"));
      var grid = el("div", "qx-mat" + (x.layout === "compare" ? " qx-mat--compare" : ""));
      x.materials.forEach(function (m) {
        var fig = el("figure");
        if (m.kind === "text") {
          fig.appendChild(el("pre", "qx-text", m.text));
        } else {
          var a = el("a");
          a.href = m.src;
          a.target = "_blank";
          a.rel = "noopener";
          var img = el("img");
          img.src = m.src;
          img.alt = m.caption;
          img.loading = "lazy";
          a.appendChild(img);
          fig.appendChild(a);
        }
        var cap = el("figcaption");
        cap.appendChild(el("span", "qx-file", (m.file || m.asset || "").split("/").pop() +
          (m.kind === "slide" ? " スライド" + m.index : m.kind === "page" ? " p" + m.index :
           m.kind === "sheet" ? " シート「" + m.sheet + "」" : "")));
        cap.appendChild(document.createTextNode(m.caption));
        fig.appendChild(cap);
        grid.appendChild(fig);
      });
      box.appendChild(grid);
    }

    box.appendChild(el("h4", null, "どう判断したか"));
    var ol = el("ol", "qx-steps");
    (x.reasoning || []).forEach(function (r) { ol.appendChild(el("li", null, r)); });
    box.appendChild(ol);
    return box;
  }

  function buildBody(body, q) {
    body.appendChild(el("h4", null, "使ったツール"));
    var chips = el("div", "legend");
    q.tools.forEach(function (t) { chips.appendChild(toolChip(t, q.counts[t])); });
    if (!q.tools.length) chips.appendChild(el("span", "head-note", "実行前に中断"));
    body.appendChild(chips);

    if (q.ev.length) {
      body.appendChild(el("h4", null, "根拠として挙げた資料"));
      var ul = el("ul", "evlist");
      q.ev.forEach(function (e) { ul.appendChild(el("li", null, e)); });
      body.appendChild(ul);
    }

    body.appendChild(el("h4", null, "探索経路（" + q.steps.length + " ステップ）"));
    var list = el("ul", "trace");
    q.steps.forEach(function (s) { list.appendChild(stepNode(s)); });
    body.appendChild(list);

    var foot = el("p", "caption");
    foot.textContent = "確信度 " + (q.conf != null ? q.conf : "—") +
      " ／ " + n(q.tok) + " tokens ／ 停止理由 " +
      (q.stop === "submitted" ? "submitted" : "APIクォータ超過で中断");
    body.appendChild(foot);
  }

  function stepNode(s) {
    var li = el("li", "step");

    if (s.k === "error") {
      li.style.color = "var(--halt)";
      li.appendChild(el("div", "step__think", s.text));
      return li;
    }

    if (s.k === "think") {
      var row = el("div", "step__row");
      var label = el("span", "argkey",
        s.calls.length ? "考えてツールを呼ぶ" : "考える");
      row.appendChild(label);
      s.calls.forEach(function (c) { row.appendChild(toolChip(c)); });
      row.appendChild(el("span", "step__t", fmtSec(s.sec)));
      li.appendChild(row);
      if (s.text) li.appendChild(el("div", "step__think", s.text));
      return li;
    }

    li.classList.add("is-tool");
    li.style.color = toolColor(s.name);
    if (s.blocked) li.classList.add("is-gate");

    var r = el("div", "step__row");
    r.appendChild(toolChip(s.name));
    if (s.blocked) {
      var g = el("span", "chip", "差し戻し");
      g.style.color = "var(--gate)";
      r.appendChild(g);
    }
    r.appendChild(el("span", "step__t", fmtSec(s.sec)));
    li.appendChild(r);

    var keys = Object.keys(s.args || {});
    if (keys.length) {
      var pre = el("pre", "pre");
      keys.forEach(function (k, i) {
        var kk = el("span", "argk", k + ": ");
        pre.appendChild(kk);
        pre.appendChild(document.createTextNode(s.args[k] + (i < keys.length - 1 ? "\n" : "")));
      });
      li.appendChild(pre);
    }
    if (s.out) {
      var out = el("pre", "pre" + (s.blocked ? " gate" : ""));
      out.textContent = s.out;
      li.appendChild(out);
    }
    return li;
  }

  // ================= TECH → ROUTES links =================

  document.querySelectorAll("[data-qrefs]").forEach(function (host) {
    host.dataset.qrefs.split(",").forEach(function (idx) {
      var b = el("button", "qref", "Q" + String(idx.trim()).padStart(2, "0"));
      b.type = "button";
      b.addEventListener("click", function () { openQuestion(Number(idx.trim())); });
      host.appendChild(b);
    });
    var lbl = el("span", "argkey", "この工夫が効いた設問 → ");
    host.insertBefore(lbl, host.firstChild);
  });

  function openQuestion(i) {
    state = { text: "", route: "", project: "", tool: "" };
    document.getElementById("q-search").value = "";
    document.getElementById("q-route").value = "";
    document.getElementById("q-project").value = "";
    document.querySelectorAll("#tool-legend-filter [data-tool]").forEach(function (c) {
      c.setAttribute("aria-pressed", "true");
    });
    selected = i;
    renderList();
    show("routes");
    history.replaceState(null, "", "#routes/" + qid(i));
    var nav = document.querySelector('#qlist .qnav[data-i="' + i + '"]');
    if (nav) nav.scrollIntoView({ block: "nearest" });
    document.querySelector(".qpane").scrollIntoView({ block: "start", behavior: "smooth" });
  }

  // ================= IMPLEMENTATION =================

  document.querySelectorAll("[data-tool-count]").forEach(function (n) {
    var c = toolCalls[n.dataset.toolCount] || 0;
    n.textContent = c.toLocaleString("ja-JP") + " 回";
  });

  // ---------- boot ----------

  var hashParts = (location.hash.slice(1) || "competition").split("/");
  var initial = hashParts[0];
  var qm = /^Q(\d+)$/.exec(hashParts[1] || "");
  if (initial === "routes" && qm && QS.some(function (q) { return q.i === Number(qm[1]); })) {
    openQuestion(Number(qm[1]));
  } else {
    show(views[initial] ? initial : "competition");
  }
}());
