/* Leaderboard page. Plain JS, no dependencies.
 * Every string from leaderboard.json is untrusted: nodes are built with
 * createElement/textContent only, and links are checked before use. */
(function () {
  "use strict";

  var CONFIG = window.SITE_CONFIG || {};
  var DATA_URL = "leaderboard.json";
  var SVG_NS = "http://www.w3.org/2000/svg";
  var TOKENS = { "lt10k": "under 10k", "10k-30k": "10–30k", "gt30k": "over 30k" };

  /* ---------- small helpers ---------- */

  function el(tag, props, children) {
    var node = document.createElement(tag);
    fill(node, props, children);
    return node;
  }
  function sv(tag, props, children) {
    var node = document.createElementNS(SVG_NS, tag);
    fill(node, props, children);
    return node;
  }
  function fill(node, props, children) {
    var k;
    for (k in props || {}) {
      if (k === "class") node.setAttribute("class", props[k]);
      else if (k === "text") node.textContent = props[k];
      else node.setAttribute(k, props[k]);
    }
    (children || []).forEach(function (c) {
      if (c === null || c === undefined) return;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
  }
  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }
  function $(id) { return document.getElementById(id); }

  function httpsUrl(u) {
    try {
      var url = new URL(String(u));
      return url.protocol === "https:" ? url.href : null;
    } catch (e) {
      return null;
    }
  }
  function profileUrl(account) {
    return /^[A-Za-z0-9-]{1,39}$/.test(account) ? "https://github.com/" + account : null;
  }
  function link(href, text, props) {
    var safe = httpsUrl(href);
    if (!safe) return document.createTextNode(text);
    var p = { href: safe, rel: "noopener noreferrer" };
    for (var k in props || {}) p[k] = props[k];
    return el("a", p, [text]);
  }

  var dateFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  var timeFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC", hour12: false });
  var shortDate = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
  function fmtScore(n) { return n.toFixed(1) + "%"; }
  function fmtCi(lo, hi) { return lo.toFixed(1) + "–" + hi.toFixed(1); }

  /* ---------- data validation ---------- */

  function num(v) { return typeof v === "number" && isFinite(v) ? v : null; }
  function str(v, max) { return typeof v === "string" ? v.slice(0, max) : ""; }

  function cleanEntry(e) {
    if (!e || typeof e !== "object") return null;
    var score = num(e.score), lo = num(e.ci_low), hi = num(e.ci_high);
    var t = Date.parse(e.submitted_at);
    var account = str(e.account, 60);
    if (score === null || lo === null || hi === null || isNaN(t) || !account) return null;
    return {
      id: str(e.id, 40), account: account, isBase: e.is_base === true, t: t,
      score: score, lo: lo, hi: hi,
      tokens: Object.prototype.hasOwnProperty.call(TOKENS, e.tokens_bucket) ? e.tokens_bucket : null,
      url: str(e.commit_url, 300)
    };
  }
  function cleanFinal(f) {
    if (!f || typeof f !== "object" || !Array.isArray(f.entries)) return null;
    var rows = [];
    f.entries.forEach(function (e) {
      if (!e || typeof e !== "object") return;
      var score = num(e.score), lo = num(e.ci_low), hi = num(e.ci_high), band = num(e.band);
      var account = str(e.account, 60);
      if (score === null || lo === null || hi === null || band === null || !account) return;
      rows.push({
        account: account, isBase: e.is_base === true, band: band, score: score, lo: lo, hi: hi,
        runs: num(e.runs) || 1,
        source: e.picks_source === "explicit" || e.picks_source === "default" ? e.picks_source : null,
        picks: (Array.isArray(e.picks) ? e.picks : []).slice(0, 2).map(function (p) {
          return { id: str(p && p.id, 40), url: str(p && p.commit_url, 300) };
        })
      });
    });
    var t = Date.parse(f.revealed_at);
    return { rows: rows, revealed: isNaN(t) ? null : t };
  }
  function cleanData(raw) {
    if (!raw || typeof raw !== "object" || !Array.isArray(raw.entries)) throw new Error("unexpected format");
    var g = Date.parse(raw.generated_at);
    return {
      generated: isNaN(g) ? null : g,
      entries: raw.entries.map(cleanEntry).filter(Boolean),
      final: raw.final ? cleanFinal(raw.final) : null
    };
  }

  /* ---------- ranking ---------- */

  function byScore(a, b) { return b.score - a.score || a.t - b.t; }

  /* Rank bands: rows whose interval overlaps the band leader's interval share
   * that band; the first row that does not starts the next band. */
  function assignBands(sorted) {
    var band = 0, leaderLo = Infinity;
    sorted.forEach(function (r) {
      if (band === 0 || r.hi < leaderLo) { band += 1; leaderLo = r.lo; }
      r.band = band;
    });
  }

  function bestPerAccount(entries) {
    var best = {}, counts = {};
    entries.forEach(function (e) {
      if (e.isBase) return;
      counts[e.account] = (counts[e.account] || 0) + 1;
      if (!best[e.account] || byScore(e, best[e.account]) < 0) best[e.account] = e;
    });
    var rows = Object.keys(best).map(function (a) {
      var r = Object.create(best[a]);
      r.count = counts[a];
      return r;
    });
    entries.filter(function (e) { return e.isBase; }).forEach(function (e) { rows.push(e); });
    return rows;
  }

  /* ---------- tables ---------- */

  function buildTable(label, headers, rows, makeCells, rowClass) {
    var head = el("tr", { role: "row" }, headers.map(function (h) {
      return el("th", { scope: "col", role: "columnheader", text: h });
    }));
    var body = el("tbody", { role: "rowgroup" }, rows.map(function (r, i) {
      var cells = makeCells(r, i);
      var tr = el("tr", { role: "row", "class": rowClass(r) }, cells.map(function (c, j) {
        return el("td", { role: "cell", "data-label": headers[j], "class": c.cls }, [el("div", { "class": "v" }, c.nodes)]);
      }));
      return tr;
    }));
    var table = el("table", { "class": "board", role: "table" }, [
      el("caption", { "class": "sr-only", text: label }),
      el("thead", { role: "rowgroup" }, [head]),
      body
    ]);
    return el("div", { "class": "table-wrap" }, [table]);
  }

  function accountNodes(r, extra) {
    var nodes;
    if (r.isBase) {
      nodes = ["Base solution", el("span", { "class": "badge", text: "Base" }),
        el("span", { "class": "sub", text: "Vanilla Inkling 2 Mini with the CaP-X harness" })];
    } else {
      var p = profileUrl(r.account);
      nodes = [p ? link(p, r.account) : r.account];
      if (extra) nodes.push(el("span", { "class": "sub", text: extra }));
    }
    return nodes;
  }
  function scoreNodes(r) {
    return [el("span", { "class": "num", text: fmtScore(r.score) }),
      el("span", { "class": "ci", text: "(" + fmtCi(r.lo, r.hi) + ")" })];
  }

  function renderStandings(data, showAll) {
    var host = $("standings-table");
    clear(host);
    var rows = (showAll ? data.entries.slice() : bestPerAccount(data.entries)).sort(byScore);
    if (!rows.length) {
      host.appendChild(el("p", { "class": "empty", text: "No submissions have been scored yet. The base solution appears here first." }));
      return;
    }
    assignBands(rows);
    host.appendChild(buildTable(
      showAll ? "All public submissions by public score" : "Best public submission per account by public score",
      ["Band", "Account", "Public score (95% interval)", "Tokens per episode", "Time (UTC)", "Submission"],
      rows,
      function (r) {
        var d = new Date(r.t);
        var extra = !showAll && r.count > 1 ? r.count + " submissions" : null;
        return [
          { cls: "cell-band", nodes: [String(r.band)] },
          { cls: "cell-account", nodes: accountNodes(r, extra) },
          { cls: "cell-score", nodes: scoreNodes(r) },
          { cls: "cell-tokens", nodes: [r.tokens ? TOKENS[r.tokens] : "–"] },
          { cls: "cell-date", nodes: [el("time", { datetime: d.toISOString(), text: timeFmt.format(d) })] },
          { cls: "cell-commit", nodes: [r.url ? link(r.url, r.id || "commit", { "aria-label": "Commit " + (r.id || "") + " by " + r.account }) : "–"] }
        ];
      },
      function (r) { return r.isBase ? "is-base" : ""; }
    ));
  }

  function renderFinal(final) {
    var section = $("final");
    var host = $("final-table");
    clear(host);
    if (!final || !final.rows.length) { section.hidden = true; return; }
    section.hidden = false;
    var rows = final.rows.slice().sort(function (a, b) { return a.band - b.band || b.score - a.score; });
    var meta = $("final-meta");
    meta.textContent = final.revealed ? "Revealed " + dateFmt.format(new Date(final.revealed)) + ". " : "";
    host.appendChild(buildTable(
      "Final ranking by private score",
      ["Band", "Account", "Private score (95% interval)", "Runs averaged", "Chosen submissions"],
      rows,
      function (r) {
        var picks = r.picks.map(function (p, i) {
          return el("span", {}, [i ? ", " : "", p.url ? link(p.url, p.id || "commit", { "aria-label": "Commit " + p.id + " by " + r.account }) : p.id]);
        });
        if (r.source === "default") picks.push(el("span", { "class": "sub", text: "most recent two (no picks made)" }));
        return [
          { cls: "cell-band", nodes: [String(r.band)] },
          { cls: "cell-account", nodes: accountNodes(r, null) },
          { cls: "cell-score", nodes: scoreNodes(r) },
          { cls: "cell-runs", nodes: [r.runs === 3 ? "3 runs (mean)" : "1 run"] },
          { cls: "cell-commit", nodes: picks.length ? picks : ["–"] }
        ];
      },
      function (r) { return r.isBase ? "is-base" : ""; }
    ));
  }

  /* ---------- progress chart ---------- */

  function chartModel(entries) {
    var base = entries.filter(function (e) { return e.isBase; })[0] || null;
    var subs = entries.filter(function (e) { return !e.isBase; }).sort(function (a, b) { return a.t - b.t; });
    var best = -Infinity;
    subs.forEach(function (e) {
      e.record = e.score > best;
      if (e.record) best = e.score;
      e.best = best;
    });
    return { base: base, subs: subs };
  }

  function niceTicks(max) {
    var step = max <= 40 ? 10 : max <= 100 ? 20 : 25;
    var t = [];
    for (var v = 0; v <= max; v += step) t.push(v);
    return t;
  }
  var DAY = 86400000;
  function timeTicks(t0, t1, maxCount) {
    var steps = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365];
    var span = (t1 - t0) / DAY, step = steps[steps.length - 1];
    for (var i = 0; i < steps.length; i++) { if (span / steps[i] <= maxCount) { step = steps[i]; break; } }
    var start = Math.floor(t0 / DAY) * DAY, ticks = [];
    for (var t = start; t <= t1; t += step * DAY) if (t >= t0) ticks.push(t);
    return ticks;
  }

  function drawChart(host, model, width) {
    clear(host);
    var narrow = width < 480;
    var H = narrow ? 240 : 300;
    var m = { l: 40, r: narrow ? 12 : 16, t: 16, b: 28 };
    var all = model.subs.concat(model.base ? [model.base] : []);
    var maxScore = Math.max.apply(null, all.map(function (e) { return e.score; }));
    var yMax = Math.min(100, Math.max(20, Math.ceil((maxScore + 5) / 10) * 10));
    var t0 = model.subs[0].t, t1 = model.subs[model.subs.length - 1].t;
    if (t1 - t0 < DAY) { t0 -= DAY / 2; t1 += DAY / 2; }
    var pad = (t1 - t0) * 0.03;
    t0 -= pad; t1 += pad;
    function X(t) { return m.l + (t - t0) / (t1 - t0) * (width - m.l - m.r); }
    function Y(v) { return H - m.b - v / yMax * (H - m.t - m.b); }

    var svg = sv("svg", { viewBox: "0 0 " + width + " " + H, width: width, height: H, role: "img",
      "aria-label": "Line chart of the best public score over time. The latest best is " + fmtScore(model.subs[model.subs.length - 1].best) +
        (model.base ? "; the base solution scored " + fmtScore(model.base.score) : "") + ". The table above lists every value." });

    niceTicks(yMax).forEach(function (v) {
      svg.appendChild(sv("line", { "class": v === 0 ? "axis" : "grid", x1: m.l, x2: width - m.r, y1: Y(v), y2: Y(v) }));
      svg.appendChild(sv("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end", text: v + "%" }));
    });
    timeTicks(t0, t1, Math.max(2, Math.floor((width - m.l - m.r) / 80))).forEach(function (t) {
      svg.appendChild(sv("text", { x: X(t), y: H - 8, "text-anchor": "middle", text: shortDate.format(new Date(t)) }));
    });

    if (model.base) {
      var by = Y(model.base.score);
      svg.appendChild(sv("line", { "class": "base", x1: m.l, x2: width - m.r, y1: by, y2: by }));
      svg.appendChild(sv("text", { "class": "lbl-base", x: width - m.r, y: by + 16, "text-anchor": "end",
        text: "Base solution " + fmtScore(model.base.score) }));
    }

    /* best-so-far as a step line, extended to the last submission */
    var d = "", prev = null;
    model.subs.forEach(function (e) {
      if (!e.record) return;
      d += (prev === null ? "M" : "H" + X(e.t).toFixed(1) + "V") + (prev === null ? X(e.t).toFixed(1) + " " : "") + Y(e.best).toFixed(1);
      prev = e;
    });
    d += "H" + X(model.subs[model.subs.length - 1].t).toFixed(1);
    svg.appendChild(sv("path", { "class": "best", d: d }));

    var pts = [];
    model.subs.forEach(function (e) {
      var cx = X(e.t), cy = Y(e.score);
      svg.appendChild(sv("circle", { "class": e.record ? "pt record" : "pt", cx: cx, cy: cy, r: 4 }));
      pts.push({ e: e, x: cx, y: cy });
    });
    var last = model.subs[model.subs.length - 1];
    svg.appendChild(sv("text", { "class": "lbl-best", x: Math.min(X(last.t), width - m.r), y: Y(last.best) - 10, "text-anchor": "end",
      text: "Best " + fmtScore(last.best) }));

    var hot = sv("circle", { "class": "hot", r: 7, visibility: "hidden" });
    svg.appendChild(hot);
    var tip = el("div", { "class": "tip", hidden: "" });

    function show(ev) {
      var box = svg.getBoundingClientRect();
      var px = (ev.clientX - box.left) * (width / box.width), py = (ev.clientY - box.top) * (H / box.height);
      var near = null, dist = 24 * 24;
      pts.forEach(function (p) {
        var dd = (p.x - px) * (p.x - px) + (p.y - py) * (p.y - py);
        if (dd < dist) { dist = dd; near = p; }
      });
      if (!near) { hide(); return; }
      hot.setAttribute("cx", near.x); hot.setAttribute("cy", near.y); hot.setAttribute("visibility", "visible");
      var e = near.e;
      tip.textContent = e.account + ": " + fmtScore(e.score) + " (" + fmtCi(e.lo, e.hi) + "), " + dateFmt.format(new Date(e.t));
      tip.hidden = false;
      var tx = Math.min(Math.max(near.x, 100), width - 100);
      tip.style.left = tx + "px"; tip.style.top = (near.y - 12) + "px";
    }
    function hide() { tip.hidden = true; hot.setAttribute("visibility", "hidden"); }
    svg.addEventListener("pointermove", show);
    svg.addEventListener("pointerdown", show);
    svg.addEventListener("pointerleave", hide);

    host.appendChild(svg);
    host.appendChild(tip);
  }

  function renderChart(entries) {
    var section = $("progress");
    var host = $("chart-host");
    var model = chartModel(entries);
    if (!model.subs.length) { section.hidden = true; return; }
    section.hidden = false;
    var lastW = 0;
    function draw() {
      var w = Math.round(host.clientWidth);
      if (w < 200 || w === lastW) return;
      lastW = w;
      drawChart(host, model, w);
    }
    draw();
    if (window.ResizeObserver) new ResizeObserver(draw).observe(host);
    else window.addEventListener("resize", draw);
  }

  /* ---------- join block ---------- */

  function promptText() {
    var repo = httpsUrl(CONFIG.repoUrl) || "https://github.com/robojongit/thinkopenrobots";
    return "I want to compete in the Inkling Robotics Leaderboard.\n" +
      "Clone " + repo + ", read its AGENTS.md\n" +
      "and follow it: set me up, reproduce the base score, then keep\n" +
      "improving agent/ and ask me before each submission.";
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var ta = el("textarea", { "aria-hidden": "true", readonly: "", style: "position:fixed;top:0;left:0;opacity:0" });
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error("copy failed"));
    });
  }

  function renderJoin() {
    var pre = $("prompt");
    if (!pre) return;
    var text = promptText();
    pre.textContent = text;
    var status = $("copy-status"), btn = $("copy-btn"), timer;
    function say(msg) {
      status.textContent = msg;
      clearTimeout(timer);
      timer = setTimeout(function () { status.textContent = ""; }, 3000);
    }
    btn.addEventListener("click", function () {
      copyText(text).then(function () { say("Copied to the clipboard."); },
        function () {
          var r = document.createRange();
          r.selectNodeContents(pre);
          var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
          say("Copy failed. The prompt is selected: press Ctrl+C or Cmd+C.");
        });
    });
  }

  /* ---------- loading ---------- */

  function sourceUrl() {
    var q = new URLSearchParams(location.search).get("sample");
    if (q === "final") return { url: "leaderboard.sample-final.json", sample: true };
    if (q) return { url: "leaderboard.sample.json", sample: true };
    return { url: DATA_URL + "?_=" + Math.floor(Date.now() / 60000), sample: false };
  }

  function setStatus(kind, msg, retry) {
    var s = $("status");
    clear(s);
    if (!kind) { s.hidden = true; return; }
    s.hidden = false;
    s.className = "notice" + (kind === "error" ? " error" : "");
    s.appendChild(document.createTextNode(msg));
    if (retry) s.appendChild(el("button", { type: "button", text: "Try again" }));
    if (retry) s.lastChild.addEventListener("click", retry);
  }

  function showData(data, sample) {
    var showAll = $("show-all");
    $("data-sections").hidden = false;
    renderStandings(data, showAll.checked);
    showAll.onchange = function () { renderStandings(data, showAll.checked); };
    renderFinal(data.final);
    renderChart(data.entries);
    setStatus(sample ? "notice" : null, sample ? "Showing sample data, not real results." : "");
  }

  function load() {
    var src = sourceUrl();
    $("data-sections").hidden = true;
    setStatus("notice", "Loading leaderboard…");
    fetch(src.url, { cache: "no-store" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (raw) {
        showData(cleanData(raw), src.sample);
      })
      .catch(function (err) {
        $("data-sections").hidden = true;
        setStatus("error", "The leaderboard could not be loaded (" + (err && err.message ? err.message : "unknown error") + ").", load);
      });
  }

  /* ---------- boot ---------- */

  function boot() {
    var repo = httpsUrl(CONFIG.repoUrl);
    Array.prototype.forEach.call(document.querySelectorAll("a[data-repo]"), function (a) {
      if (repo) a.setAttribute("href", repo + (a.getAttribute("data-repo") || ""));
    });
    if ($("standings-table")) { renderJoin(); load(); }
  }
  boot();
})();
