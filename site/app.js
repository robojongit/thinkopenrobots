/* Leaderboard page. Plain JS, no dependencies.
 * Every string from leaderboard.json is untrusted: nodes are built with
 * createElement/textContent only, and links are checked before use. */
(function () {
  "use strict";

  var CONFIG = window.SITE_CONFIG || {};
  var DATA_URL = "leaderboard.json";
  var SVG_NS = "http://www.w3.org/2000/svg";
  var TOKENS = { "lt10k": ["under 10k", 1], "10k-30k": ["10–30k", 2], "gt30k": ["over 30k", 3] };
  var LOGIN_RE = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/;
  /* same rule as the schema: ASCII letters, digits, spaces, . _ + - / ( ) and the middle dot */
  var CODING_AGENT = /^[A-Za-z0-9 ._+\-\/()\u00b7]{1,40}$/;

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
  function validLogin(account) { return LOGIN_RE.test(account); }
  function profileUrl(account) {
    return validLogin(account) ? "https://github.com/" + account : null;
  }
  function link(href, text, props) {
    var safe = httpsUrl(href);
    if (!safe) return document.createTextNode(text);
    var p = { href: safe, rel: "noopener noreferrer" };
    for (var k in props || {}) p[k] = props[k];
    return el("a", p, [text]);
  }
  function reveal(node) {
    if (window.SiteMotion) window.SiteMotion.reveal(node);
  }

  var dateFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
  var timeFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", timeZone: "UTC", hour12: false });
  var shortDate = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
  function fmtScore(n) { return n.toFixed(1) + "%"; }
  function fmtCi(lo, hi) { return lo.toFixed(1) + "–" + hi.toFixed(1); }

  /* ---------- data validation ---------- */

  function num(v) { return typeof v === "number" && isFinite(v) ? v : null; }
  function str(v, max) { return typeof v === "string" ? v.slice(0, max) : ""; }
  function agentName(v) {
    if (typeof v !== "string") return "";
    var s = v.trim();
    return CODING_AGENT.test(s) && s === s.trim() ? s : "";
  }

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
      url: str(e.commit_url, 300),
      agent: agentName(e.coding_agent)
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

  /* A real <table>; the explicit ARIA roles keep its semantics when CSS lays rows out as grids. */
  function buildTable(label, headers, rows, makeCells, extraClass) {
    var head = el("tr", { role: "row" }, headers.map(function (h) {
      var th = el("th", { scope: "col", role: "columnheader", "class": h.cls || "" }, [h.text]);
      if (h.extra) th.appendChild(h.extra);
      return th;
    }));
    var body = el("tbody", { role: "rowgroup" }, rows.map(function (r, i) {
      var cells = makeCells(r, i);
      var cls = (r.isBase ? "is-base " : "") + (r.bandFirst ? "band-first" : "");
      return el("tr", { role: "row", "class": cls }, cells.map(function (c, j) {
        return el("td", { role: "cell", "data-label": headers[j].text, "class": c.cls }, c.nodes);
      }));
    }));
    var table = el("table", { "class": "board" + (extraClass ? " " + extraClass : ""), role: "table" }, [
      el("caption", { "class": "sr-only", text: label }),
      el("thead", { role: "rowgroup" }, [head]),
      body
    ]);
    return el("div", { "class": "table-wrap" }, [table]);
  }

  function markBandStarts(rows) {
    rows.forEach(function (r, i) { r.bandFirst = i === 0 || rows[i - 1].band !== r.band; });
  }

  function hash(s) {
    var h = 0;
    for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return h;
  }

  function avatar(r) {
    if (r.isBase) return el("span", { "class": "avatar av-base", "aria-hidden": "true" }, [el("span")]);
    var initial = (r.account.match(/[A-Za-z0-9]/) || ["?"])[0].toUpperCase();
    var disc = el("span", { "class": "avatar av-" + (hash(r.account) % 4), "aria-hidden": "true" }, [initial]);
    if (validLogin(r.account)) {
      var img = el("img", {
        src: "https://github.com/" + r.account + ".png?size=64", alt: "", width: "32", height: "32",
        loading: "lazy", decoding: "async", referrerpolicy: "no-referrer"
      });
      img.addEventListener("error", function () { if (img.parentNode) img.parentNode.removeChild(img); });
      disc.appendChild(img);
    }
    return disc;
  }

  function agentIcon(color, framed) {
    return sv("svg", { "aria-hidden": "true", width: framed ? "14" : "16", height: framed ? "14" : "16", viewBox: "0 0 16 16" }, [
      framed ? sv("rect", { x: "0.75", y: "0.75", width: "14.5", height: "14.5", fill: "none", stroke: color, "stroke-width": "1.5" })
        : sv("rect", { x: "0", y: "0", width: "16", height: "16", fill: "#11141B" }),
      sv("path", { d: "M4 4.5 L7.5 8 L4 11.5", stroke: color, "stroke-width": "1.8", fill: "none" }),
      sv("path", { d: "M9 11.5 H12.5", stroke: color, "stroke-width": "1.8" })
    ]);
  }

  function accountNodes(r, sub) {
    var line = el("span", { "class": "who-line" });
    if (r.isBase) {
      line.appendChild(el("span", { "class": "who-name", text: "Base solution" }));
      line.appendChild(el("span", { "class": "base-tag", text: "Base" }));
      sub = "Vanilla Inkling 2 Mini with the CaP-X harness";
    } else {
      var p = profileUrl(r.account);
      line.appendChild(p ? link(p, r.account, { "class": "who-name" }) : el("span", { "class": "who-name", text: r.account }));
    }
    if (r.agent) {
      line.appendChild(el("span", { "class": "agent-tag", title: "Coding agent" }, [
        agentIcon("#FFFFFF", false),
        el("span", {}, [el("span", { "class": "sr-only", text: "Coding agent: " }), r.agent])
      ]));
    }
    var who = el("span", { "class": "who" }, [line]);
    if (sub) who.appendChild(el("span", { "class": "who-sub", text: sub }));
    return [avatar(r), who];
  }

  /* scale for the interval bars: whole tens around every interval shown */
  function scaleFor(rows) {
    var d0 = Math.floor(Math.min.apply(null, rows.map(function (r) { return r.lo; })) / 10) * 10;
    var d1 = Math.ceil(Math.max.apply(null, rows.map(function (r) { return r.hi; })) / 10) * 10;
    d0 = Math.max(0, d0); d1 = Math.min(100, d1);
    if (d1 <= d0) { d0 = Math.max(0, d0 - 10); d1 = Math.min(100, d0 + 20); }
    return { d0: d0, d1: d1, at: function (v) { return Math.max(0, Math.min(100, (v - d0) / (d1 - d0) * 100)); } };
  }
  function scaleHeader(scale) {
    return el("span", { "class": "th-scale", "aria-hidden": "true" }, [
      el("span"), el("span", {}, [el("span", { text: scale.d0 + "%" }), el("span", { text: scale.d1 + "%" })])
    ]);
  }

  function scoreNodes(r, scale, baseScore, i) {
    var left = scale.at(Math.min(r.lo, r.score)), right = scale.at(Math.max(r.hi, r.score)), dot = scale.at(r.score);
    var width = Math.max(right - left, 0.5);
    var range = el("span", { "class": "range" });
    range.style.left = left.toFixed(2) + "%";
    range.style.width = width.toFixed(2) + "%";
    range.style.setProperty("--o", ((dot - left) / width * 100).toFixed(1) + "%");
    range.style.setProperty("--i", String(Math.min(i, 20)));
    var dotEl = el("span", { "class": "dot" });
    dotEl.style.left = dot.toFixed(2) + "%";
    var bar = el("span", { "class": "ivl", "aria-hidden": "true" }, [el("span", { "class": "track" })]);
    if (baseScore !== null) {
      var tick = el("span", { "class": "base-tick" });
      tick.style.left = scale.at(baseScore).toFixed(2) + "%";
      bar.appendChild(tick);
    }
    bar.appendChild(range);
    bar.appendChild(dotEl);
    bar.appendChild(el("span", { "class": "scale" }, [el("span", { text: scale.d0 + "%" }), el("span", { text: scale.d1 + "%" })]));
    return [
      el("span", { "class": "score-num" }, [
        el("span", { "class": "num", text: fmtScore(r.score) }),
        el("span", { "class": "ci", text: "(" + fmtCi(r.lo, r.hi) + ")" })
      ]),
      bar
    ];
  }
  /* "17 Dec 2026, 11:35:00" as two parts, so the column wraps the same way on every row */
  function timeNode(d) {
    var full = timeFmt.format(d), cut = full.lastIndexOf(", ");
    var parts = cut > 0 ? [full.slice(0, cut + 2), full.slice(cut + 2)] : [full];
    return el("time", { datetime: d.toISOString() }, parts.map(function (t) { return el("span", { text: t }); }));
  }
  function bandNodes(r) { return [el("span", { "class": "band-badge", text: String(r.band) })]; }

  function renderStandings(data, showAll) {
    var host = $("standings-table");
    clear(host);
    var rows = (showAll ? data.entries.slice() : bestPerAccount(data.entries)).sort(byScore);
    if (!rows.length) {
      host.appendChild(el("div", { "class": "table-wrap" }, [
        el("p", { "class": "empty", text: "No submissions have been scored yet. The base solution appears here first." })
      ]));
      return;
    }
    assignBands(rows);
    markBandStarts(rows);
    var scale = scaleFor(rows);
    var base = data.entries.filter(function (e) { return e.isBase; })[0];
    var baseScore = base ? base.score : null;
    host.appendChild(buildTable(
      showAll ? "All public submissions by public score" : "Best public submission per account by public score",
      [{ text: "Band" }, { text: "Account" },
        { text: "Public score (95% interval)", cls: "th-score", extra: scaleHeader(scale) },
        { text: "Tokens per episode" }, { text: "Time (UTC)" }, { text: "Submission" }],
      rows,
      function (r, i) {
        var d = new Date(r.t);
        var sub = !showAll && r.count > 1 ? r.count + " submissions" : null;
        var tk = r.tokens ? TOKENS[r.tokens] : null;
        var name = r.isBase ? "the base solution" : r.account;
        return [
          { cls: "cell-band", nodes: bandNodes(r) },
          { cls: "cell-account", nodes: accountNodes(r, sub) },
          { cls: "cell-score", nodes: scoreNodes(r, scale, baseScore, i) },
          { cls: "cell-tokens", nodes: tk ? [el("span", { "class": "pips", "aria-hidden": "true" }, [1, 2, 3].map(function (n) {
            return el("span", { "class": n <= tk[1] ? "on" : "" });
          })), el("span", { text: tk[0] })] : ["–"] },
          { cls: "cell-date", nodes: [timeNode(d)] },
          { cls: "cell-commit", nodes: [r.url ? link(r.url, r.id || "commit", { "aria-label": "Commit " + (r.id || "") + " by " + name }) : "–"] }
        ];
      }
    ));
    reveal(host);
  }

  function renderFinal(final) {
    var section = $("final");
    var host = $("final-table");
    clear(host);
    if (!final || !final.rows.length) { section.hidden = true; return; }
    section.hidden = false;
    var rows = final.rows.slice().sort(function (a, b) { return a.band - b.band || b.score - a.score; });
    markBandStarts(rows);
    var scale = scaleFor(rows);
    var base = rows.filter(function (r) { return r.isBase; })[0];
    var meta = $("final-meta");
    meta.textContent = final.revealed ? "Revealed " + dateFmt.format(new Date(final.revealed)) + ". " : "";
    host.appendChild(buildTable(
      "Final ranking by private score",
      [{ text: "Band" }, { text: "Account" },
        { text: "Private score (95% interval)", cls: "th-score", extra: scaleHeader(scale) },
        { text: "Runs averaged" }, { text: "Chosen submissions" }],
      rows,
      function (r, i) {
        var picks = r.picks.map(function (p) {
          return el("span", {}, [p.url ? link(p.url, p.id || "commit", { "aria-label": "Commit " + p.id + " by " + r.account }) : p.id]);
        });
        if (r.source === "default") picks.push(el("span", { "class": "who-sub", text: "most recent two (no picks made)" }));
        return [
          { cls: "cell-band", nodes: bandNodes(r) },
          { cls: "cell-account", nodes: accountNodes(r, null) },
          { cls: "cell-score", nodes: scoreNodes(r, scale, base ? base.score : null, i) },
          { cls: "cell-runs", nodes: [r.runs === 3 ? "3 runs (mean)" : "1 run"] },
          { cls: "cell-picks", nodes: [picks.length ? el("span", { "class": "picks" }, picks) : "–"] }
        ];
      },
      "final"
    ));
    reveal(host);
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

  var DAY = 86400000;
  function timeTicks(t0, t1, maxCount) {
    var steps = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365];
    var span = (t1 - t0) / DAY, step = steps[steps.length - 1];
    for (var i = 0; i < steps.length; i++) { if (span / steps[i] <= maxCount) { step = steps[i]; break; } }
    var start = Math.floor(t0 / DAY) * DAY, ticks = [];
    for (var t = start; t <= t1; t += step * DAY) if (t >= t0) ticks.push(t);
    return ticks;
  }
  function pct(v) { return v.toFixed(2) + "%"; }
  function who(e) { return e.isBase ? "Base solution" : e.account; }

  function drawChart(host, model, w, h) {
    clear(host);
    var all = model.subs.concat(model.base ? [model.base] : []);
    var maxScore = Math.max.apply(null, all.map(function (e) { return e.score; }));
    var yMax = Math.min(100, Math.max(20, Math.ceil((maxScore + 5) / 10) * 10));
    var timed = model.subs.length ? model.subs : [model.base];
    var t0 = timed[0].t, t1 = timed[timed.length - 1].t;
    if (t1 - t0 < DAY) { t0 -= DAY / 2; t1 += DAY / 2; }
    var padT = (t1 - t0) * 0.04;
    t0 -= padT; t1 += padT;
    function X(t) { return (t - t0) / (t1 - t0) * 100; }
    function Y(v) { return 100 - v / yMax * 100; }

    var last = model.subs.length ? model.subs[model.subs.length - 1] : null;
    host.setAttribute("role", "group");
    host.setAttribute("aria-label", "Chart of the best public score over time. " +
      (last ? "The latest best is " + fmtScore(last.best) + ". " : "No submissions yet. ") +
      (model.base ? "The base solution scored " + fmtScore(model.base.score) + ". " : "") + "The table below lists every value.");

    var plot = el("div", { "class": "plot" });
    var deco = el("div", { "aria-hidden": "true" });
    plot.appendChild(deco);

    var yStep = yMax <= 60 ? 10 : 20;
    for (var v = 0; v <= yMax; v += yStep) {
      var yt = el("div", { "class": v === 0 ? "ytick zero" : "ytick" }, [el("span", { text: v + "%" })]);
      yt.style.top = pct(Y(v));
      deco.appendChild(yt);
    }
    var xs = model.subs.length ? timeTicks(t0, t1, Math.max(2, Math.min(7, Math.floor(w / 72)))) : [model.base.t];
    xs.forEach(function (t) {
      var xt = el("div", { "class": "xtick", text: shortDate.format(new Date(t)) });
      xt.style.left = pct(X(t));
      deco.appendChild(xt);
    });

    if (model.base) {
      var bl = el("div", { "class": "base-line" });
      bl.style.top = pct(Y(model.base.score));
      var lbl = el("div", { "class": "base-lbl", text: "Base solution " + fmtScore(model.base.score) });
      lbl.style.top = pct(Y(model.base.score));
      deco.appendChild(bl);
      deco.appendChild(lbl);
    }

    if (last) {
      /* best-so-far as a step line in pixels, so it can be drawn in with a dash offset */
      var d = "";
      model.subs.forEach(function (e) {
        if (!e.record) return;
        var x = (X(e.t) / 100 * w).toFixed(1), y = (Y(e.best) / 100 * h).toFixed(1);
        d += d ? " H" + x + " V" + y : "M" + x + " " + y;
      });
      d += " H" + (X(last.t) / 100 * w).toFixed(1);
      var path = sv("path", { "class": "step", d: d });
      var svg = sv("svg", { "class": "step-svg", width: String(w), height: String(h), viewBox: "0 0 " + w + " " + h }, [path]);
      deco.appendChild(svg);
      var best = el("div", { "class": "best-lbl", text: "Best " + fmtScore(last.best) });
      best.style.left = pct(X(last.t));
      best.style.top = pct(Y(last.best));
      deco.appendChild(best);
    }

    var tip = el("div", { "class": "tip", "aria-hidden": "true", hidden: "" });
    var current = null;
    function show(e, btn) {
      if (current) current.classList.remove("on");
      current = btn;
      btn.classList.add("on");
      clear(tip);
      tip.appendChild(el("span", { "class": "t-who", text: who(e) }));
      if (e.agent) tip.appendChild(el("span", { "class": "t-agent" }, [agentIcon("#FFC800", true), el("span", { text: e.agent })]));
      tip.appendChild(el("span", { "class": "t-score", text: fmtScore(e.score) + "  (" + fmtCi(e.lo, e.hi) + ")" }));
      tip.appendChild(el("span", { "class": "t-when", text: timeFmt.format(new Date(e.t)) + " UTC" }));
      var tx = Math.min(Math.max(X(e.t), 12), 88), ty = Y(e.score);
      tip.style.left = pct(tx);
      tip.style.top = pct(ty);
      tip.style.transform = ty < 35 ? "translate(-50%, 18px)" : "translate(-50%, calc(-100% - 18px))";
      tip.hidden = false;
    }
    function hide() {
      tip.hidden = true;
      if (current) current.classList.remove("on");
      current = null;
    }

    var pts = model.subs.length ? model.subs : [model.base];
    pts.forEach(function (e, i) {
      var label = who(e) + (e.agent && !e.isBase ? ", built with " + e.agent : "") + ": " + fmtScore(e.score) +
        " (" + fmtCi(e.lo, e.hi) + "), " + timeFmt.format(new Date(e.t)) + " UTC";
      var btn = el("button", { type: "button", "class": e.record ? "pt record" : "pt", "aria-label": label }, [el("i")]);
      btn.style.left = pct(X(e.t));
      btn.style.top = pct(Y(e.score));
      btn.style.setProperty("--i", String(Math.min(i, 30)));
      function on() { show(e, btn); }
      btn.addEventListener("mouseenter", on);
      btn.addEventListener("focus", on);
      btn.addEventListener("click", on);
      btn.addEventListener("mouseleave", hide);
      btn.addEventListener("blur", hide);
      btn.addEventListener("keydown", function (ev) { if (ev.key === "Escape") hide(); });
      plot.appendChild(btn);
    });
    plot.appendChild(tip);
    host.appendChild(plot);

    if (path && path.getTotalLength) {
      var len = Math.ceil(path.getTotalLength()) + 2;
      path.style.setProperty("--len", String(len));
    }
  }

  var chartObserver = null;
  function renderChart(entries) {
    var section = $("progress");
    var host = $("chart-host");
    var model = chartModel(entries);
    if (!model.subs.length && !model.base) { section.hidden = true; clear(host); return; }
    section.hidden = false;
    var lastW = 0, lastH = 0;
    function draw() {
      /* plot area = box minus the axis margins (60 + 24, 24 + 44) and borders */
      var w = Math.round(host.clientWidth) - 84, h = Math.round(host.clientHeight) - 68;
      if (w < 100 || h < 100 || (w === lastW && h === lastH)) return;
      lastW = w; lastH = h;
      drawChart(host, model, w, h);
    }
    draw();
    if (chartObserver) chartObserver.disconnect();
    if (window.ResizeObserver) { chartObserver = new ResizeObserver(draw); chartObserver.observe(host); }
    else window.addEventListener("resize", draw);
    reveal(host);
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
      status.classList.remove("pop");
      void status.offsetWidth; /* restart the small entry animation */
      status.classList.add("pop");
      clearTimeout(timer);
      timer = setTimeout(function () { status.textContent = ""; status.classList.remove("pop"); }, 3000);
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

  function showSampleTags(on) {
    Array.prototype.forEach.call(document.querySelectorAll("[data-sample-tag]"), function (t) { t.hidden = !on; });
  }
  function hideData() {
    $("standings-body").hidden = true;
    $("standings-controls").hidden = true;
    $("progress").hidden = true;
    $("final").hidden = true;
  }

  function showData(data, sample) {
    var showAll = $("show-all");
    $("standings-body").hidden = false;
    $("standings-controls").hidden = false;
    renderStandings(data, showAll.checked);
    showAll.onchange = function () { renderStandings(data, showAll.checked); };
    renderFinal(data.final);
    renderChart(data.entries);
    showSampleTags(sample);
    setStatus(null, "");
  }

  function load() {
    var src = sourceUrl();
    hideData();
    showSampleTags(false);
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
        hideData();
        showSampleTags(false);
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
