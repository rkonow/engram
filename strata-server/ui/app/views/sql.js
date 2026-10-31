// The SQL workbench: tabs of queries in an editor that completes tables,
// fields and functions, parameters filled in a form (a vector parameter
// can be text the server embeds), the answer as a grid, a chart, the plan
// as a tree with what ran on each segment, its profile (every step it
// took, with the request's trace and CPU time when asked) and the JSON; a
// sidebar of tables and fields to drag into the editor, snippets written
// for the table's own fields, history and saved queries.

import { h, icon, clear, fmt, api, store, toast, copy, modal, drawer, empty, seg, sqlBlock, debounce, ApiError, failure } from "../core.js";
import { editor, quoteName } from "../sqledit.js";
import { tables, cached, invalidate, typeBadge } from "../catalog.js";
import { grid, resultChart, planTree, segments, answerStats, exportMenu } from "../results.js";
import { profileView } from "../profile.js";

export default { id: "sql", title: "SQL workbench", icon: "sql", group: "Data", full: true, keywords: "query editor sql run explain plan results chart history saved snippets parameters", render, onParams };

let openSql = null;
function onParams(r) { const q = r.params.get("q"); if (q && openSql) openSql(q, r.params.get("p")); }

const uid = () => Math.random().toString(36).slice(2, 9);
const READ = /^\s*(\(|SELECT\b|WITH\b)/i;
const stripStrings = (sql) => sql.replace(/'(?:[^']|'')*'/g, "''").replace(/--[^\n]*/g, "");

/** The `:name` parameters a statement uses, in order. */
export function paramNames(sql) { return [...new Set([...stripStrings(sql).matchAll(/(?<![:\w]):([A-Za-z_]\w*)/g)].map((m) => m[1]))]; }

/** The statement the caret is in, or the selection. */
function statementAt(text, [a, b]) {
  if (b > a) return text.slice(a, b);
  let start = 0, inStr = false;
  for (let i = 0; i <= text.length; i++) {
    const c = text[i];
    if (c === "'") inStr = !inStr;
    if ((c === ";" && !inStr) || i === text.length) {
      if (a <= i || i === text.length) { const s = text.slice(start, i); if (s.trim()) return s; }
      start = i + 1;
    }
  }
  return text.split(";").map((x) => x.trim()).filter(Boolean).pop() || "";
}

const UPPER = new Set("select from where and or not in is null between like order by group having limit offset asc desc nulls distinct on as join left inner with exists case when then else end true false insert into values update set delete create table drop primary references explain union all".split(" "));
const CLAUSE1 = new Set(["SELECT", "FROM", "WHERE", "HAVING", "LIMIT", "OFFSET", "JOIN", "UNION", "VALUES", "SET", "WITH", "UPDATE"]);
const CLAUSE2 = new Set(["GROUP BY", "ORDER BY", "LEFT JOIN", "INNER JOIN", "INSERT INTO", "DELETE FROM"]);
/** Lays a statement out a clause a line, keywords in capitals, strings as they are. */
export function formatSql(sql) {
  const words = sql.match(/--[^\n]*|'(?:[^']|'')*'|"(?:[^"]|"")*"|:[A-Za-z_]\w*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*|[(),;]|[^\sA-Za-z_(),;'"]+/g) || [];
  let out = "", depth = 0, prev = "", skip = false, cond = false;
  for (let i = 0; i < words.length; i++) {
    let t = words[i];
    if (t.startsWith("--")) { out += (out && !out.endsWith("\n") ? "\n" : "") + t + "\n"; prev = ""; continue; }
    if (/^[A-Za-z_]\w*$/.test(t) && UPPER.has(t.toLowerCase())) t = t.toUpperCase();
    const two = t + " " + String(words[i + 1] || "").toUpperCase();
    const clause = depth === 0 && !skip && (CLAUSE2.has(two) || CLAUSE1.has(t));
    skip = depth === 0 && CLAUSE2.has(two);
    if (clause) cond = t === "WHERE" || t === "HAVING";
    const andOr = depth === 0 && cond && (t === "AND" || t === "OR") && !/BETWEEN\s+\S+$/.test(out);
    if (clause && out.trim()) out = out.replace(/\s+$/, "") + "\n";
    else if (andOr) out = out.replace(/\s+$/, "") + "\n  ";
    else if (out && !/[\n(]$/.test(out) && t !== ")" && t !== "," && t !== ";" && !(t === "(" && /^[A-Za-z_]/.test(prev) && !UPPER.has(prev.toLowerCase()))) out += " ";
    out += t;
    if (t === "(") depth++;
    if (t === ")") depth = Math.max(0, depth - 1);
    if (t === ";") out += "\n";
    prev = t;
  }
  return out.trim() + "\n";
}

function levenshtein(a, b) {
  const d = Array.from({ length: a.length + 1 }, (_, i) => [i]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  return d[a.length][b.length];
}

/** Snippets written for one table's own fields. */
export function snippets(t) {
  const fs = t?.short?.fields || [], name = quoteName(t?.table || "t"), key = t?.short?.key;
  const of = (ty) => fs.filter((f) => f.type === ty && !f.multi);
  const text = of("text")[0], vec = fs.find((f) => f.type === "vector" && !f.multi), kw = fs.find((f) => f.type === "keyword"), num = fs.find((f) => f.type === "int" || f.type === "float"), ts = of("timestamp")[0], edges = fs.find((f) => f.type === "edges");
  const cols = [key, ...fs.filter((f) => f.name !== key && (f.type === "text" || f.type === "keyword") && f.indexed !== false && !f.from).slice(0, 2).map((f) => f.name)].filter(Boolean).map(quoteName).join(", ") || "*";
  const out = [];
  out.push(["First rows", "Every column of the first rows", `SELECT * FROM ${name} LIMIT 20;`]);
  if (text) out.push(["Search words", "Rows with the words, best first by BM25", `SELECT ${cols}, score()\nFROM ${name}\nWHERE MATCH(${quoteName(text.name)}, 'your words')\nORDER BY bm25(${quoteName(text.name)}, 'your words') DESC\nLIMIT 10;`]);
  if (text) out.push(["Allow typos", "Words within two edits, ranked", `SELECT ${cols}, score()\nFROM ${name}\nORDER BY bm25(${quoteName(text.name)}, 'your wrods', fuzzy => 'auto') DESC\nLIMIT 10;`]);
  if (vec) out.push(["Similar to a text", "Nearest rows to a text the server embeds (:q)", `SELECT ${cols}, score()\nFROM ${name}\nORDER BY similarity(${quoteName(vec.name)}, :q) DESC\nLIMIT 10;`]);
  if (vec && text) out.push(["Hybrid", "30% words, 70% meaning", `SELECT ${cols}, score()\nFROM ${name}\nORDER BY 0.3 * tmm(bm25(${quoteName(text.name)}, :words)) + 0.7 * tmm(similarity(${quoteName(vec.name)}, :q)) DESC\nLIMIT 10;`]);
  if (vec && text) out.push(["Hybrid by rank", "Words and meaning fused by rank (RRF)", `SELECT ${cols}, score()\nFROM ${name}\nORDER BY rrf(bm25(${quoteName(text.name)}, :words), similarity(${quoteName(vec.name)}, :q)) DESC\nLIMIT 10;`]);
  if (kw) out.push(["Facet", `Rows by ${kw.name}, largest first`, `SELECT ${quoteName(kw.name)}, COUNT(*) AS n${num ? `, AVG(${quoteName(num.name)}) AS avg_${num.name}` : ""}\nFROM ${name}\nGROUP BY ${quoteName(kw.name)}\nORDER BY n DESC\nLIMIT 20;`]);
  if (num && text) out.push(["Boost by a value", `BM25 times ${num.name}`, `SELECT ${cols}, ${quoteName(num.name)}, score()\nFROM ${name}\nORDER BY bm25(${quoteName(text.name)}, 'your words') * (1 + 0.1 * coalesce(${quoteName(num.name)}, 0)) DESC\nLIMIT 10;`]);
  if (ts && text) out.push(["Fresh first", "BM25 halved for every 30 days of age", `SELECT ${cols}, ${quoteName(ts.name)}, score()\nFROM ${name}\nORDER BY bm25(${quoteName(text.name)}, 'your words') * decay(${quoteName(ts.name)}, now(), 2592000) DESC\nLIMIT 10;`]);
  if (edges) out.push(["Graph walk", "Rows within two hops of a row", `SELECT ${cols}, hops()\nFROM ${name}\nWHERE REACHABLE(${quoteName(edges.name)}, ('a key'), hops => 2)\nORDER BY hops()\nLIMIT 50;`]);
  if (edges) out.push(["PageRank", "Rows by personalized PageRank from a seed", `SELECT ${cols}, score()\nFROM ${name}\nORDER BY ppr(${quoteName(edges.name)}, ('a key')) DESC\nLIMIT 20;`]);
  if (key) out.push(["One row", "A row by its key", `SELECT * FROM ${name} WHERE ${quoteName(key)} = :key;`]);
  return out;
}

function render(page, r) {
  let tabs = store.get("sql.tabs", null);
  if (!tabs?.length) tabs = [{ id: uid(), name: "Query 1", sql: "", params: {} }];
  let active = store.get("sql.active", tabs[0].id);
  if (!tabs.some((t) => t.id === active)) active = tabs[0].id;
  let trace = store.get("sql.trace", true);
  let last = null, lastKind = "results", running = null;
  const tab = () => tabs.find((t) => t.id === active);
  const save = debounce(() => { store.set("sql.tabs", tabs); store.set("sql.active", active); }, 300);

  // ------------------------------------------------------------ the layout
  const side = h("aside", { class: "wb-side" });
  const tabBar = h("div", { class: "qtabs", role: "tablist" });
  const paramsBox = h("div", { class: "params hidden" });
  const out = h("div", { class: "wb-out" });
  const outTabs = h("div", { class: "tabs", role: "tablist", style: { marginBottom: "0" } });
  const outBody = h("div", { class: "wb-res" });
  const status = h("div", { class: "wb-status" });
  const runBtn = h("button", { class: "btn primary", title: "Run the statement at the caret (Ctrl+Enter)", onclick: () => run() }, icon("play"), "Run");
  const traceSw = h("label", { class: "row nw small dim", title: "Ask the server for the request's trace and the CPU time of every step the query took (its Profile tab)", style: { gap: "6px", cursor: "pointer" } },
    h("input", { type: "checkbox", checked: trace, onchange: (e) => { trace = e.target.checked; store.set("sql.trace", trace); } }), "profile");
  const ed = editor({
    value: tab().sql,
    onChange: (v) => { tab().sql = v; save(); drawParams(); },
    onRun: () => run(),
    onExplain: () => run(true),
    catalog: cached,
    params: () => paramNames(ed.value),
  });
  const edWrap = h("div", { class: "wb-ed", style: { height: store.get("sql.edH", 260) + "px" } }, ed.el);
  const splitter = h("div", { class: "splitter", title: "Drag to resize", role: "separator", "aria-orientation": "horizontal" });
  splitter.onpointerdown = (e) => {
    e.preventDefault();
    const y0 = e.clientY, h0 = edWrap.offsetHeight;
    const mv = (ev) => { edWrap.style.height = Math.max(120, Math.min(innerHeight - 220, h0 + ev.clientY - y0)) + "px"; };
    const up = () => { removeEventListener("pointermove", mv); removeEventListener("pointerup", up); store.set("sql.edH", edWrap.offsetHeight); };
    addEventListener("pointermove", mv); addEventListener("pointerup", up);
  };
  const tools = h("div", { class: "wb-tools" }, runBtn,
    h("button", { class: "btn", title: "Show the plan without the trace (Ctrl+Shift+Enter)", onclick: () => run(true) }, icon("layers"), "Explain"),
    h("button", { class: "btn ghost", title: "Lay the SQL out a clause a line", onclick: () => { ed.value = formatSql(ed.value); tab().sql = ed.value; save(); ed.focus(); } }, icon("menu"), "Format"),
    traceSw,
    h("span", { class: "sep" }),
    h("span", { class: "small muted hide-sm" }, h("kbd", {}, "Ctrl"), " ", h("kbd", {}, "Enter"), " run · ", h("kbd", {}, "Ctrl"), " ", h("kbd", {}, "Space"), " complete"),
    h("button", { class: "btn ghost icon", title: "Copy a link to this query", onclick: () => copy(location.origin + "/#/sql?q=" + encodeURIComponent(ed.value), "Link copied") }, icon("link")),
    h("button", { class: "btn ghost icon", title: "Save this query", onclick: saveQuery }, icon("plus")),
    h("button", { class: "btn ghost", title: "Saved queries and history", onclick: library }, icon("clock"), "Library"));
  page.append(h("div", { class: "wb" }, side, h("div", { class: "wb-main" }, tabBar, tools, edWrap, paramsBox, splitter, out)));
  out.append(h("div", { class: "row between wb-outhd" }, outTabs, status), outBody);

  // ------------------------------------------------------------ tabs
  function drawTabs() {
    clear(tabBar, tabs.map((t) => {
      const b = h("button", { role: "tab", class: "qt", "aria-selected": String(t.id === active), title: "Double-click to rename" },
        h("span", {}, t.name),
        tabs.length > 1 ? h("span", { class: "x", title: "Close", onclick: (e) => { e.stopPropagation(); close(t.id); } }, "×") : null);
      b.onclick = () => select(t.id);
      b.ondblclick = () => { const n = prompt("Name this tab", t.name); if (n?.trim()) { t.name = n.trim(); save(); drawTabs(); } };
      return b;
    }), h("button", { class: "qt add", title: "A new query", onclick: () => add("") }, icon("plus")));
  }
  function select(id) { active = id; ed.value = tab().sql; save(); drawTabs(); drawParams(); }
  function add(sql, name) { const t = { id: uid(), name: name || `Query ${tabs.length + 1}`, sql, params: {} }; tabs.push(t); select(t.id); ed.focus(); return t; }
  function close(id) { const i = tabs.findIndex((t) => t.id === id); tabs.splice(i, 1); if (active === id) active = tabs[Math.max(0, i - 1)].id; select(active); }
  // `p` is the parameters as JSON: {name: {kind, value, field}}
  openSql = (sql, p) => {
    let t = tabs.find((x) => !x.sql.trim());
    if (t) { t.sql = sql; select(t.id); } else t = add(sql);
    if (p) { try { t.params = JSON.parse(p); save(); drawParams(); } catch { /* a link with broken parameters opens without them */ } }
    history.replaceState(null, "", "#/sql");
  };

  // ------------------------------------------------------------ parameters
  function drawParams() {
    const names = paramNames(ed.value);
    paramsBox.classList.toggle("hidden", !names.length);
    if (!names.length) return;
    const t = tab(); t.params ||= {};
    const vecs = cached().flatMap((x) => (x.short?.fields || []).filter((f) => f.type === "vector" && f.embed).map((f) => ({ table: x.table, field: f.name })));
    clear(paramsBox, h("div", { class: "small muted", style: { marginBottom: "6px" } }, "Parameters"), names.map((n) => {
      const p = (t.params[n] ||= { kind: "auto", value: "" });
      const inp = h("input", { class: "input sm mono grow", value: p.value, placeholder: p.kind === "embed" ? "text for the server to embed" : p.kind === "json" ? "[0.1, 0.2] or {…}" : "a value", oninput: () => { p.value = inp.value; save(); } });
      const field = h("select", { class: "select sm", title: "The vector field whose model embeds the text", onchange: () => { p.field = field.value; save(); } }, h("option", { value: "" }, "first embedded field"), vecs.map((v) => h("option", { value: v.field, selected: p.field === v.field }, `${v.table}.${v.field}`)));
      return h("div", { class: "row nw", style: { gap: "8px", marginBottom: "4px" } }, h("span", { class: "mono", style: { minWidth: "90px" } }, ":" + n),
        seg([["auto", "auto"], ["text", "text"], ["number", "number"], ["json", "JSON"], ["embed", "embed text"]], p.kind, (v) => { p.kind = v; save(); drawParams(); }), inp, p.kind === "embed" ? field : null);
    }));
  }
  function buildParams(sql) {
    const out = {}, t = tab();
    for (const n of paramNames(sql)) {
      const p = t.params?.[n] || { kind: "auto", value: "" };
      const v = p.value ?? "";
      if (p.kind === "embed") out[n] = p.field ? { embed: v, field: p.field } : { embed: v };
      else if (p.kind === "number") out[n] = Number(v);
      else if (p.kind === "text") out[n] = v;
      else if (p.kind === "json" || (p.kind === "auto" && /^\s*[[{]/.test(v))) { try { out[n] = JSON.parse(v); } catch (e) { throw new Error(`:${n} is not JSON: ${e.message}`); } }
      else out[n] = /^-?\d+(\.\d+)?(e-?\d+)?$/i.test(v.trim()) ? Number(v) : v;
    }
    return out;
  }

  // ------------------------------------------------------------ running
  async function run(explain = false) {
    if (running) { running.abort(); return; }
    const sql = statementAt(ed.value, ed.selection).trim().replace(/;\s*$/, "");
    if (!sql) { toast("Write a statement first"); return; }
    let params;
    try { params = buildParams(sql); } catch (e) { showError(e.message, sql); return; }
    running = new AbortController();
    clear(runBtn, icon("stop"), "Stop"); runBtn.classList.add("danger");
    out.classList.add("loading");
    const t0 = performance.now();
    try {
      let j;
      if (explain) j = await api("/v1/explain", { body: { sql, params, hits: 5, profile: trace }, signal: running.signal });
      else if (READ.test(stripStrings(sql))) j = await api("/v1/query", { body: { sql, params, trace, profile: trace }, signal: running.signal });
      else j = await api("/v1/sql", { body: { sql, params }, signal: running.signal });
      j.roundtrip_ms = performance.now() - t0;
      last = { j, sql, explain };
      remember(sql, j, null);
      if (/^\s*(CREATE|DROP)\b/i.test(sql)) { invalidate(); tables(true).then(drawSide); toast(/^\s*CREATE/i.test(sql) ? "Table created" : "Table dropped", "ok"); }
      show(explain ? "plan" : j.written != null ? "results" : lastKind === "plan" ? "results" : lastKind);
    } catch (e) {
      if (e.name === "AbortError") { toast("Stopped waiting: the server may still finish it"); }
      else { remember(sql, null, e.message); showError(e.message, sql, e); }
    } finally {
      running = null; clear(runBtn, icon("play"), "Run"); runBtn.classList.remove("danger"); out.classList.remove("loading");
    }
  }
  function remember(sql, j, error) {
    const hist = store.get("sql.history", []);
    hist.unshift({ sql, ts: Date.now(), ms: j?.took_ms, rows: j?.rows?.length ?? j?.written, error });
    store.set("sql.history", hist.slice(0, 100));
  }
  function showError(msg, sql, e) {
    last = null;
    const placed = ed.markFrom(msg);
    const fix = h("div", { class: "row" });
    const m = /no such (?:column|field|table)s?:?\s+"?([\w.]+)"?/i.exec(msg);
    if (m) {
      const word = m[1].split(".").pop();
      const names = [...new Set(cached().flatMap((t) => [t.table, ...(t.short?.fields || []).map((f) => f.name)]))];
      const near = names.map((n) => [n, levenshtein(n.toLowerCase(), word.toLowerCase())]).filter(([, d]) => d > 0 && d <= (word.length < 4 ? 1 : Math.max(2, Math.floor(word.length / 3)))).sort((a, b) => a[1] - b[1]).slice(0, 3);
      for (const [n] of near) fix.append(h("button", { class: "btn sm", onclick: () => { ed.value = ed.value.replace(new RegExp(`\\b${word}\\b`, "g"), quoteName(n)); tab().sql = ed.value; save(); run(); } }, icon("wrench"), `Use ${n}`));
    }
    clear(outTabs); clear(status);
    clear(outBody, e && e.status === 403 ? failure(e, "This statement") : h("div", { class: "stack", style: { padding: "14px" } },
      h("div", { class: "error" }, msg),
      placed ? h("div", { class: "small muted" }, "The editor marks where the server stopped reading.") : null,
      fix.childElementCount ? h("div", {}, h("div", { class: "small muted", style: { marginBottom: "6px" } }, "Did you mean"), fix) : null,
      /MATCH|bm25|similarity|syntax|expected/i.test(msg) ? h("div", { class: "small muted" }, "The query language guide has an example of every function: ", h("a", { href: "https://rkonow.github.io/strata/strata/query-language.html", target: "_blank", rel: "noopener" }, "Query language"), ".") : null));
  }

  // ------------------------------------------------------------ the answer
  function show(kind) {
    lastKind = kind;
    if (!last) return;
    const { j, explain } = last;
    const kinds = explain ? [["plan", "Plan"], ["profile", "Profile"], ["json", "JSON"]] : j.written != null ? [["results", "Written"], ["json", "JSON"]] : [["results", `Results ${fmt.n(j.rows?.length || 0)}`], ["chart", "Chart"], ["plan", "Plan"], ["profile", "Profile"], ["json", "JSON"]];
    if (kind === "trace") kind = "profile";
    if (!kinds.some(([k]) => k === kind)) kind = kinds[0][0];
    clear(outTabs, kinds.map(([k, l]) => h("button", { role: "tab", "aria-selected": String(k === kind), onclick: () => show(k) }, l)));
    clear(status, explain ? h("span", { class: "small" }, h("b", {}, j.kind), " · ", fmt.ms(j.took_ms), " · ", fmt.plural(j.rows, "row")) : answerStats(j),
      j.rows?.length ? h("span", { class: "row nw", style: { gap: "2px" } }, exportMenu(j, "strata-query")) : null);
    if (kind === "results" && j.written != null) clear(outBody, h("div", { class: "stack", style: { padding: "16px" } }, h("div", { class: "banner ok" }, icon("okc", "lg"), h("div", {}, h("div", { class: "big" }, `${fmt.plural(j.written, "row")} written`), h("div", { class: "small muted" }, `in ${fmt.ms(j.took_ms)}${j.seq_no != null ? ` · sequence number ${j.seq_no}` : ""}${j.skipped ? ` · ${fmt.n(j.skipped)} skipped` : ""}`)))));
    else if (kind === "results") clear(outBody, grid(j));
    else if (kind === "chart") clear(outBody, h("div", { style: { padding: "12px 14px" } }, resultChart(j)));
    else if (kind === "profile") {
      const steps = explain ? j.steps : j.stats?.steps;
      const cpu = steps?.some((x) => x.cpu_us != null);
      clear(outBody, h("div", { style: { padding: "12px 14px" } },
        h("div", { class: "row", style: { marginBottom: "10px" } }, h("span", { class: "small muted" }, cpu ? "Every step the query took, with its CPU time." : "Every step the query took. Tick “profile” and run it again for the CPU time of each and the request's trace."), h("span", { class: "sep" }),
          h("a", { class: "btn sm", href: "#/profile?tab=query&sql=" + encodeURIComponent(last.sql), title: "Run it in the Profiler, as many times as you like" }, icon("flame"), "Open in the Profiler")),
        profileView({ steps, trace: j.trace, took_ms: j.took_ms, roundtrip_ms: j.roundtrip_ms, plan: explain ? j.plan : j.stats?.plan, rows: explain ? j.rows : j.rows?.length, filter_docs: explain ? j.filter_docs : j.stats?.filter_docs, scored: explain ? j.scored : j.stats?.scored, profiled: cpu }, { compact: true })));
    }
    else if (kind === "json") clear(outBody, h("pre", { class: "code", style: { margin: "12px 14px", maxHeight: "60vh" } }, JSON.stringify(j.rows && j.rows.length > 200 ? { ...j, rows: j.rows.slice(0, 200), note: `${j.rows.length - 200} more rows left out` } : j, null, 2)));
    else if (kind === "plan") explainView(explain ? j : null);
  }
  async function explainView(j) {
    if (!j) {
      clear(outBody, h("div", { class: "small muted", style: { padding: "14px" } }, "Explaining…"));
      try { j = await api("/v1/explain", { body: { sql: last.sql, params: buildParams(last.sql), hits: 5 } }); }
      catch (e) { clear(outBody, h("div", { class: "error", style: { margin: "14px" } }, e.message)); return; }
    }
    const hits = j.hits?.length ? h("div", { class: "tbl-wrap" }, grid({ columns: Object.keys(j.hits[0]), rows: j.hits.map((x) => Object.values(x)) })) : null;
    clear(outBody, h("div", { class: "plan-view" },
      h("div", { class: "grid g2" },
        h("section", { class: "card" }, h("div", { class: "hd" }, h("h3", {}, "Plan"), h("span", { class: "sub" }, "the statement as bound: rows flow up")), h("div", { class: "bd" }, planTree(j.tree))),
        h("section", { class: "card" }, h("div", { class: "hd" }, h("h3", {}, "What ran"), h("span", { class: "sub" }, `${fmt.n(j.filter_docs)} passed the filter · ${fmt.n(j.scored)} scored`)), h("div", { class: "bd" }, segments(j.plan)))),
      hits ? h("section", { class: "card", style: { marginTop: "14px" } }, h("div", { class: "hd" }, h("h3", {}, "The top rows and the values they were ranked by")), h("div", { class: "bd flush" }, hits)) : null,
      h("details", { class: "more" }, h("summary", {}, "The plan as JSON, which runs on its own (POST /v1/plan)"),
        h("div", { class: "row", style: { margin: "8px 0" } }, h("button", { class: "btn sm", onclick: () => copy(JSON.stringify(j.ir), "Plan copied") }, icon("copy"), "Copy")),
        h("pre", { class: "code", style: { maxHeight: "40vh" } }, JSON.stringify(j.ir, null, 2))),
      h("details", { class: "more" }, h("summary", {}, "The query as bound to the schema"), h("pre", { class: "code", style: { maxHeight: "40vh" } }, j.logical))));
  }

  // ------------------------------------------------------------ the library
  function saveQuery() {
    const inp = h("input", { class: "input", value: tab().name, style: { width: "100%" } });
    const m = modal({ title: "Save this query", body: [inp, sqlBlock(ed.value, { style: { maxHeight: "30vh" } })], foot: [h("button", { class: "btn", onclick: () => m.close() }, "Cancel"), h("button", { class: "btn primary", onclick: go }, "Save")] });
    inp.onkeydown = (e) => { if (e.key === "Enter") go(); };
    function go() { const saved = store.get("sql.saved", []).filter((x) => x.name !== inp.value.trim()); saved.unshift({ name: inp.value.trim() || "Query", sql: ed.value, params: tab().params, ts: Date.now() }); store.set("sql.saved", saved); tab().name = inp.value.trim() || tab().name; save(); drawTabs(); m.close(); toast("Saved in this browser", "ok"); }
  }
  function library() {
    const q = h("input", { class: "input", placeholder: "Search the SQL…", style: { width: "100%" } });
    const list = h("div", { class: "stack", style: { gap: "8px" } });
    let which = "saved";
    const tabsEl = seg([["saved", "Saved"], ["history", "History"]], which, (v) => { which = v; draw(); });
    const d = drawer({ title: "Library", sub: "kept in this browser", wide: true, body: [h("div", { class: "row" }, tabsEl, h("span", { class: "sep" })), q, list] });
    q.oninput = draw;
    function draw() {
      const needle = q.value.toLowerCase();
      const items = (which === "saved" ? store.get("sql.saved", []) : store.get("sql.history", [])).filter((x) => !needle || x.sql.toLowerCase().includes(needle) || x.name?.toLowerCase().includes(needle));
      clear(list, items.length ? items.map((x, i) => h("div", { class: "lib-it" },
        h("div", { class: "row between" }, h("b", {}, x.name || fmt.datetime(x.ts)), h("span", { class: "small muted" }, x.error ? h("span", { class: "pill bad" }, "failed") : x.ms != null ? `${fmt.ms(x.ms)} · ${fmt.plural(x.rows ?? 0, "row")}` : "", x.name ? " · " + fmt.ago(x.ts) : "")),
        sqlBlock(x.sql, { style: { maxHeight: "120px", marginTop: "6px" } }),
        h("div", { class: "row", style: { marginTop: "6px" } },
          h("button", { class: "btn sm primary", onclick: () => { d.close(); const t = add(x.sql, x.name); if (x.params) { t.params = x.params; drawParams(); } } }, "Open in a tab"),
          h("button", { class: "btn sm ghost", onclick: () => copy(x.sql, "SQL copied") }, icon("copy"), "Copy"),
          which === "saved" ? h("button", { class: "btn sm ghost danger", onclick: () => { const s = store.get("sql.saved", []); s.splice(i, 1); store.set("sql.saved", s); draw(); } }, icon("trash"), "Remove") : null)))
        : empty(which === "saved" ? "plus" : "clock", which === "saved" ? "Nothing saved yet" : "No history yet", which === "saved" ? "The + above the editor saves a query here." : "Every statement you run is kept here."));
    }
    draw();
  }

  // ------------------------------------------------------------ the sidebar
  const filter = h("input", { class: "input sm", placeholder: "Filter fields…", style: { width: "100%" }, oninput: () => drawSide() });
  const open = new Set(store.get("sql.open", []));
  function drawSide() {
    const ts = cached();
    const q = filter.value.toLowerCase();
    const first = ts.find((t) => open.has(t.table)) || ts[0];
    clear(side, h("div", { class: "wb-sh" }, h("b", {}, "Tables"), h("span", { class: "sep" }), h("button", { class: "btn ghost icon sm", title: "Read the tables again", onclick: () => tables(true).then(drawSide) }, icon("refresh"))), filter,
      h("div", { class: "tree" }, ts.map((t) => {
        const fs = (t.short?.fields || []).filter((f) => !q || f.name.toLowerCase().includes(q) || t.table.toLowerCase().includes(q));
        if (q && !fs.length) return null;
        const isOpen = open.has(t.table) || !!q || ts.length === 1;
        return h("div", { class: "tt" },
          h("div", { class: "tt-h", draggable: "true", title: "Click to open · drag into the editor", ondragstart: (e) => e.dataTransfer.setData("text/plain", quoteName(t.table)), onclick: () => { if (open.has(t.table)) open.delete(t.table); else open.add(t.table); store.set("sql.open", [...open]); drawSide(); } },
            icon(isOpen ? "down" : "right"), icon("table"), h("span", { class: "grow ellipsis" }, t.table), h("span", { class: "small muted" }, fs.length),
            h("button", { class: "btn ghost icon sm", title: `SELECT * FROM ${t.table} in a new tab`, onclick: (e) => { e.stopPropagation(); add(`SELECT * FROM ${quoteName(t.table)} LIMIT 20;`, t.table); run(); } }, icon("play"))),
          isOpen ? h("div", { class: "tt-f" }, fs.map((f) => h("div", { class: "fi", draggable: "true", title: `${f.name}: drag into the editor, or click to insert`, ondragstart: (e) => e.dataTransfer.setData("text/plain", quoteName(f.name)), onclick: () => ed.insert(quoteName(f.name)) },
            typeBadge(f), h("span", { class: "grow ellipsis" }, f.name), t.short.key === f.name ? h("span", { class: "badge accent", title: "the primary key" }, "key") : null, f.multi ? h("span", { class: "badge" }, "[]") : null))) : null);
      })),
      h("div", { class: "wb-sh", style: { marginTop: "14px" } }, h("b", {}, "Snippets"), h("span", { class: "small muted" }, first ? "for " + first.table : "")),
      h("div", { class: "snips" }, snippets(first).map(([name, what, sql]) => h("div", { class: "snip", draggable: "true", title: "Drag into the editor, or click to open in a new tab", ondragstart: (e) => e.dataTransfer.setData("text/plain", sql), onclick: () => add(sql, name) }, h("b", {}, name), h("div", { class: "small muted" }, what)))));
  }

  drawTabs(); drawParams();
  clear(outBody, empty("sql", "Run a query to see its answer", "Ctrl+Enter runs the statement at the caret; Ctrl+Shift+Enter explains it."));
  tables().then(() => { drawSide(); drawParams(); if (!tab().sql.trim() && cached()[0]) { ed.value = `SELECT * FROM ${quoteName(cached()[0].table)} LIMIT 20;`; tab().sql = ed.value; save(); } }).catch((e) => clear(side, failure(e, "The tables")));
  if (r.params.get("q")) openSql(r.params.get("q"), r.params.get("p"));
  setTimeout(() => ed.focus(), 50);
  return () => { openSql = null; running?.abort(); };
}

