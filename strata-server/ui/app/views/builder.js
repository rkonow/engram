// The query builder: a query as a flow of stages, built by dragging
// blocks and fields onto them. Rows come from a table, words and meaning
// find and rank them (fused when there are several), filters keep some,
// boosts lift some, groups count them, and the answer shows the columns
// chosen. The SQL is written as you build, runs as you build, and the
// connectors between stages show how many rows each step kept.

import { h, icon, clear, fmt, api, store, toast, copy, menu, empty, seg, sqlBlock, debounce, failure, sortable } from "../core.js";
import { quoteName } from "../sqledit.js";
import { tables, cached, table as tableOf, typeBadge, typeMeta, operators, lit } from "../catalog.js";
import { grid, answerStats, exportMenu } from "../results.js";
import { SERIES } from "../charts.js";

export default { id: "builder", title: "Query builder", icon: "builder", group: "Data", full: true, keywords: "visual query builder drag drop filter facet boost hybrid fusion rrf group no code", render, onParams };

let reopen = null;
function onParams(r) { const t = r.params.get("table"); if (t && reopen) reopen(t); }

const uid = () => Math.random().toString(36).slice(2, 9);
const isNum = (f) => f && (f.type === "int" || f.type === "float");
const isTime = (f) => f && f.type === "timestamp";
const searchable = (f) => f && f.type === "text" && f.indexed !== false;
const groupable = (f) => f && (f.type === "keyword" || isNum(f)) && f.indexed !== false;
const sortableField = (f) => f && (f.type === "keyword" || isNum(f) || isTime(f) || f.type === "bool") && !f.multi;
const showable = (f) => f && f.type !== "vector" && f.type !== "sparse" && f.stored !== false;
const filterable = (f) => operators(f).length > 0 && (f.type !== "text" || f.indexed !== false);

/** The blocks of the palette, and the stage each one lands in. */
const BLOCKS = [
  { kind: "words", stage: "find", icon: "search", label: "Words", what: "match and rank by BM25, typos allowed" },
  { kind: "meaning", stage: "find", icon: "sparkle", label: "Meaning", what: "nearest vectors to a text the server embeds" },
  { kind: "filter", stage: "filter", icon: "filter", label: "Filter", what: "keep the rows a condition holds for" },
  { kind: "closeness", stage: "boost", icon: "clock", label: "Recency", what: "decay by age, or by distance from a value" },
  { kind: "number", stage: "boost", icon: "bolt", label: "Number boost", what: "lift rows by a column: rating, sales" },
  { kind: "group", stage: "group", icon: "layers", label: "Group", what: "facets: one row per value" },
  { kind: "measure", stage: "group", icon: "perf", label: "Measure", what: "count, sum, average, min, max" },
  { kind: "sort", stage: "order", icon: "up", label: "Sort", what: "by a column instead of the score" },
  { kind: "column", stage: "output", icon: "eye", label: "Column", what: "a field to show" },
];
const STAGES = [
  { id: "source", title: "Rows from", icon: "table", hint: "Drop a table here, or click one on the left." },
  { id: "find", title: "Find", icon: "search", hint: "Drop a text field to search its words, or a vector field to search by meaning." },
  { id: "filter", title: "Keep rows where", icon: "filter", hint: "Drop any field to filter on it." },
  { id: "boost", title: "Boost", icon: "bolt", hint: "Drop a number or a time to lift the rows with more of it, or the newest." },
  { id: "group", title: "Group and measure", icon: "layers", hint: "Drop a keyword or a number to count the rows of each value." },
  { id: "order", title: "Order", icon: "up", hint: "Drop a field to sort by it." },
  { id: "output", title: "Show", icon: "eye", hint: "Drop fields to show them; with none, every column." },
];
const MEASURES = [["count", "COUNT(*)"], ["distinct", "COUNT(DISTINCT …)"], ["sum", "SUM"], ["avg", "AVG"], ["min", "MIN"], ["max", "MAX"]];

const blank = (table) => ({ table, signals: [], fusion: "weighted", filters: [], any: false, boosts: [], group: null, sort: null, columns: [], score: true, limit: 20 });

// ------------------------------------------------------------ SQL
const num = (v) => { const x = Number(v); if (v === "" || v == null || !Number.isFinite(x)) throw new Error(`“${v ?? ""}” is not a number`); return String(x); };
const round = (x, d = 2) => String(Math.round(x * 10 ** d) / 10 ** d);

/** One value of a field, as SQL. */
function value(f, v) {
  if (isNum(f)) return num(v);
  if (v == null || v === "") throw new Error(`${f.name} needs a value`);
  return lit(v);
}

function condition(f, c) {
  const n = quoteName(f.name);
  switch (c.op) {
    case "matches": if (!c.value) throw new Error(`${f.name} needs words`); return `MATCH(${n}, ${lit(c.value)})`;
    case "in": { const vs = String(c.value || "").split(",").map((x) => x.trim()).filter(Boolean); if (!vs.length) throw new Error(`${f.name} needs values`); return `${n} IN (${vs.map((v) => value(f, v)).join(", ")})`; }
    case "prefix": return `PREFIX(${n}, ${lit(c.value || "")})`;
    case "null": return `${n} IS NULL`;
    case "notnull": return `${n} IS NOT NULL`;
    case "true": return `${n} = TRUE`;
    case "false": return `${n} = FALSE`;
    case "between": return `${n} BETWEEN ${value(f, c.value)} AND ${value(f, c.value2)}`;
    default: return `${n} ${c.op} ${value(f, c.value)}`;
  }
}

/** The SELECT a state stands for: {sql, params, ranked, grouped, notes}. */
export function toSql(st) {
  const t = tableOf(st.table);
  if (!t) throw new Error("Pick a table");
  const fields = t.short?.fields || [];
  const F = (name) => { const f = fields.find((x) => x.name === name); if (!f) throw new Error(`${st.table} has no field ${name}`); return f; };
  const params = {}, notes = [];
  const grouped = !!(st.group && (st.group.by.length || st.group.measures.length));

  // finding: one score per signal, and MATCH for the words that must be there
  const where = [], scores = [];
  let pn = 0;
  for (const s of st.signals) {
    const f = F(s.field);
    if (s.kind === "words") {
      if (!s.text?.trim()) continue;
      const opt = s.fuzzy ? ", fuzzy => 'auto'" : "";
      if (s.must || grouped) where.push(`MATCH(${quoteName(f.name)}, ${lit(s.text)}${opt})`);
      scores.push({ s, sql: `bm25(${quoteName(f.name)}, ${lit(s.text)}${opt})` });
    } else {
      if (!s.text?.trim()) continue;
      const p = pn++ ? `q${pn}` : "q";
      if (f.embed) params[p] = { embed: s.text, field: f.name };
      else { try { params[p] = JSON.parse(s.text); } catch { throw new Error(`${f.name} has no model to embed text: give a vector, such as [0.1, 0.2, …]`); } }
      if (grouped) notes.push("Grouping counts the rows the filters keep: a meaning search ranks, it does not filter, so it is left out.");
      else scores.push({ s, sql: `similarity(${quoteName(f.name)}, :${p})` });
    }
  }
  const conds = [];
  for (const c of st.filters) { const f = F(c.field); conds.push(condition(f, c)); }
  if (conds.length) where.push(st.any && conds.length > 1 ? `(${conds.join(" OR ")})` : conds.join(" AND "));

  // ranking: the scores fused, times the boosts
  let rank = null;
  if (!grouped) {
    if (scores.length === 1) rank = scores[0].sql;
    else if (scores.length > 1 && st.fusion === "weighted") {
      const total = scores.reduce((a, x) => a + (x.s.w ?? 1), 0) || 1;
      rank = scores.map((x) => `${round((x.s.w ?? 1) / total)} * tmm(${x.sql})`).join(" + ");
    } else if (scores.length > 1) rank = `rrf(${scores.map((x) => x.sql).join(", ")})`;
    const factors = [];
    for (const b of st.boosts) {
      const f = F(b.field);
      if (b.kind === "closeness") {
        if (isTime(f)) factors.push(`decay(${quoteName(f.name)}, now(), ${Math.round(Number(b.days || 30) * 86400)})`);
        else factors.push(`decay(${quoteName(f.name)}, ${num(b.origin)}, ${num(b.scale)})`);
      } else factors.push(`(1 + ${num(b.w ?? 0.1)} * coalesce(${quoteName(f.name)}, 0))`);
    }
    if (factors.length) rank = [rank && / \+ /.test(rank) ? `(${rank})` : rank, ...factors].filter(Boolean).join(" * ");
  } else if (st.boosts.length) notes.push("Boosts rank rows; a grouped query counts them, so they are left out.");

  let sql;
  if (grouped) {
    const by = st.group.by.map((n) => quoteName(F(n).name));
    const ms = st.group.measures.map((m) => measureSql(m, F));
    if (!by.length && !ms.length) throw new Error("Group by a field or add a measure");
    const cols = [...by, ...ms.map((m) => `${m.sql} AS ${m.alias}`)];
    sql = `SELECT ${cols.join(", ")}\nFROM ${quoteName(t.table)}`;
    if (where.length) sql += `\nWHERE ${where.join("\n  AND ")}`;
    if (by.length) sql += `\nGROUP BY ${by.join(", ")}`;
    if (by.length && st.group.min > 1) sql += `\nHAVING COUNT(*) >= ${num(st.group.min)}`;
    const order = st.sort && (st.group.by.includes(st.sort.field) || ms.some((m) => m.alias === st.sort.field)) ? `${st.group.by.includes(st.sort.field) ? quoteName(st.sort.field) : st.sort.field} ${st.sort.dir.toUpperCase()}` : ms.length ? `${ms[0].alias} DESC` : null;
    if (order && by.length) sql += `\nORDER BY ${order}`;
    if (by.length) sql += `\nLIMIT ${num(st.limit || 20)}`;
  } else {
    const cols = st.columns.map((n) => quoteName(F(n).name));
    const ranked = !!rank && !st.sort;
    if (ranked && st.score) cols.push("score()");
    sql = `SELECT ${st.columns.length ? cols.join(", ") : ranked && st.score ? "*, score()" : "*"}\nFROM ${quoteName(t.table)}`;
    if (where.length) sql += `\nWHERE ${where.join("\n  AND ")}`;
    if (st.sort) {
      const f = F(st.sort.field);
      sql += `\nORDER BY ${quoteName(f.name)} ${st.sort.dir.toUpperCase()}${f.name !== t.short.key && t.short.key ? `, ${quoteName(t.short.key)}` : ""}`;
      if (rank) notes.push("Sorted by a column, so the scores only find rows: words that must match filter, the rest is left out.");
    } else if (rank) sql += `\nORDER BY ${rank} DESC`;
    sql += `\nLIMIT ${num(st.limit || 20)}`;
  }
  return { sql: sql + ";", params, ranked: !!rank && !st.sort && !grouped, grouped, notes };
}

function measureSql(m, F) {
  if (m.fn === "count") return { sql: "COUNT(*)", alias: "n" };
  const f = F(m.field);
  const n = quoteName(f.name);
  const alias = `${m.fn === "distinct" ? "distinct" : m.fn}_${f.name}`.replace(/\W/g, "_");
  return { sql: m.fn === "distinct" ? `COUNT(DISTINCT ${n})` : `${m.fn.toUpperCase()}(${n})`, alias };
}

// ------------------------------------------------------------ the view
function render(page, r) {
  let st = store.get("builder.state", null);
  let drag = null;              // {kind: "field"|"block"|"table", value}
  let live = store.get("builder.live", true);
  let last = null, counts = {}, running = null;
  const stats = new Map();      // "table.field" -> {min, max, avg} or {top: [[v, n]]}
  const totals = new Map();     // table -> rows

  const palette = h("aside", { class: "bld-pal" });
  const flow = h("div", { class: "bld-flow" });
  const sqlBox = h("div", { class: "bld-sql" });
  const notesBox = h("div", { class: "bld-notes" });
  const out = h("div", { class: "bld-out" });
  const runBtn = h("button", { class: "btn primary", onclick: () => run(true) }, icon("play"), "Run");
  const right = h("div", { class: "bld-right" },
    h("section", { class: "card" },
      h("div", { class: "hd" }, h("h3", {}, "SQL"), h("span", { class: "sub" }, "written as you build"),
        h("div", { class: "tools" },
          h("button", { class: "btn sm ghost", title: "Copy the SQL", onclick: () => last?.q && copy(last.q.sql, "SQL copied") }, icon("copy"), "Copy"),
          h("button", { class: "btn sm", title: "Open this query in the SQL workbench", onclick: openInWorkbench }, icon("sql"), "Edit as SQL"))),
      h("div", { class: "bd" }, sqlBox, notesBox),
      h("div", { class: "ft" }, runBtn, h("label", { class: "row nw small dim", style: { gap: "6px", cursor: "pointer" } }, h("input", { type: "checkbox", checked: live, onchange: (e) => { live = e.target.checked; store.set("builder.live", live); if (live) run(); } }), "run as I build"), h("span", { class: "sep" }), h("button", { class: "btn sm ghost", title: "Start again on this table", onclick: () => { st = blank(st.table); changed(); } }, icon("refresh"), "Clear"))),
    out);
  page.append(h("div", { class: "bld" }, palette, h("div", { class: "bld-canvas" }, flow), right));

  const save = debounce(() => store.set("builder.state", st), 300);
  const runSoon = debounce(() => { if (live) run(); }, 450);
  function changed(redraw = true) { save(); if (redraw) drawFlow(); drawSql(); runSoon(); }

  // ------------------------------------------------------------ dragging
  const startDrag = (kind, value) => (e) => {
    drag = { kind, value };
    e.dataTransfer.effectAllowed = "copy";
    e.dataTransfer.setData("text/plain", kind === "field" ? quoteName(value) : kind === "table" ? quoteName(value) : value);
    document.body.classList.add("bld-dragging");
    for (const el of flow.querySelectorAll(".stage")) el.classList.toggle("can", accepts(el.dataset.stage));
  };
  const endDrag = () => { drag = null; document.body.classList.remove("bld-dragging"); for (const el of flow.querySelectorAll(".stage")) el.classList.remove("can", "over"); };
  addEventListener("dragend", endDrag);
  const field = (name) => (tableOf(st.table)?.short?.fields || []).find((f) => f.name === name);
  function accepts(stage) {
    if (!drag) return false;
    if (drag.kind === "table") return stage === "source";
    if (drag.kind === "block") return BLOCKS.find((b) => b.kind === drag.value)?.stage === stage;
    const f = field(drag.value);
    return { source: false, find: searchable(f) || f?.type === "vector", filter: filterable(f), boost: isNum(f) || isTime(f), group: groupable(f) || isNum(f), order: sortableField(f), output: showable(f) }[stage];
  }
  function dropOn(stage) {
    if (!drag || !accepts(stage)) return;
    const d = drag; endDrag();
    if (d.kind === "table") return useTable(d.value);
    if (d.kind === "block") return addBlock(d.value);
    addField(stage, d.value);
  }

  // ------------------------------------------------------------ adding
  const first = (pred) => (tableOf(st.table)?.short?.fields || []).find(pred);
  function addBlock(kind) {
    const pick = { words: searchable, meaning: (f) => f.type === "vector", filter: filterable, closeness: (f) => isTime(f) || isNum(f), number: isNum, group: groupable, measure: isNum, sort: sortableField, column: (f) => showable(f) && !st.columns.includes(f.name) }[kind];
    const f = first(pick);
    if (!f && kind !== "measure") { toast(`${st.table} has no field for ${BLOCKS.find((b) => b.kind === kind).label.toLowerCase()}`); return; }
    if (kind === "measure") { ensureGroup().measures.push({ id: uid(), fn: f ? "avg" : "count", field: f?.name }); return changed(); }
    addField(BLOCKS.find((b) => b.kind === kind).stage, f.name, kind);
  }
  function ensureGroup() { return (st.group ||= { by: [], measures: [{ id: uid(), fn: "count" }], min: 0 }); }
  function addField(stage, name, kind) {
    const f = field(name);
    if (stage === "find") st.signals.push(f.type === "vector" ? { id: uid(), kind: "meaning", field: name, text: "", w: 1 } : { id: uid(), kind: "words", field: name, text: "", must: true, fuzzy: false, w: 1 });
    else if (stage === "filter") { st.filters.push({ id: uid(), field: name, op: operators(f)[0][0], value: "", value2: "" }); loadStats(f); }
    else if (stage === "boost") {
      if (isTime(f) || kind === "closeness") { st.boosts.push({ id: uid(), kind: "closeness", field: name, days: 30, origin: "", scale: "" }); loadStats(f).then(() => fillCloseness()); }
      else st.boosts.push({ id: uid(), kind: "number", field: name, w: 0.1 });
    } else if (stage === "group") {
      const g = ensureGroup();
      if (groupable(f) && kind !== "measure" && !g.by.includes(name)) g.by.push(name);
      else if (isNum(f)) g.measures.push({ id: uid(), fn: "avg", field: name });
    } else if (stage === "order") st.sort = { field: name, dir: isNum(f) || isTime(f) ? "desc" : "asc" };
    else if (stage === "output" && !st.columns.includes(name)) st.columns.push(name);
    changed();
    requestAnimationFrame(() => flow.querySelector(`.stage[data-stage="${stage}"] .item:last-of-type input:not([type=checkbox]):not([type=range])`)?.focus());
  }
  function fillCloseness() {
    for (const b of st.boosts) {
      const s = stats.get(`${st.table}.${b.field}`);
      if (b.kind === "closeness" && s && !isTime(field(b.field)) && b.origin === "") { b.origin = s.max; b.scale = round(Math.max((s.max - s.min) / 4, 1e-9), 3); }
    }
    changed();
  }
  function useTable(name) {
    if (name === st.table) return;
    store.set(`builder.state.${st.table}`, st);
    st = store.get(`builder.state.${name}`, null) || starter(tableOf(name));
    last = null; clear(out);
    drawPalette(); changed(); loadTotal();
    for (const c of st.filters) { const f = field(c.field); if (f) loadStats(f); }
  }
  reopen = (name) => { if (tableOf(name)) useTable(name); history.replaceState(null, "", "#/builder"); };

  // ------------------------------------------------------------ the field values the editors suggest
  async function loadStats(f) {
    const k = `${st.table}.${f.name}`;
    if (stats.has(k) || !(isNum(f) || isTime(f) || f.type === "keyword")) return stats.get(k);
    stats.set(k, null);
    const n = quoteName(f.name), t = quoteName(st.table);
    try {
      if (f.type === "keyword") {
        const j = await api("/v1/query", { body: { sql: `SELECT ${n}, COUNT(*) AS n FROM ${t} GROUP BY ${n} ORDER BY n DESC LIMIT 12` } });
        stats.set(k, { top: j.rows });
      } else {
        const j = await api("/v1/query", { body: { sql: `SELECT MIN(${n}), MAX(${n})${isNum(f) ? `, AVG(${n})` : ""} FROM ${t}` } });
        const [min, max, avg] = j.rows[0];
        stats.set(k, { min, max, avg });
      }
      drawFlow();
    } catch { stats.delete(k); }
    return stats.get(k);
  }
  async function loadTotal() {
    if (totals.has(st.table)) return;
    try { const j = await api("/v1/query", { body: { sql: `SELECT COUNT(*) FROM ${quoteName(st.table)}` } }); totals.set(st.table, j.rows[0][0]); drawFlow(); } catch { /* the funnel shows less */ }
  }

  // ------------------------------------------------------------ the palette
  function drawPalette() {
    const ts = cached();
    const t = tableOf(st.table);
    const fq = h("input", { class: "input sm", placeholder: "Filter fields…", style: { width: "100%" } });
    const list = h("div", { class: "bld-fields" });
    const drawFields = () => clear(list, (t?.short?.fields || []).filter((f) => !fq.value || f.name.toLowerCase().includes(fq.value.toLowerCase())).map((f) => {
      const el = h("div", { class: "fi", draggable: "true", tabIndex: 0, title: `${f.name}: drag onto a stage, or click for what it can do`, ondragstart: startDrag("field", f.name) },
        typeBadge(f), h("span", { class: "grow ellipsis" }, f.name), f.multi ? h("span", { class: "badge" }, "[]") : null, t.short.key === f.name ? h("span", { class: "badge accent" }, "key") : null, icon("grip", "drag"));
      el.onclick = () => fieldMenu(el, f);
      el.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fieldMenu(el, f); } };
      return el;
    }));
    fq.oninput = drawFields;
    drawFields();
    clear(palette,
      h("div", { class: "wb-sh" }, h("b", {}, "Tables")),
      h("div", { class: "bld-tables" }, ts.map((x) => h("button", { class: "tt-h", "aria-pressed": String(x.table === st.table), draggable: "true", title: "Click to build on it, or drag it onto “Rows from”", ondragstart: startDrag("table", x.table), onclick: () => useTable(x.table) }, icon("table"), h("span", { class: "grow ellipsis" }, x.table), h("span", { class: "small muted" }, (x.short?.fields || []).length)))),
      h("div", { class: "wb-sh", style: { marginTop: "14px" } }, h("b", {}, "Blocks"), h("span", { class: "small muted" }, "drag or click")),
      h("div", { class: "bld-blocks" }, BLOCKS.map((b) => {
        const el = h("button", { class: "blk", draggable: "true", dataset: { stage: b.stage }, title: b.what, ondragstart: startDrag("block", b.kind), onclick: () => addBlock(b.kind) }, icon(b.icon), h("span", {}, h("b", {}, b.label), h("span", { class: "small muted" }, b.what)));
        return el;
      })),
      h("div", { class: "wb-sh", style: { marginTop: "14px" } }, h("b", {}, "Fields"), h("span", { class: "small muted" }, t ? `of ${t.table}` : "")), fq, list);
  }
  function fieldMenu(el, f) {
    const items = [];
    if (searchable(f)) items.push({ label: "Search its words", icon: "search", onclick: () => addField("find", f.name) });
    if (f.type === "vector") items.push({ label: "Search by meaning", icon: "sparkle", onclick: () => addField("find", f.name) });
    if (filterable(f)) items.push({ label: "Filter on it", icon: "filter", onclick: () => addField("filter", f.name) });
    if (isNum(f) || isTime(f)) items.push({ label: isTime(f) ? "Boost the newest" : "Boost by it", icon: "bolt", onclick: () => addField("boost", f.name) });
    if (groupable(f)) items.push({ label: "Group by it", icon: "layers", onclick: () => addField("group", f.name) });
    if (isNum(f)) items.push({ label: "Measure it", icon: "perf", onclick: () => { ensureGroup().measures.push({ id: uid(), fn: "avg", field: f.name }); changed(); } });
    if (sortableField(f)) items.push({ label: "Sort by it", icon: "up", onclick: () => addField("order", f.name) });
    if (showable(f)) items.push({ label: "Show it", icon: "eye", onclick: () => addField("output", f.name) });
    menu(el, items);
  }

  // ------------------------------------------------------------ the flow
  function drawFlow() {
    const a = document.activeElement;
    const inputs = () => [...flow.querySelectorAll("input, select, button")];
    const keep = a && flow.contains(a) ? { i: inputs().indexOf(a), s: a.selectionStart, e: a.selectionEnd } : null;
    let q = null;
    try { q = toSql(st); } catch { /* drawSql says what is missing */ }
    const grouped = q?.grouped;
    const total = totals.get(st.table);
    const stages = STAGES.map((sd) => stage(sd, q));
    const nodes = [];
    stages.forEach((el, i) => {
      nodes.push(el);
      const sd = STAGES[i];
      const link = connector(sd.id, total, grouped, q);
      if (link) nodes.push(link);
    });
    clear(flow, nodes);
    if (keep) { const el = inputs()[keep.i]; if (el) { el.focus(); try { if (keep.s != null) el.setSelectionRange(keep.s, keep.e); } catch { /* not a text input */ } } }
  }
  /** The counts between the stages, after a run, without redrawing the stages. */
  function drawConns() {
    let q = null;
    try { q = toSql(st); } catch { /* nothing to count */ }
    const total = totals.get(st.table);
    for (const el of flow.querySelectorAll(".conn")) el.replaceWith(connector(el.dataset.after, total, q?.grouped, q));
  }
  function connector(after, total, grouped, q) {
    if (after === "output") return null;
    let n = null, label = "";
    const stt = last?.j?.stats;
    if (after === "source") { n = total; label = "rows"; }
    else if (after === "filter" && stt?.filter_docs != null && (st.filters.length || st.signals.some((s) => s.kind === "words" && (s.must || grouped)))) { n = stt.filter_docs; label = "pass"; }
    else if (after === "boost" && stt?.scored && q?.ranked) { n = stt.scored; label = "scored"; }
    else if (after === "group" && grouped && last?.j?.rows) { n = last.j.rows.length; label = "groups"; }
    else if (after === "order" && last?.j?.rows && !grouped) { n = last.j.rows.length; label = "shown"; }
    const share = n != null && total ? Math.max(0.02, Math.min(1, n / total)) : null;
    return h("div", { class: "conn" + (n != null ? " on" : ""), dataset: { after } },
      h("span", { class: "rail" }),
      n != null ? h("span", { class: "cnt", title: total ? `${fmt.pct(n / total)} of the table` : "" }, h("b", {}, fmt.n(n)), " ", label, share != null ? h("span", { class: "fun" }, h("span", { style: { width: share * 100 + "%" } })) : null) : null);
  }
  function stage(sd, q) {
    const body = stageBody(sd.id, q);
    const has = body && body.length !== 0;
    const el = h("section", { class: "stage" + (has ? "" : " st-empty"), dataset: { stage: sd.id } },
      h("div", { class: "st-hd" }, h("span", { class: "st-ic" }, icon(sd.icon)), h("b", {}, sd.title), stageTools(sd.id)),
      has ? h("div", { class: "st-bd" }, body) : h("div", { class: "st-hint small muted" }, sd.hint));
    el.ondragover = (e) => { if (accepts(sd.id)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; el.classList.add("over"); } };
    el.ondragleave = (e) => { if (!el.contains(e.relatedTarget)) el.classList.remove("over"); };
    el.ondrop = (e) => { e.preventDefault(); el.classList.remove("over"); dropOn(sd.id); };
    return el;
  }
  function stageTools(id) {
    const add = (kind, label) => h("button", { class: "btn sm ghost", title: `Add: ${label}`, onclick: () => addBlock(kind) }, icon("plus"), label);
    return h("span", { class: "st-tools" }, {
      find: [add("words", "words"), add("meaning", "meaning")],
      filter: [st.filters.length > 1 ? seg([[false, "all of"], [true, "any of"]], st.any, (v) => { st.any = v; changed(); }) : null, add("filter", "filter")],
      boost: [add("closeness", "recency"), add("number", "number")],
      group: [st.group ? h("button", { class: "btn sm ghost", title: "Stop grouping", onclick: () => { st.group = null; changed(); } }, icon("x"), "ungroup") : add("group", "group")],
      order: [],
      output: [h("label", { class: "row nw small dim", title: "Show each row's score", style: { gap: "5px" } }, h("input", { type: "checkbox", checked: st.score, onchange: (e) => { st.score = e.target.checked; changed(false); } }), "score"), add("column", "column")],
    }[id] || []);
  }
  const fieldSel = (cur, pred, onChange) => {
    const fs = (tableOf(st.table)?.short?.fields || []).filter(pred);
    return h("select", { class: "select sm", onchange: (e) => onChange(e.target.value) }, fs.map((f) => h("option", { value: f.name, selected: f.name === cur }, `${typeMeta(f).glyph}  ${f.name}`)));
  };
  const rm = (list, it) => h("button", { class: "btn ghost icon sm x", title: "Remove", onclick: () => { list.splice(list.indexOf(it), 1); changed(); } }, icon("x"));

  function stageBody(id, q) {
    const t = tableOf(st.table);
    if (id === "source") return t ? [h("div", { class: "item src", draggable: "false" }, icon("table"), h("b", {}, t.table), h("span", { class: "small muted" }, `${fmt.plural((t.short?.fields || []).length, "field")} · key ${t.short?.key || "none"}`), totals.has(t.table) ? h("span", { class: "badge" }, fmt.plural(totals.get(t.table), "row")) : null)] : [];
    if (id === "find") {
      const items = st.signals.map((s, i) => {
        const f = field(s.field);
        const color = SERIES[i % SERIES.length];
        const text = h("input", { class: "input sm grow", value: s.text, placeholder: s.kind === "words" ? "words to find, such as quiet charger" : f?.embed ? "a text the server embeds" : "a vector, such as [0.1, 0.2, …]", oninput: () => { s.text = text.value; changed(false); } });
        return h("div", { class: "item sig", style: { "--c": color } },
          h("div", { class: "row nw" }, h("span", { class: "kind" }, icon(s.kind === "words" ? "search" : "sparkle"), s.kind === "words" ? "Words in" : "Meaning of"),
            fieldSel(s.field, s.kind === "words" ? searchable : (x) => x.type === "vector", (v) => { s.field = v; changed(); }), text, rm(st.signals, s)),
          h("div", { class: "row small dim opts" },
            s.kind === "words" ? [
              h("label", { class: "row nw", title: "Leave out the rows without the words (MATCH)" }, h("input", { type: "checkbox", checked: s.must, onchange: (e) => { s.must = e.target.checked; changed(); } }), "must match"),
              h("label", { class: "row nw", title: "Also match words one or two edits away" }, h("input", { type: "checkbox", checked: s.fuzzy, onchange: (e) => { s.fuzzy = e.target.checked; changed(false); } }), "allow typos")] : h("span", { class: "muted" }, f?.embed ? `embedded by ${f.embed.model}` : "no model: give a vector"),
            st.signals.length > 1 && st.fusion === "weighted" ? weight(s) : null));
      });
      if (st.signals.length > 1) items.push(h("div", { class: "row fuse" }, icon("route"), h("span", { class: "small dim" }, "Fuse the rankings by"),
        seg([["weighted", "weighted scores", "A weighted sum of the scores, each divided by its best row's from the least it can be (tmm): better than RRF once the weights are tuned on a few graded queries (the Relevance lab tunes them)"], ["rrf", "rank (RRF)", "Reciprocal rank fusion: a row near the top of several lists wins; no weights to tune"]], st.fusion, (v) => { st.fusion = v; changed(); })));
      return items;
    }
    if (id === "filter") return st.filters.map((c) => filterItem(c));
    if (id === "boost") return st.boosts.map((b) => {
      const f = field(b.field);
      if (b.kind === "closeness") {
        const s = stats.get(`${st.table}.${b.field}`);
        return h("div", { class: "item" }, h("div", { class: "row nw" }, h("span", { class: "kind" }, icon("clock"), isTime(f) ? "Newest by" : "Closest to"),
          fieldSel(b.field, (x) => isTime(x) || isNum(x), (v) => { b.field = v; b.origin = ""; b.scale = ""; loadStats(field(v)).then(fillCloseness); changed(); }),
          isTime(f) ? [h("span", { class: "small dim" }, "halves every"), numIn(b, "days", { min: 0.01, step: 1, w: "70px" }), h("span", { class: "small dim" }, "days")]
            : [numIn(b, "origin", { w: "90px", ph: "value" }), h("span", { class: "small dim" }, "halves every"), numIn(b, "scale", { w: "80px", ph: "distance" })],
          h("span", { class: "sep" }), rm(st.boosts, b)),
          !isTime(f) && s ? h("div", { class: "small muted opts" }, `${b.field} runs from ${fmt.n(s.min)} to ${fmt.n(s.max)}`) : null);
      }
      return h("div", { class: "item" }, h("div", { class: "row nw" }, h("span", { class: "kind" }, icon("bolt"), "More"),
        fieldSel(b.field, isNum, (v) => { b.field = v; changed(); }), h("span", { class: "small dim" }, "lifts by"),
        h("input", { type: "range", min: 0, max: 1, step: 0.01, value: b.w, style: { width: "110px" }, oninput: (e) => { b.w = Number(e.target.value); e.target.nextSibling.textContent = round(b.w); e.target.closest(".item").querySelector(".formula").textContent = `× (1 + ${round(b.w)} × ${b.field})`; changed(false); } }), h("span", { class: "mono small" }, round(b.w)),
        h("span", { class: "sep" }), rm(st.boosts, b)),
        h("div", { class: "small muted opts mono formula" }, `× (1 + ${round(b.w)} × ${b.field})`));
    });
    if (id === "group") {
      if (!st.group) return [];
      const g = st.group;
      return [
        h("div", { class: "row" }, h("span", { class: "small dim" }, "one row per"), g.by.length ? g.by.map((n) => h("span", { class: "chip on", title: "Remove" }, typeBadge(field(n) || { type: "keyword" }), n, h("span", { class: "x", onclick: () => { g.by.splice(g.by.indexOf(n), 1); changed(); } }, "×"))) : h("span", { class: "small muted" }, "nothing: one row of totals")),
        ...g.measures.map((m) => h("div", { class: "item" }, h("div", { class: "row nw" }, h("span", { class: "kind" }, icon("perf"), "Measure"),
          h("select", { class: "select sm", onchange: (e) => { m.fn = e.target.value; if (m.fn !== "count" && !m.field) m.field = first(m.fn === "sum" || m.fn === "avg" ? isNum : groupable)?.name; changed(); } }, MEASURES.map(([v, l]) => h("option", { value: v, selected: v === m.fn }, l))),
          m.fn === "count" ? h("span", { class: "small muted" }, "the rows") : fieldSel(m.field, m.fn === "sum" || m.fn === "avg" ? isNum : (x) => groupable(x) || isNum(x), (v) => { m.field = v; changed(); }),
          h("span", { class: "sep" }), rm(g.measures, m)))),
        g.by.length ? h("div", { class: "row small dim" }, "keep groups of at least", numIn(g, "min", { min: 0, step: 1, w: "70px", ph: "1" }), "rows") : null];
    }
    if (id === "order") {
      const items = [];
      if (st.sort) {
        items.push(h("div", { class: "item" }, h("div", { class: "row nw" }, h("span", { class: "kind" }, icon("up"), "Sort by"),
          q?.grouped ? h("select", { class: "select sm", onchange: (e) => { st.sort.field = e.target.value; changed(); } }, [...st.group.by, ...st.group.measures.map((m) => { try { return measureSql(m, (n) => field(n)).alias; } catch { return null; } })].filter(Boolean).map((n) => h("option", { value: n, selected: n === st.sort.field }, n)))
            : fieldSel(st.sort.field, sortableField, (v) => { st.sort.field = v; changed(); }),
          seg([["asc", "low to high"], ["desc", "high to low"]], st.sort.dir, (v) => { st.sort.dir = v; changed(); }),
          h("span", { class: "sep" }), h("button", { class: "btn ghost icon sm x", title: "Back to the score", onclick: () => { st.sort = null; changed(); } }, icon("x")))));
      } else items.push(h("div", { class: "row small dim" }, icon(q?.ranked ? "sparkle" : "info"), q?.ranked ? "Best first by the score above." : q?.grouped ? "The largest groups first." : "In the table's order.",
        h("button", { class: "btn sm ghost", onclick: () => addBlock("sort") }, "sort by a column")));
      items.push(h("div", { class: "row small dim" }, "at most", numIn(st, "limit", { min: 1, step: 1, w: "80px" }), q?.grouped ? "groups" : "rows"));
      return items;
    }
    if (id === "output") {
      if (q?.grouped) return [h("div", { class: "small muted" }, "A grouped query shows the groups and their measures.")];
      if (!st.columns.length) return [];
      const list = h("div", { class: "row cols" }, st.columns.map((n) => h("span", { class: "chip on", draggable: "true", dataset: { name: n }, title: "Drag to reorder" }, icon("grip", "drag"), typeBadge(field(n) || { type: "keyword" }), n, h("span", { class: "x", title: "Remove", onclick: () => { st.columns.splice(st.columns.indexOf(n), 1); changed(); } }, "×"))));
      sortable(list, (kids) => { st.columns = kids.map((k) => k.dataset.name); changed(); });
      return [list];
    }
    return [];
  }
  function numIn(obj, key, { min, step = "any", w = "90px", ph = "" } = {}) {
    return h("input", { class: "input sm num", type: "number", min, step, value: obj[key] ?? "", placeholder: ph, style: { width: w }, oninput: (e) => { obj[key] = e.target.value === "" ? "" : Number(e.target.value); changed(false); } });
  }
  function weight(s) {
    const share = (x) => fmt.pct((x.w ?? 1) / (st.signals.reduce((a, y) => a + (y.w ?? 1), 0) || 1));
    return h("span", { class: "row nw", style: { marginLeft: "auto" } }, "weight",
      h("input", { type: "range", min: 0, max: 1, step: 0.05, value: s.w ?? 1, style: { width: "100px" }, oninput: (e) => {
        s.w = Number(e.target.value);
        for (const el of flow.querySelectorAll(".wpct")) el.textContent = share(st.signals.find((x) => x.id === el.dataset.id));
        changed(false);
      } }),
      h("span", { class: "mono wpct", dataset: { id: s.id } }, share(s)));
  }
  function filterItem(c) {
    const f = field(c.field);
    if (!f) return h("div", { class: "item" }, h("span", { class: "error small" }, `${c.field} is not a field of ${st.table}`), rm(st.filters, c));
    const ops = operators(f);
    const s = stats.get(`${st.table}.${f.name}`);
    const needs = !["null", "notnull", "true", "false"].includes(c.op);
    const inp = (key, ph) => {
      const type = isNum(f) ? "number" : isTime(f) ? "date" : "text";
      const el = h("input", { class: "input sm" + (isNum(f) ? " num" : ""), type, step: "any", value: c[key] ?? "", placeholder: ph, style: { width: isNum(f) ? "100px" : isTime(f) ? "150px" : "auto" }, oninput: () => { c[key] = el.value; changed(false); syncRange(); } });
      return el;
    };
    const a = needs ? inp("value", c.op === "in" ? "a, b, c" : c.op === "matches" ? "words" : isNum(f) && s ? `${fmt.n(s.min)} – ${fmt.n(s.max)}` : "value") : null;
    const b = c.op === "between" ? inp("value2", "and") : null;
    let range = null;
    const syncRange = () => { if (range && c.value !== "") range.value = c.value; };
    if (needs && isNum(f) && s && s.max > s.min && c.op !== "between") {
      const step = f.type === "int" ? 1 : Math.max((s.max - s.min) / 200, 1e-6);
      range = h("input", { type: "range", min: s.min, max: s.max, step, value: c.value === "" ? s.min : c.value, style: { width: "120px" }, title: `${fmt.n(s.min)} to ${fmt.n(s.max)}`, oninput: (e) => { c.value = f.type === "int" ? Number(e.target.value) : Number(round(Number(e.target.value), 2)); a.value = c.value; changed(false); } });
    }
    const top = needs && f.type === "keyword" && s?.top && ["=", "!=", "in"].includes(c.op) ? h("div", { class: "row vals" }, s.top.map(([v, n]) => h("button", { class: "chip", "aria-pressed": String(String(c.value).split(",").map((x) => x.trim()).includes(String(v))), title: `${fmt.n(n)} rows`, onclick: () => {
      if (c.op === "in") { const vs = String(c.value || "").split(",").map((x) => x.trim()).filter(Boolean); const i = vs.indexOf(String(v)); if (i >= 0) vs.splice(i, 1); else vs.push(String(v)); c.value = vs.join(", "); }
      else c.value = String(v);
      changed();
    } }, String(v), h("span", { class: "muted" }, fmt.si(n))))) : null;
    if (needs && f.type === "keyword" && !s) loadStats(f);
    return h("div", { class: "item" },
      h("div", { class: "row nw" }, h("span", { class: "kind" }, icon("filter"), "Where"),
        fieldSel(c.field, filterable, (v) => { c.field = v; const nf = field(v); if (!operators(nf).some(([o]) => o === c.op)) c.op = operators(nf)[0][0]; c.value = ""; c.value2 = ""; loadStats(nf); changed(); }),
        h("select", { class: "select sm", onchange: (e) => { c.op = e.target.value; changed(); } }, ops.map(([v, l]) => h("option", { value: v, selected: v === c.op }, l))),
        a, b ? h("span", { class: "small dim" }, "and") : null, b, range, h("span", { class: "sep" }), rm(st.filters, c)),
      top);
  }

  // ------------------------------------------------------------ SQL and running
  function drawSql() {
    clear(notesBox);
    try {
      const q = toSql(st);
      last = { ...last, q };
      clear(sqlBox, sqlBlock(q.sql));
      if (Object.keys(q.params).length) sqlBox.append(h("div", { class: "small muted", style: { marginTop: "6px" } }, "Parameters: ", Object.entries(q.params).map(([k, v]) => h("span", { class: "badge mono", style: { marginRight: "4px" } }, `:${k} = ${v.embed != null ? `embed(${JSON.stringify(v.embed)})` : JSON.stringify(v).slice(0, 40)}`))));
      for (const n of q.notes) notesBox.append(h("div", { class: "small note" }, icon("info"), n));
      runBtn.disabled = false;
    } catch (e) {
      last = { ...last, q: null };
      clear(sqlBox, h("div", { class: "small muted" }, e.message));
      runBtn.disabled = true;
    }
  }
  async function run(force = false) {
    const q = last?.q;
    if (!q) return;
    if (!force && running) running.abort();
    running = new AbortController();
    const mine = running;
    out.classList.add("loading");
    try {
      const j = await api("/v1/query", { body: { sql: q.sql, params: q.params }, signal: mine.signal });
      if (mine !== running) return;
      last = { ...last, j, ran: q.sql };
      clear(out, h("section", { class: "card" },
        h("div", { class: "hd" }, h("h3", {}, "Answer"), h("div", { class: "tools" }, j.rows?.length ? exportMenu(j, `strata-${st.table}`) : null)),
        h("div", { class: "bd", style: { paddingBottom: "4px" } }, answerStats(j)),
        h("div", { class: "bd flush" }, j.rows?.length ? h("div", { class: "tbl-wrap" }, grid(j, { max: 200 })) : empty("search", "No rows", "Loosen a filter, or try other words."))));
      drawConns();
    } catch (e) {
      if (e.name === "AbortError") return;
      last = { ...last, j: null };
      clear(out, h("div", { class: "error" }, e.message));
      drawConns();
    } finally { if (mine === running) { out.classList.remove("loading"); running = null; } }
  }
  function openInWorkbench() {
    const q = last?.q;
    if (!q) return;
    const p = Object.fromEntries(Object.entries(q.params).map(([k, v]) => [k, v.embed != null ? { kind: "embed", value: v.embed, field: v.field || "" } : { kind: "json", value: JSON.stringify(v) }]));
    location.hash = "#/sql?q=" + encodeURIComponent(q.sql) + (Object.keys(p).length ? "&p=" + encodeURIComponent(JSON.stringify(p)) : "");
  }

  // ------------------------------------------------------------ start
  clear(flow, h("div", { class: "small muted", style: { padding: "20px" } }, "Reading the tables…"));
  tables().then((ts) => {
    if (!ts.length) { clear(flow, empty("table", "No tables yet", "Import a file or create a table first.", h("a", { class: "btn primary", href: "#/import" }, icon("import"), "Import"))); return; }
    const want = r.params.get("table");
    if (want && tableOf(want)) st = store.get(`builder.state.${tableOf(want).table}`, null) || starter(tableOf(want));
    if (!st || !tableOf(st.table)) st = starter(ts[0]);
    drawPalette(); changed();
    loadTotal();
    for (const c of st.filters) { const f = field(c.field); if (f) loadStats(f); }
  }).catch((e) => clear(flow, failure(e, "The tables")));
  return () => { reopen = null; running?.abort(); removeEventListener("dragend", endDrag); if (st) store.set(`builder.state.${st.table}`, st); };

  /** A first query on a table: its words, its key and two fields to show. */
  function starter(t) {
    const s = blank(t.table);
    const fs = t.short?.fields || [];
    const text = fs.find(searchable);
    if (text) s.signals.push({ id: uid(), kind: "words", field: text.name, text: "", must: true, fuzzy: false, w: 1 });
    s.columns = fs.filter((f) => showable(f) && (f.name === t.short.key || f.type === "text" || isNum(f))).slice(0, 4).map((f) => f.name);
    return s;
  }
}
