// The schema: every table as a card in a diagram, with the links between
// them (references, edge tables, EDGES fields) and inside them (vectors
// embedded from text, generated and derived text, semantic IDs). Drag the
// cards to arrange them, click one for its fields, its CREATE TABLE and
// what to do with it. The designer builds a new table by dragging types
// into it, checks it with the server as you go, and creates it.

import { h, s, icon, clear, fmt, api, store, toast, copy, drawer, confirm, empty, seg, sqlBlock, debounce, failure, showTip, hideTip } from "../core.js";
import { tables, invalidate, cached, table as tableOf, TYPES, typeMeta, typeBadge, links } from "../catalog.js";
import { quoteName } from "../sqledit.js";
import { viewport, draggable } from "./topology.js";
import { SERIES } from "../charts.js";

export default { id: "schema", title: "Schema", icon: "schema", group: "Data", full: true, keywords: "tables fields diagram er entity relationship references edges embeddings create table designer ddl drop", render, onParams };

let pick = null;
function onParams(r) { if (pick) pick(r); }

const W = 256, HD = 44, ROW = 22, FT = 22;
const KIND = {
  ref: { color: "var(--s1)", label: "references" },
  edges: { color: "var(--s8)", label: "links to (EDGES)" },
  graph: { color: "var(--s8)", label: "edge table over" },
  embed: { color: "var(--s5)", label: "embedded into" },
  from: { color: "var(--s3)", label: "joined into" },
  derive: { color: "var(--s2)", label: "derived into" },
  sid: { color: "var(--s7)", label: "semantic IDs of" },
};
const optsLine = (sh) => [sh.edges && `edges ${sh.edges}`, sh.shard_by && `shard by ${sh.shard_by}`, sh.cluster_by && `cluster by ${sh.cluster_by}`, sh.retain && `keep ${sh.retain}`].filter(Boolean).join(" · ");

/** What a field's short JSON says beyond its name and type, in words. */
export function fieldNotes(f, key) {
  const out = [];
  if (f.name === key) out.push("primary key");
  if (f.references) out.push(`references ${f.references}`);
  if (f.not_null) out.push("not null");
  if (f.default !== undefined) out.push(`default ${f.default === "now" ? "now" : JSON.stringify(f.default)}`);
  if (f.from) out.push(`joins ${[].concat(f.from).join(", ")}`);
  if (f.embed) out.push(`embeds ${[].concat(f.embed.from).join(", ")} with ${f.embed.model}`);
  if (f.derive) out.push(`${f.derive.model} of ${f.derive.from}`);
  if (f.semantic_id) out.push(`semantic IDs of ${f.semantic_id.from}`);
  if (f.dim) out.push(`${f.dim} dimensions`);
  if (f.index) out.push(typeof f.index === "string" ? f.index : Object.keys(f.index)[0]);
  if (f.codes) out.push(`${f.codes} codes`);
  if (f.analyzer && f.analyzer !== "english") out.push(`${typeof f.analyzer === "string" ? f.analyzer : "custom"} analyzer`);
  if (f.positions === false) out.push("no positions");
  if (f.indexed === false) out.push("not indexed");
  if (f.stored === false) out.push("not stored");
  if (f.kind) out.push(f.kind);
  return out;
}

function render(page, r) {
  let mode = r.params.get("new") ? "design" : "diagram";
  const body = h("div", { class: "sch-body" });
  const modeSeg = seg([["diagram", "Diagram"], ["design", "Design a table"]], mode, (v) => { mode = v; draw(); });
  const bar = h("div", { class: "sch-bar" }, modeSeg, h("span", { class: "sep" }));
  page.append(h("div", { class: "sch" }, bar, body));
  let cleanup = null, design = null;

  function draw() {
    cleanup?.(); cleanup = null;
    modeSeg.set(mode);
    clear(bar, modeSeg, h("span", { class: "sep" }));
    if (mode === "diagram") cleanup = diagram(body, bar, { openDesigner });
    else designer(body, bar, { seed: design, onCreated: (name) => { design = null; mode = "diagram"; draw(); setTimeout(() => pick?.({ params: new URLSearchParams("table=" + name) }), 300); } });
  }
  function openDesigner(seed) { design = seed; mode = "design"; draw(); }
  pick = (rr) => {
    if (rr.params.get("new")) { openDesigner(null); history.replaceState(null, "", "#/schema"); return; }
    const t = rr.params.get("table");
    if (t && mode === "diagram") { body.openTable?.(t); history.replaceState(null, "", "#/schema"); }
  };
  draw();
  if (r.params.get("table")) setTimeout(() => pick(r), 400);
  return () => { cleanup?.(); pick = null; };
}

// ------------------------------------------------------------ the diagram
function diagram(host, bar, { openDesigner }) {
  const wrap = h("div", { class: "diagram sch-diagram" });
  const canvas = h("div", { style: { position: "absolute", inset: 0 } });
  const find = h("input", { class: "input sm", placeholder: "Find a field…", style: { width: "200px" } });
  let vp = null, ts = [], sizes = {}, focusT = null;
  bar.append(find,
    h("button", { class: "btn sm", title: "Arrange the tables again", onclick: () => { store.del("schema.pos"); paint(true); } }, icon("refresh"), "Arrange"),
    h("button", { class: "btn sm primary", onclick: () => openDesigner(null) }, icon("plus"), "New table"));
  const controls = h("div", { class: "controls" },
    h("button", { class: "btn icon sm", title: "Zoom in", onclick: () => vp?.zoom(1.25) }, icon("zin")),
    h("button", { class: "btn icon sm", title: "Zoom out", onclick: () => vp?.zoom(0.8) }, icon("zout")),
    h("button", { class: "btn icon sm", title: "Fit", onclick: () => vp?.fit() }, icon("fit")));
  const legend = h("div", { class: "legend-box" }, Object.entries(KIND).filter(([k]) => k !== "edges").map(([k, v]) => h("span", { class: "row nw", style: { gap: "5px" } },
    h("span", { style: { width: "18px", borderTop: `2px ${k === "graph" ? "dashed" : "solid"} ${v.color}` } }), v.label)), h("span", { class: "muted" }, "drag tables · scroll to zoom · click for details"));
  wrap.append(canvas, controls, legend);
  clear(host, wrap);
  host.openTable = open;
  find.oninput = debounce(() => paint(false), 120);

  tables(true).then(async (list) => {
    ts = list;
    if (!ts.length) { clear(host, empty("schema", "No tables yet", "Design one, or import a file.", h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => openDesigner(null) }, icon("plus"), "Design a table"), h("a", { class: "btn", href: "#/import" }, icon("import"), "Import a file")))); return; }
    paint(true);
    sizes = await tableSizes(ts);
    paint(false);
  }).catch((e) => clear(host, failure(e, "The tables")));

  function layout(all) {
    const saved = store.get("schema.pos", {});
    const refs = (t) => [...new Set((t.short?.fields || []).map((f) => f.references).filter((x) => x && x !== t.table))];
    const level = {}, visiting = new Set();
    const lv = (t) => {
      if (level[t.table] != null) return level[t.table];
      if (visiting.has(t.table)) return 0;
      visiting.add(t.table);
      const rs = refs(t).map((n) => all.find((x) => x.table === n)).filter(Boolean);
      return (level[t.table] = rs.length ? 1 + Math.max(...rs.map(lv)) : 0);
    };
    all.forEach(lv);
    const cols = {};
    for (const t of all) (cols[level[t.table]] ||= []).push(t);
    const pos = {};
    for (const [l, col] of Object.entries(cols).sort((a, b) => a[0] - b[0])) {
      if (+l > 0) col.sort((a, b) => avgY(a) - avgY(b));
      let y = 40;
      for (const t of col) { pos[t.table] = [40 + l * (W + 150), y]; y += height(t) + 48; }
    }
    function avgY(t) { const ys = refs(t).map((n) => pos[n]?.[1]).filter((v) => v != null); return ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 1e9; }
    for (const t of all) if (saved[t.table]) pos[t.table] = saved[t.table];
    return pos;
  }
  const height = (t) => HD + (t.short?.fields || []).length * ROW + (optsLine(t.short || {}) ? FT + 6 : 10);

  let pos = {};
  function paint(fit) {
    const old = vp?.state;
    pos = layout(ts);
    const svgEl = s("svg", { role: "img", "aria-label": "the tables and the links between them" });
    const root = s("g");
    svgEl.append(s("rect", { class: "bg", x: -5000, y: -5000, width: 10000, height: 10000, fill: "transparent" }), root);
    clear(canvas, svgEl);
    vp = viewport(svgEl, root);
    const gl = s("g", { class: "sch-links" }), gc = s("g");
    root.append(gl, gc);
    const q = find.value.trim().toLowerCase();
    const ls = links(ts);
    const cards = {};
    ts.forEach((t, i) => { cards[t.table] = tableCard(t, i, q); gc.append(cards[t.table]); });
    const drawLinks = () => { clear(gl); for (const [k, l] of ls.entries()) gl.append(linkPath(l, k, ls)); };
    drawLinks();
    for (const t of ts) {
      const g = cards[t.table];
      draggable(g, vp, (dx, dy) => { const p = pos[t.table]; p[0] += dx; p[1] += dy; g.setAttribute("transform", `translate(${p[0]},${p[1]})`); drawLinks(); },
        () => { const saved = store.get("schema.pos", {}); saved[t.table] = pos[t.table]; store.set("schema.pos", saved); });
      g.addEventListener("click", () => { if (!g.moved) open(t.table); });
      g.addEventListener("pointerenter", () => hot(t.table, true));
      g.addEventListener("pointerleave", () => hot(t.table, false));
    }
    function hot(name, on) {
      focusT = on ? name : null;
      svgEl.classList.toggle("focus", on);
      for (const el of gl.children) el.classList.toggle("hot", on && (el.dataset.a === name || el.dataset.b === name));
      for (const [n, g] of Object.entries(cards)) g.classList.toggle("hot", on && (n === name || ls.some((l) => (l.from.table === name && l.to.table === n) || (l.to.table === name && l.from.table === n))));
    }
    if (old && !fit) vp.state = old;
    else requestAnimationFrame(() => vp.fit(30));
  }

  function rowY(table, field) {
    const t = tableOf(table);
    const i = (t?.short?.fields || []).findIndex((f) => f.name === field);
    return i < 0 ? HD / 2 : HD + i * ROW + ROW / 2;
  }
  function linkPath(l, k, all) {
    const A = pos[l.from.table], B = pos[l.to.table];
    if (!A || !B) return s("g");
    const kind = KIND[l.kind] || KIND.ref;
    const g = s("g", { class: "lk " + l.kind, dataset: { a: l.from.table, b: l.to.table }, style: { "--c": kind.color } });
    let d, ends = [];
    const same = l.from.table === l.to.table;
    if (same && l.kind !== "edges" && l.kind !== "ref") {
      // a field made from others in the same row: an arc on the card's left
      const inTable = all.filter((x) => x.from.table === l.from.table && x.from.table === x.to.table && x.kind !== "edges" && x.kind !== "ref");
      const n = inTable.indexOf(l);
      const off = 14 + (n % 4) * 7;
      const y1 = A[1] + rowY(l.from.table, l.from.field), y2 = A[1] + rowY(l.to.table, l.to.field), x = A[0];
      d = `M${x},${y1} C${x - off * 2},${y1} ${x - off * 2},${y2} ${x},${y2}`;
      ends.push(arrow(x, y2, 1));
    } else if (same) {
      const y1 = A[1] + rowY(l.from.table, l.from.field), y2 = A[1] + rowY(l.to.table, l.to.field), x = A[0] + W;
      d = `M${x},${y1} C${x + 70},${y1} ${x + 70},${y2} ${x},${y2}`;
      ends.push(arrow(x, y2, -1));
    } else {
      const graph = l.kind === "graph";
      const y1 = A[1] + (graph ? HD / 2 : rowY(l.from.table, l.from.field)), y2 = B[1] + (graph ? HD / 2 : rowY(l.to.table, l.to.field));
      let x1, x2, c1, c2;
      if (A[0] + W + 30 < B[0]) { x1 = A[0] + W; x2 = B[0]; c1 = x1 + 70; c2 = x2 - 70; }
      else if (B[0] + W + 30 < A[0]) { x1 = A[0]; x2 = B[0] + W; c1 = x1 - 70; c2 = x2 + 70; }
      else { x1 = A[0] + W; x2 = B[0] + W; c1 = Math.max(x1, x2) + 80; c2 = c1; }
      d = `M${x1},${y1} C${c1},${y1} ${c2},${y2} ${x2},${y2}`;
      const dir1 = c1 > x1 ? 1 : -1, dir2 = c2 > x2 ? 1 : -1;
      if (graph) ends.push(arrow(x2, y2, -dir2));
      else { ends.push(crowFoot(x1, y1, dir1), oneEnd(x2, y2, dir2)); }
    }
    const tip = `${l.from.table}${l.from.field ? "." + l.from.field : ""} ${kind.label === "references" ? "references" : kind.label} ${l.to.table}${l.to.field ? "." + l.to.field : ""}${l.label && !["references", "links to", "joined into"].includes(l.label) ? ` (${l.label})` : ""}`;
    g.append(s("path", { class: "hit", d }), s("path", { class: "ln" + (l.kind === "graph" ? " dash" : ""), d }), ...ends);
    g.addEventListener("pointermove", (e) => showTip(e.clientX, e.clientY, [h("div", { class: "t" }, tip)]));
    g.addEventListener("pointerleave", hideTip);
    return g;
  }
  const arrow = (x, y, dir) => s("path", { class: "end", d: `M${x},${y} l${-dir * 7},-4 l0,8 z` });
  const crowFoot = (x, y, dir) => s("path", { class: "end stroke", d: `M${x + dir * 10},${y} L${x},${y - 5} M${x + dir * 10},${y} L${x},${y + 5} M${x + dir * 10},${y} L${x},${y}` });
  const oneEnd = (x, y, dir) => s("path", { class: "end stroke", d: `M${x + dir * 7},${y - 5} L${x + dir * 7},${y + 5}` });

  function tableCard(t, i, q) {
    const sh = t.short || {};
    const fs = sh.fields || [];
    const [x, y] = pos[t.table];
    const H = height(t);
    const color = SERIES[i % SERIES.length];
    const sz = sizes[t.table];
    const matches = q ? fs.filter((f) => f.name.toLowerCase().includes(q)).map((f) => f.name) : [];
    const g = s("g", { class: "node tbl" + (q && !matches.length && !t.table.toLowerCase().includes(q) ? " faded" : ""), transform: `translate(${x},${y})`, tabindex: 0, role: "button", "aria-label": `table ${t.table}` });
    g.append(s("rect", { class: "box", width: W, height: H, rx: 11 }),
      s("path", { class: "stripe", d: `M0,11 a11,11 0 0 1 11,-11 h${W - 22} a11,11 0 0 1 11,11 v3 h-${W} z`, style: { fill: color } }),
      s("text", { class: "t", x: 14, y: 30 }, t.table),
      s("text", { class: "s", x: W - 12, y: 30, "text-anchor": "end" }, sz ? `${fmt.si(sz.rows)} rows${sz.bytes ? " · " + fmt.bytes(sz.bytes) : ""}` : `${fs.length} fields`),
      s("line", { class: "sepl", x1: 0, x2: W, y1: HD - 4, y2: HD - 4 }));
    fs.forEach((f, k) => {
      const m = typeMeta(f);
      const yy = HD + k * ROW;
      const flags = [f.name === sh.key ? "KEY" : null, f.references ? `→ ${f.references}` : null, f.multi ? "[ ]" : null, f.not_null && f.name !== sh.key ? "NOT NULL" : null, f.embed ? f.embed.model : null].filter(Boolean).join("  ");
      g.append(s("g", { class: "frow" + (matches.includes(f.name) ? " match" : "") },
        s("rect", { class: "rbg", x: 4, y: yy - 1, width: W - 8, height: ROW, rx: 5 }),
        s("rect", { class: "glyph-bg", x: 12, y: yy + 3, width: 16, height: 16, rx: 4, style: { fill: m.color } }),
        s("text", { class: "glyph", x: 20, y: yy + 15, "text-anchor": "middle", style: { fill: m.color } }, m.glyph),
        s("text", { class: "fn" + (f.name === sh.key ? " key" : ""), x: 36, y: yy + 15 }, f.name),
        s("text", { class: "fl", x: W - 12, y: yy + 15, "text-anchor": "end" }, flags)));
    });
    const o = optsLine(sh);
    if (o) g.append(s("text", { class: "s", x: 14, y: H - 10 }, o));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter") open(t.table); });
    return g;
  }

  function open(name) {
    const t = tableOf(name);
    if (!t) return;
    const sh = t.short || {};
    const sz = sizes[t.table];
    const first = ts[0]?.table === t.table;
    const ls = links(ts).filter((l) => l.from.table === t.table || l.to.table === t.table);
    const d = drawer({
      title: h("span", { class: "row nw" }, icon("table"), t.table), wide: true,
      sub: [sz ? `${fmt.plural(sz.rows, "row")}` : null, sz?.bytes ? fmt.bytes(sz.bytes) : null, sz?.segments ? fmt.plural(sz.segments, "segment") : null, `key ${sh.key || "none"}`].filter(Boolean).join(" · "),
      body: [
        h("div", { class: "row" },
          h("a", { class: "btn primary", href: "#/sql?q=" + encodeURIComponent(`SELECT * FROM ${quoteName(t.table)} LIMIT 20;`), onclick: () => d.close() }, icon("sql"), "Query it"),
          h("a", { class: "btn", href: "#/builder?table=" + encodeURIComponent(t.table), onclick: () => d.close() }, icon("builder"), "Build a query"),
          h("button", { class: "btn", title: "Start a new table from this one", onclick: () => { d.close(); openDesigner({ ...JSON.parse(JSON.stringify(sh)), table: sh.table + "_copy" }); } }, icon("copy"), "Use as a template"),
          h("span", { class: "sep" }),
          h("button", { class: "btn ghost danger", disabled: first, title: first ? "The first table of a database cannot be dropped" : `DROP TABLE ${t.table}`, onclick: () => drop(t, d) }, icon("trash"), "Drop")),
        optsLine(sh) ? h("div", { class: "row", style: { gap: "6px" } }, [sh.edges && h("span", { class: "badge accent" }, `edges ${sh.edges}`), sh.shard_by && h("span", { class: "badge" }, `shard by ${sh.shard_by}`), sh.cluster_by && h("span", { class: "badge" }, `cluster by ${sh.cluster_by}`), sh.retain && h("span", { class: "badge" }, `keeps ${sh.retain}${sh.retain_by ? ` by ${sh.retain_by}` : ""}`)]) : null,
        h("section", { class: "card" }, h("div", { class: "hd" }, h("h3", {}, "Fields"), h("span", { class: "sub" }, fmt.plural((sh.fields || []).length, "field"))),
          h("div", { class: "bd flush" }, h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "field"), h("th", {}, "type"), h("th", {}, "notes"))),
            h("tbody", {}, (sh.fields || []).map((f) => h("tr", {},
              h("td", {}, h("span", { class: "row nw", style: { gap: "6px" } }, typeBadge(f), h("b", {}, f.name))),
              h("td", { class: "mono small" }, typeMeta(f).label + (f.multi ? "[]" : "")),
              h("td", { class: "small dim" }, fieldNotes(f, sh.key).join(" · ") || h("span", { class: "muted" }, "–")))))))),
        ls.length ? h("section", { class: "card" }, h("div", { class: "hd" }, h("h3", {}, "Links")), h("div", { class: "bd stack", style: { gap: "6px" } }, ls.map((l) => h("div", { class: "row nw small" },
          h("span", { style: { width: "18px", borderTop: `2px ${l.kind === "graph" ? "dashed" : "solid"} ${(KIND[l.kind] || KIND.ref).color}` } }),
          h("span", { class: "mono" }, `${l.from.table}${l.from.field ? "." + l.from.field : ""}`), h("span", { class: "muted" }, (KIND[l.kind] || KIND.ref).label), h("span", { class: "mono" }, `${l.to.table}${l.to.field ? "." + l.to.field : ""}`))))) : null,
        (t.embeddings || []).length || (t.profiles || []).length || (t.derived || []).length ? h("section", { class: "card" }, h("div", { class: "hd" }, h("h3", {}, "Models and rank profiles")),
          h("div", { class: "bd stack", style: { gap: "6px" } },
            (t.embeddings || []).map((e) => h("div", { class: "row nw small" }, e.error ? h("span", { class: "pill bad" }, "error") : e.loaded ? h("span", { class: "pill ok" }, "loaded") : h("span", { class: "pill" }, "not loaded"), h("b", {}, e.field), h("span", { class: "dim" }, `embeds ${[].concat(e.from).join(", ")} with ${e.model}`), e.error ? h("span", { class: "error small" }, e.error) : null)),
            (t.derived || []).map((e) => h("div", { class: "row nw small" }, h("span", { class: "pill" }, "derived"), h("b", {}, e.field), h("span", { class: "dim" }, JSON.stringify(e)))),
            (t.profiles || []).map((p) => h("div", { class: "row nw small" }, h("span", { class: "pill info" }, "profile"), h("b", {}, p.name || p.profile || ""), h("span", { class: "mono dim ellipsis" }, p.sql || p.expression || JSON.stringify(p)))))) : null,
        h("section", { class: "card" }, h("div", { class: "hd" }, h("h3", {}, "CREATE TABLE"), h("div", { class: "tools" }, h("button", { class: "btn sm ghost", onclick: () => copy(t.sql, "CREATE TABLE copied") }, icon("copy"), "Copy"))),
          h("div", { class: "bd" }, sqlBlock(t.sql || "", { style: { maxHeight: "40vh" } }))),
        h("details", { class: "more" }, h("summary", {}, "The schema as JSON (POST /v1/schema/check takes it)"), h("pre", { class: "code", style: { maxHeight: "40vh" } }, JSON.stringify(sh, null, 2))),
      ],
    });
  }
  async function drop(t, d) {
    const refs = links(ts).filter((l) => l.kind === "ref" && l.to.table === t.table && l.from.table !== t.table).map((l) => l.from.table);
    const ok = await confirm({ title: `Drop ${t.table}?`, danger: true, ok: "Drop the table",
      body: h("div", { class: "stack", style: { gap: "8px" } }, h("p", { style: { margin: 0 } }, `Every row of ${t.table} is deleted, and it cannot be undone.`), refs.length ? h("p", { class: "small dim", style: { margin: 0 } }, `${[...new Set(refs)].join(", ")} reference it: CASCADE removes those references.`) : null, sqlBlock(`DROP TABLE ${quoteName(t.table)}${refs.length ? " CASCADE" : ""};`)) });
    if (!ok) return;
    try { await api("/v1/sql", { body: { sql: `DROP TABLE ${quoteName(t.table)}${refs.length ? " CASCADE" : ""}` } }); toast(`${t.table} dropped`, "ok"); d.close(); invalidate(); ts = await tables(true); paint(false); }
    catch (e) { toast(e.message, "bad", 8000); }
  }
  return () => { hideTip(); host.openTable = null; };
}

/** Rows, bytes and segments of each table: the storage stats where the server has them, else COUNT(*). */
async function tableSizes(ts) {
  const out = {};
  try {
    const st = await api("/v1/stats");
    for (const sg of st.storage?.segments || []) {
      const o = (out[sg.table || ts[0].table] ||= { rows: 0, bytes: 0, segments: 0 });
      o.rows += sg.n_docs - sg.n_deleted; o.bytes += sg.bytes; o.segments++;
    }
  } catch { /* a coordinator has no storage of its own */ }
  await Promise.all(ts.map(async (t) => {
    try { const j = await api("/v1/query", { body: { sql: `SELECT COUNT(*) FROM ${quoteName(t.table)}` } }); (out[t.table] ||= { bytes: 0, segments: 0 }).rows = j.rows[0][0]; } catch { /* left without a count */ }
  }));
  return out;
}

// ------------------------------------------------------------ the designer
const PALETTE = [
  ["text", false, "Words to search"], ["keyword", false, "An exact value"], ["keyword", true, "A list of values: tags"], ["int", false, "A whole number"], ["float", false, "A number"],
  ["bool", false, "True or false"], ["timestamp", false, "A moment"], ["vector", false, "An embedding"], ["sparse", false, "Learned term weights"], ["media", false, "An image, sound or file"], ["edges", false, "Links to other rows"],
];
const TEMPLATES = {
  "Products": { table: "products", key: "id", fields: [{ name: "id", type: "keyword" }, { name: "title", type: "text" }, { name: "description", type: "text" }, { name: "brand", type: "keyword" }, { name: "tags", type: "keyword", multi: true }, { name: "price", type: "float" }, { name: "rating", type: "float" }, { name: "added", type: "timestamp", default: "now" }, { name: "emb", type: "vector", embed: { from: ["title", "description"], model: "potion-retrieval-32M" } }] },
  "Event log": { table: "events", key: "id", shard_by: "usr", cluster_by: "usr", retain: "400 days", fields: [{ name: "id", type: "int" }, { name: "usr", type: "keyword", not_null: true }, { name: "item", type: "keyword" }, { name: "action", type: "keyword", not_null: true, default: "view" }, { name: "ts", type: "timestamp", default: "now" }] },
  "Edge table": { table: "cooc", key: "id", edges: "src, dst, w", fields: [{ name: "id", type: "keyword" }, { name: "src", type: "keyword", not_null: true }, { name: "dst", type: "keyword", not_null: true }, { name: "w", type: "float", default: 0 }] },
  "Documents": { table: "docs", key: "id", fields: [{ name: "id", type: "keyword" }, { name: "title", type: "text" }, { name: "body", type: "text" }, { name: "source", type: "keyword" }, { name: "published", type: "timestamp" }, { name: "emb", type: "vector", embed: { from: ["title", "body"], model: "potion-retrieval-32M" }, index: "hnsw" }] },
};
const MODELS = ["potion-retrieval-32M", "potion-base-8M", "hash:64"];
// The models a schema can name and whether each opens on this server
// (GET /v1/models): fetched here, or hosted with its key set.
let catalogOf = null;
const modelCatalog = () => (catalogOf ||= api("/v1/models").then((j) => j.models || []).catch(() => []));

// A picker of embedding models: those ready here first, then those that
// need fetching or a key, each saying so.
function modelSelect(f, changed) {
  const sel = h("select", { class: "select sm", onchange: (e) => { f.embed.model = e.target.value; note(); changed(false); } });
  const hint = h("div", { class: "small muted" });
  let known = [];
  const opt = (m) => h("option", { value: m.name, selected: m.name === f.embed.model }, m.name + (m.dim ? ` · ${m.dim}` : "") + (m.hosted ? " · hosted" : ""));
  const note = () => {
    const m = known.find((x) => x.name === f.embed.model);
    hint.textContent = !m ? "" : m.ready ? m.what : `${m.what}. Not ready here: ${m.error}`;
    hint.className = `small ${m && !m.ready ? "error" : "muted"}`;
  };
  const fill = (list) => {
    known = list.length ? list : MODELS.map((name) => ({ name, ready: true, what: "" }));
    if (!known.some((m) => m.name === f.embed.model)) known = [{ name: f.embed.model, ready: true, what: "" }, ...known];
    const ready = known.filter((m) => m.ready), not = known.filter((m) => !m.ready);
    clear(sel, h("optgroup", { label: "Ready on this server" }, ready.map(opt)), not.length ? h("optgroup", { label: "Needs fetching or an API key" }, not.map(opt)) : null);
    note();
  };
  fill([]);
  modelCatalog().then(fill);
  return h("div", { class: "stack", style: { gap: "3px" } }, sel, hint);
}

function designer(host, bar, { seed, onCreated }) {
  const uid = () => Math.random().toString(36).slice(2, 9);
  const withIds = (sh) => ({ ...sh, fields: (sh.fields || []).map((f) => ({ ...f, _id: uid() })) });
  let d = withIds(seed || store.get("schema.draft", null) || { table: "new_table", key: "id", fields: [{ name: "id", type: "keyword" }] });
  let open = null, checked = null;
  const ts = cached();
  const pal = h("aside", { class: "sch-pal" });
  const editor = h("div", { class: "sch-ed" });
  const right = h("div", { class: "sch-right" });
  clear(host, h("div", { class: "sch-design" }, pal, h("div", { class: "sch-canvas" }, editor), right));
  bar.append(h("span", { class: "small muted" }, "Drag types into the table · the server checks it as you go"));

  const toShort = () => {
    const out = { table: d.table, key: d.key || undefined, fields: d.fields.map(({ _id, ...f }) => f) };
    for (const k of ["shard_by", "cluster_by", "retain", "retain_by", "edges"]) if (d[k]) out[k] = d[k];
    return out;
  };
  const save = debounce(() => store.set("schema.draft", toShort()), 300);
  const check = debounce(async () => {
    const mine = (checked = {});
    try { const j = await api("/v1/schema/check", { body: toShort() }); if (mine === checked) { checked = { ok: true, j }; drawRight(); } }
    catch (e) { if (mine === checked) { checked = { ok: false, error: e.message }; drawRight(); } }
  }, 250);
  const changed = (redraw = true) => { save(); check(); if (redraw) drawEditor(); };

  // the palette
  let dragType = null;
  clear(pal,
    h("div", { class: "wb-sh" }, h("b", {}, "Types"), h("span", { class: "small muted" }, "drag or click")),
    h("div", { class: "sch-types" }, PALETTE.map(([type, multi, what]) => {
      const m = TYPES[type];
      return h("button", { class: "blk", draggable: "true", title: m.what, style: { "--c": m.color },
        ondragstart: (e) => { dragType = { type, multi }; e.dataTransfer.effectAllowed = "copy"; e.dataTransfer.setData("text/plain", type); },
        ondragend: () => { dragType = null; placeholder.remove(); },
        onclick: () => addField(type, multi) },
      h("span", { class: "tb", style: { "--c": m.color } }, m.glyph), h("span", {}, h("b", {}, m.label + (multi ? "[]" : "")), h("span", { class: "small muted" }, what)));
    })),
    h("div", { class: "wb-sh", style: { marginTop: "14px" } }, h("b", {}, "Start from")),
    h("div", { class: "stack", style: { gap: "5px" } }, Object.entries(TEMPLATES).map(([name, t]) => h("button", { class: "btn sm", style: { justifyContent: "flex-start" }, onclick: () => { d = withIds(JSON.parse(JSON.stringify(t))); open = null; changed(); } }, icon("table"), name)),
      h("button", { class: "btn sm ghost", style: { justifyContent: "flex-start" }, onclick: () => { d = withIds({ table: "new_table", key: "id", fields: [{ name: "id", type: "keyword" }] }); open = null; changed(); } }, icon("x"), "Empty")));

  const placeholder = h("div", { class: "sch-ph" });
  function addField(type, multi, at = d.fields.length) {
    let n = type === "vector" ? "emb" : type === "timestamp" ? "ts" : type, k = 1;
    while (d.fields.some((f) => f.name === n)) n = `${type}_${++k}`;
    const f = { _id: uid(), name: n, type };
    if (multi) f.multi = true;
    if (type === "vector") { const src = d.fields.filter((x) => x.type === "text").map((x) => x.name); if (src.length) f.embed = { from: src.slice(0, 2), model: MODELS[0] }; else f.dim = 384; }
    d.fields.splice(at, 0, f);
    open = f._id;
    changed();
    requestAnimationFrame(() => { const el = editor.querySelector(`[data-id="${f._id}"] input.fname`); el?.focus(); el?.select(); });
  }

  // the table being designed
  function drawEditor() {
    const list = h("div", { class: "sch-fields" });
    for (const f of d.fields) list.append(fieldRow(f));
    list.ondragover = (e) => {
      if (dragType) {
        e.preventDefault();
        const rows = [...list.querySelectorAll(".sf")];
        const before = rows.find((r) => { const b = r.getBoundingClientRect(); return e.clientY < b.top + b.height / 2; });
        if (before) list.insertBefore(placeholder, before); else list.append(placeholder);
        return;
      }
      if (!dragRow) return;
      e.preventDefault();
      const over = e.target.closest(".sf");
      if (!over || over === dragRow) return;
      const b = over.getBoundingClientRect();
      list.insertBefore(dragRow, e.clientY > b.top + b.height / 2 ? over.nextSibling : over);
    };
    list.ondrop = (e) => {
      if (!dragType) return;
      e.preventDefault();
      const at = [...list.children].filter((c) => c.classList.contains("sf") || c === placeholder).indexOf(placeholder);
      placeholder.remove();
      const t = dragType; dragType = null;
      addField(t.type, t.multi, at < 0 ? d.fields.length : at);
    };
    list.ondragleave = (e) => { if (!list.contains(e.relatedTarget)) placeholder.remove(); };
    const name = h("input", { class: "input tname", value: d.table, spellcheck: "false", "aria-label": "table name", oninput: () => { d.table = name.value.trim(); changed(false); } });
    const textFields = d.fields.filter((f) => f.type === "keyword" || f.type === "int");
    clear(editor, h("div", { class: "sch-card" },
      h("div", { class: "sch-hd" }, icon("table"), name, h("span", { class: "small muted" }, fmt.plural(d.fields.length, "field"))),
      list,
      h("div", { class: "sch-add" }, icon("plus"), "Drop a type here, or click one on the left"),
      h("details", { class: "more sch-topts", open: !!(d.shard_by || d.cluster_by || d.retain || d.edges) }, h("summary", {}, "Table options"),
        h("div", { class: "sch-grid" },
          lab("Shard by", "Place rows on shards by this column, not the key: a user's events together", sel(d, "shard_by", textFields.map((f) => f.name))),
          lab("Cluster by", "Keep each segment's rows sorted by it, so one value's rows are read in one run", sel(d, "cluster_by", d.fields.filter((f) => ["keyword", "int", "timestamp"].includes(f.type) && !f.multi).map((f) => f.name))),
          lab("Keep rows for", "Drop rows older than this, by a TIMESTAMP field: 90 days, 12 h, 2 weeks", h("input", { class: "input sm", value: d.retain || "", placeholder: "forever", oninput: (e) => { d.retain = e.target.value.trim() || undefined; changed(false); } })),
          lab("… by the time in", "Which TIMESTAMP field the period counts from, when there are several", sel(d, "retain_by", d.fields.filter((f) => f.type === "timestamp").map((f) => f.name))),
          lab("Edges", "Each row is an edge src → dst (weight) of a graph over another table: src, dst, w", h("input", { class: "input sm mono", value: d.edges || "", placeholder: "src, dst, w", oninput: (e) => { d.edges = e.target.value.trim() || undefined; changed(false); } }))))));
  }
  let dragRow = null;
  function fieldRow(f) {
    const m = typeMeta(f);
    const row = h("div", { class: "sf" + (open === f._id ? " open" : ""), dataset: { id: f._id }, style: { "--c": m.color } });
    const grip = h("span", { class: "grip", title: "Drag to reorder" }, icon("grip"));
    grip.onpointerdown = () => { row.draggable = true; };
    row.ondragstart = (e) => { if (!row.draggable) return; dragRow = row; row.classList.add("dragging-src"); e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", f.name); };
    row.ondragend = () => { row.draggable = false; row.classList.remove("dragging-src"); if (dragRow) { const ids = [...row.parentNode.querySelectorAll(".sf")].map((r) => r.dataset.id); d.fields.sort((a, b) => ids.indexOf(a._id) - ids.indexOf(b._id)); dragRow = null; changed(); } };
    const nm = h("input", { class: "input sm fname mono", value: f.name, spellcheck: "false", "aria-label": "field name", oninput: () => { const old = f.name; f.name = nm.value.trim(); if (d.key === old) d.key = f.name; changed(false); } });
    const ty = h("select", { class: "select sm", "aria-label": "type", onchange: () => { f.type = ty.value; for (const k of ["dim", "index", "embed", "codes", "analyzer", "positions", "from", "kind", "references", "semantic_id", "derive"]) delete f[k]; if (!["keyword", "int"].includes(f.type)) delete f.multi; if (f.type === "vector") f.dim = 384; changed(); } },
      Object.entries(TYPES).map(([k, v]) => h("option", { value: k, selected: k === f.type }, v.label)));
    const canKey = (f.type === "keyword" || f.type === "int") && !f.multi;
    row.append(
      h("div", { class: "row nw sf-main" }, grip, h("span", { class: "tb", style: { "--c": m.color } }, m.glyph), nm, ty,
        f.type === "keyword" || f.type === "int" ? h("label", { class: "chk", title: "A list of values" }, h("input", { type: "checkbox", checked: !!f.multi, onchange: (e) => { f.multi = e.target.checked || undefined; if (f.multi && d.key === f.name) d.key = undefined; changed(); } }), "[ ]") : h("span", { class: "chk-sp" }),
        h("label", { class: "chk" + (canKey ? "" : " off"), title: canKey ? "The primary key: rows are upserted and deleted by it" : "A key is one KEYWORD or INT" }, h("input", { type: "radio", name: "key", disabled: !canKey, checked: d.key === f.name, onchange: () => { d.key = f.name; changed(); } }), "key"),
        h("label", { class: "chk", title: "Refuse rows without a value" }, h("input", { type: "checkbox", checked: !!f.not_null || d.key === f.name, disabled: d.key === f.name, onchange: (e) => { f.not_null = e.target.checked || undefined; changed(false); } }), "not null"),
        h("span", { class: "sep" }),
        h("span", { class: "small muted ellipsis notes" }, fieldNotes(f, null).filter((x) => x !== "primary key").join(" · ")),
        h("button", { class: "btn ghost icon sm", title: "Options", "aria-expanded": String(open === f._id), onclick: () => { open = open === f._id ? null : f._id; drawEditor(); } }, icon(open === f._id ? "up" : "dots")),
        h("button", { class: "btn ghost icon sm", title: "Remove", onclick: () => { d.fields.splice(d.fields.indexOf(f), 1); if (d.key === f.name) d.key = undefined; changed(); } }, icon("x"))),
      ...(open === f._id ? [h("div", { class: "sf-opts" }, fieldOptions(f))] : []));
    return row;
  }
  function fieldOptions(f) {
    const others = (pred) => d.fields.filter((x) => x !== f && pred(x)).map((x) => x.name);
    const out = [];
    const def = h("input", { class: "input sm", value: f.default === undefined ? "" : f.default === "now" ? "now" : String(f.default), placeholder: f.type === "timestamp" ? "now, or 2026-01-01" : "none",
      oninput: (e) => { const v = e.target.value.trim(); f.default = v === "" ? undefined : f.type === "int" || f.type === "float" ? (Number.isFinite(Number(v)) ? Number(v) : v) : f.type === "bool" ? v === "true" : v; changed(false); } });
    if (["text", "keyword", "int", "float", "bool", "timestamp"].includes(f.type) && !f.multi && d.key !== f.name) out.push(lab("Default", f.type === "timestamp" ? "now: the time of the write" : "the value a write that leaves the field out gets", def));
    if ((f.type === "keyword" || f.type === "int") && !f.multi) out.push(lab("References", "This field holds keys of that table (not enforced; joins follow it)", sel(f, "references", [d.table, ...cached().map((t) => t.table)].filter((v, i, a) => v && a.indexOf(v) === i))));
    if (f.type === "text") {
      out.push(lab("Analyzer", "english: stopwords dropped, words stemmed; simple: every word as written", sel(f, "analyzer", ["english", "simple"], "english")));
      out.push(lab("Made from", "A generated field: the text of these fields joined, searched as one", multiSel(f, "from", others((x) => x.type === "text"))));
      out.push(lab("Phrases", "Keep word positions, for \"quoted phrases\"", tog(f, "positions", true)));
      out.push(lab("Searchable", "Index the words; off keeps the text only to show", tog(f, "indexed", true)));
    }
    if (f.type === "vector") {
      const emb = !!f.embed;
      out.push(lab("Vectors from", "Embed these text fields with a model when rows are written, and text in queries", seg([["model", "a model"], ["given", "the rows"]], emb ? "model" : "given", (v) => { if (v === "model") { f.embed = { from: others((x) => x.type === "text").slice(0, 2), model: MODELS[0] }; delete f.dim; } else { delete f.embed; f.dim = 384; } changed(); })));
      if (emb) {
        out.push(lab("Model", "Computed here, or behind an API (OpenAI, Cohere, Voyage, Gemini, Mistral, Jina, Ollama) with its key in the server's environment", modelSelect(f, changed)));
        out.push(lab("Of the fields", "", multiSel(f.embed, "from", others((x) => x.type === "text" || x.type === "keyword"))));
      } else out.push(lab("Dimensions", "", h("input", { class: "input sm num", type: "number", min: 1, value: f.dim || "", style: { width: "90px" }, oninput: (e) => { f.dim = Number(e.target.value) || undefined; changed(false); } })));
      const ix = typeof f.index === "object" ? Object.keys(f.index)[0] : f.index || "hnsw";
      out.push(lab("Index", "hnsw: fast approximate search (the default); flat: exact, for small tables; ivf: clusters", seg([["hnsw", "HNSW"], ["flat", "flat"], ["ivf", "IVF"]], ix, (v) => { f.index = v === "ivf" ? { ivf: { nlist: 256 } } : v === "hnsw" ? undefined : v; changed(); })));
      if (ix === "ivf") out.push(lab("Lists", "", h("input", { class: "input sm num", type: "number", min: 1, value: f.index.ivf.nlist, style: { width: "90px" }, oninput: (e) => { f.index.ivf.nlist = Number(e.target.value) || 256; changed(false); } })));
      out.push(lab("Codes", "sq8: 8 bits a dimension (the default); binary: 1 bit, 32 times smaller", sel(f, "codes", ["sq8", "binary"], "sq8")));
    }
    if (f.type === "media") out.push(lab("Kind", "", sel(f, "kind", ["image", "audio", "video", "pdf", "file"])));
    if (!out.length) out.push(h("div", { class: "small muted" }, `${typeMeta(f).label} has no options.`));
    return h("div", { class: "sch-grid" }, out);
  }
  const lab = (label, hint, ctl) => h("label", { class: "f", title: hint }, h("span", {}, label, hint ? h("span", { class: "muted small hint" }, hint) : null), ctl);
  function sel(obj, key, values, dflt) {
    return h("select", { class: "select sm", onchange: (e) => { obj[key] = e.target.value === "" || e.target.value === dflt ? undefined : e.target.value; changed(); } },
      dflt ? null : h("option", { value: "" }, "none"), values.map((v) => h("option", { value: v, selected: (obj[key] ?? dflt) === v }, v)));
  }
  function tog(obj, key, dflt) {
    return h("input", { type: "checkbox", checked: obj[key] ?? dflt, onchange: (e) => { obj[key] = e.target.checked === dflt ? undefined : e.target.checked; changed(false); } });
  }
  function multiSel(obj, key, values) {
    const cur = new Set([].concat(obj[key] || []));
    return h("div", { class: "row", style: { gap: "4px" } }, values.length ? values.map((v) => h("button", { type: "button", class: "chip", "aria-pressed": String(cur.has(v)), onclick: () => { if (cur.has(v)) cur.delete(v); else cur.add(v); obj[key] = cur.size ? [...cur] : undefined; changed(); } }, v)) : h("span", { class: "small muted" }, "no fields to choose"));
  }

  function drawRight() {
    const exists = ts.some((t) => t.table.toLowerCase() === (d.table || "").toLowerCase());
    const create = h("button", { class: "btn primary", disabled: !checked?.ok || exists, onclick: go }, icon("plus"), "Create table");
    clear(right,
      h("section", { class: "card" }, h("div", { class: "hd" }, h("h3", {}, "The server's check"), h("div", { class: "tools" }, checked?.ok ? h("span", { class: "pill ok" }, icon("okc"), "valid") : checked?.error ? h("span", { class: "pill bad" }, icon("bad"), "not yet") : h("span", { class: "pill" }, "checking"))),
        h("div", { class: "bd stack", style: { gap: "10px" } },
          checked?.error ? h("div", { class: "error" }, checked.error.replace(/^schema error: /, "")) : null,
          exists ? h("div", { class: "small note-warn" }, icon("warn"), `${d.table} exists: name the new table something else.`) : null,
          checked?.ok ? [h("div", { class: "small muted" }, "The statement that creates it:"), sqlBlock(checked.j.sql, { style: { maxHeight: "46vh" } })] : null),
        h("div", { class: "ft" }, create, h("button", { class: "btn ghost", disabled: !checked?.ok, onclick: () => copy(checked.j.sql, "CREATE TABLE copied") }, icon("copy"), "Copy SQL"))),
      h("div", { class: "small muted", style: { padding: "0 4px" } }, "Tables cannot be altered yet: a field added later means a new table. References are recorded, not enforced."));
  }
  async function go() {
    try {
      await api("/v1/sql", { body: { sql: checked.j.sql } });
      toast(`${d.table} created`, "ok");
      store.del("schema.draft");
      invalidate(); await tables(true);
      onCreated(d.table);
    } catch (e) { toast(e.message, "bad", 9000); }
  }
  drawEditor(); drawRight(); check();
}
