// The command center's shell: the navigation, the top bar (time range,
// live or paused, alerts, theme, who is signed in), the router, the
// command palette (Ctrl+K) and the keyboard shortcuts.

import { h, icon, clear, store, bus, api, setToken, hasToken, modal, toast, menu, seg, fmt } from "./core.js";
import { state, start, setRange, setZoom, setPaused, refreshWho } from "./data.js";
import { VIEWS, GROUPS } from "./views/index.js";

const app = document.getElementById("app");

// ---------------------------------------------------------------- theme
const applyTheme = (t) => { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; bus.emit("theme"); };
const isDark = () => document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
function toggleTheme() { const t = isDark() ? "light" : "dark"; applyTheme(t); store.set("theme", t); themeBtn.replaceChildren(icon(isDark() ? "sun" : "moon")); }
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => bus.emit("theme"));

// ---------------------------------------------------------------- shell
const nav = h("nav", { class: "nav", "aria-label": "Views" });
const side = h("aside", { class: "side" },
  h("div", { class: "brand" },
    h("svg", { class: "mark", viewBox: "0 0 32 32", "aria-hidden": "true" }),
    h("div", {}, "Strata", h("small", {}, "command center"))),
  nav,
  h("div", { class: "side-foot" },
    h("button", { class: "btn ghost sm", title: "Collapse the sidebar ( [ )", onclick: () => toggleSide() }, icon("left"), h("span", {}, "Collapse"))));
side.querySelector(".mark").innerHTML = '<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="var(--s1)"/><stop offset="1" stop-color="var(--s7)"/></linearGradient></defs><rect x="2" y="2" width="28" height="28" rx="8" fill="url(#lg)"/><path d="M8 11.5 16 8l8 3.5-8 3.5z" fill="#fff" opacity=".95"/><path d="m8 16 8 3.5 8-3.5" stroke="#fff" stroke-width="2" fill="none" stroke-linejoin="round" opacity=".8"/><path d="m8 20.5 8 3.5 8-3.5" stroke="#fff" stroke-width="2" fill="none" stroke-linejoin="round" opacity=".6"/>';

const crumbs = h("div", { class: "crumbs" });
const connDot = h("span", { class: "dot ok pulse" });
const connText = h("span", {}, "connecting…");
const rangeSeg = seg([[5, "5m"], [15, "15m"], [30, "30m"], [60, "1h"]], state.range, (v) => setRange(v), { "aria-label": "Time range", title: "The time the charts show" });
const zoomChip = h("button", { class: "chip hidden", "aria-pressed": "true", title: "Drag across a chart to zoom; this resets it", onclick: () => setZoom(null) }, icon("zout"), "zoomed", h("span", { class: "x" }, "×"));
const liveBtn = h("button", { class: "btn sm", title: "Pause or resume the live view (p)", onclick: () => setPaused(!state.paused) });
const bell = h("a", { class: "btn ghost icon", href: "#/alerts", title: "Alerts", style: { position: "relative" } }, icon("alerts"));
const bellCount = h("span", { class: "count hidden", style: { position: "absolute", top: "-3px", right: "-4px", fontSize: "10px", minWidth: "16px", padding: "0 4px", borderRadius: "8px", background: "var(--crit)", color: "#fff", textAlign: "center", fontWeight: 700 } });
bell.append(bellCount);
const themeBtn = h("button", { class: "btn ghost icon", title: "Light or dark (t)", onclick: toggleTheme }, icon(isDark() ? "sun" : "moon"));
const whoBtn = h("button", { class: "btn ghost sm", title: "Who you are signed in as" }, icon("user"), h("span", {}, "…"));
whoBtn.onclick = () => menu(whoBtn, [
  { label: "Use another key or token", icon: "access", onclick: askToken },
  { label: "Sign out of this tab", icon: "unlock", onclick: () => { setToken(null); toast("Signed out: the key is forgotten for this tab"); } },
  "-",
  { label: "Manage API keys", icon: "lock", onclick: () => (location.hash = "#/access") },
]);
const searchBtn = h("button", { class: "search-btn", onclick: () => openPalette(), title: "Search views and actions" }, icon("search"), h("span", {}, "Search or jump to…"), h("kbd", {}, navigator.platform?.includes("Mac") ? "⌘K" : "Ctrl K"));
const top = h("header", { class: "top" }, crumbs, h("div", { class: "sep" }), searchBtn, rangeSeg, zoomChip, liveBtn,
  h("div", { class: "conn", title: "The server this page talks to" }, connDot, connText), bell, themeBtn, whoBtn);
const content = h("main", { class: "content", id: "content" });
const shell = h("div", { class: "shell" + (store.get("side", false) ? " collapsed" : "") }, side, top, content);
app.replaceWith(shell);

function toggleSide() { shell.classList.toggle("collapsed"); store.set("side", shell.classList.contains("collapsed")); }
function drawLive() { clear(liveBtn, icon(state.paused ? "play" : "pause"), state.paused ? "Resume" : "Live"); liveBtn.classList.toggle("primary", state.paused); }
drawLive();
bus.on("paused", drawLive);
bus.on("range", () => { rangeSeg.set(state.range); zoomChip.classList.toggle("hidden", !state.zoom); });

function drawConn() {
  const hl = state.health;
  if (!state.online) { connDot.className = "dot bad pulse"; connText.textContent = "offline: " + (state.error || "no answer"); return; }
  if (!hl) return;
  connDot.className = "dot pulse " + (hl.status === "ok" ? "ok" : hl.status === "degraded" ? "warn" : "bad");
  const role = hl.role === "coordinator" ? `coordinator · ${fmt.plural(hl.shards?.length || 0, "shard")}` : hl.replication?.role || (hl.slots ? `shard ${hl.slots.shard} of ${hl.slots.of}` : "standalone");
  connText.textContent = `${location.host} · ${role}`;
}
bus.on("health", drawConn); bus.on("online", drawConn);
function drawWho() {
  const w = state.whoami;
  whoBtn.querySelector("span").textContent = w ? `${w.name} · ${w.scope}` : hasToken() ? "key refused" : "signed out";
}
bus.on("whoami", drawWho);
function drawBell() {
  const n = state.alerts?.firing?.length || 0;
  bellCount.textContent = n; bellCount.classList.toggle("hidden", !n);
  for (const a of nav.querySelectorAll("a[data-view=alerts] .count")) a.remove();
  if (n) nav.querySelector("a[data-view=alerts]")?.append(h("span", { class: "count" }, n));
}
bus.on("alerts", drawBell);

// ---------------------------------------------------------------- nav
for (const g of GROUPS) {
  nav.append(h("h6", {}, g));
  for (const v of VIEWS.filter((x) => x.group === g)) nav.append(h("a", { href: "#/" + v.id, dataset: { view: v.id }, title: v.title }, icon(v.icon), h("span", {}, v.title)));
}

// ---------------------------------------------------------------- router
let current = null, cleanup = null;
function parse() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, q] = raw.split("?");
  const [id, ...rest] = path.split("/");
  return { id: id || "overview", rest, params: new URLSearchParams(q || "") };
}
function route() {
  const r = parse();
  const v = VIEWS.find((x) => x.id === r.id) || VIEWS[0];
  for (const a of nav.querySelectorAll("a")) a.toggleAttribute("aria-current", a.dataset.view === v.id) || a.removeAttribute("aria-current");
  nav.querySelector(`a[data-view="${v.id}"]`)?.setAttribute("aria-current", "page");
  if (current === v && v.onParams) { v.onParams(r); return; }
  try { cleanup?.(); } catch (e) { console.error(e); }
  current = v;
  clear(crumbs, icon(v.icon), v.title, v.sub ? h("span", { class: "sub" }, "· " + v.sub) : null);
  document.title = `${v.title} · Strata`;
  const page = h("div", { class: "page" + (v.full ? " full" : "") });
  if (v.full) page.style.cssText = "padding:0;max-width:none";
  clear(content, page);
  content.scrollTop = 0;
  try { cleanup = v.render(page, r) || null; } catch (e) { console.error(e); page.append(h("div", { class: "error" }, "This view failed: " + e.message)); }
}
addEventListener("hashchange", route);

// ---------------------------------------------------------------- the key
function askToken() {
  const inp = h("input", { class: "input", type: "password", placeholder: "sk_… or the server's --token", autocomplete: "off", style: { width: "100%" } });
  const save = h("button", { class: "btn primary" }, "Use it");
  const m = modal({
    title: "Sign in to this server",
    body: [h("p", { class: "dim", style: { margin: 0 } }, "Paste an API key (read, write or admin) or the server's token. It is kept for this tab only, and sent as a bearer token."), inp],
    foot: [h("button", { class: "btn", onclick: () => m.close() }, "Cancel"), save],
  });
  const go = () => { setToken(inp.value.trim()); m.close(); refreshWho(); toast("Key saved for this tab", "ok"); };
  save.onclick = go;
  inp.onkeydown = (e) => { if (e.key === "Enter") go(); };
}
let asked = false;
bus.on("unauthorized", () => { if (!document.querySelector(".modal")) { if (!asked || !hasToken()) askToken(); asked = true; } });

// ---------------------------------------------------------------- palette
const ACTIONS = [
  { name: "Toggle light or dark", icon: "moon", run: toggleTheme, hint: "t" },
  { name: "Collapse or expand the sidebar", icon: "menu", run: toggleSide, hint: "[" },
  { name: "Pause or resume the live view", icon: "pause", run: () => setPaused(!state.paused), hint: "p" },
  { name: "Show the last 5 minutes", icon: "clock", run: () => setRange(5) },
  { name: "Show the last 15 minutes", icon: "clock", run: () => setRange(15) },
  { name: "Show the last hour", icon: "clock", run: () => setRange(60) },
  { name: "Use another API key or token", icon: "access", run: askToken },
  { name: "Create an API key", icon: "plus", run: () => (location.hash = "#/access?new=1") },
  { name: "Open the raw Prometheus metrics", icon: "ext", run: () => window.open("/metrics", "_blank") },
  { name: "Open the query console", icon: "classic", run: () => (location.hash = "#/classic") },
  { name: "Keyboard shortcuts", icon: "keyboard", run: () => shortcuts(), hint: "?" },
];
export function addActions(items) { ACTIONS.push(...items); }

function score(q, text) {
  if (!q) return 1;
  q = q.toLowerCase(); const t = text.toLowerCase();
  const i = t.indexOf(q);
  if (i >= 0) return 100 - i;
  let j = 0, sc = 0;
  for (const c of t) { if (c === q[j]) { j++; sc += 2; } if (j === q.length) return sc; }
  return 0;
}
function mark(text, q) {
  const i = q ? text.toLowerCase().indexOf(q.toLowerCase()) : -1;
  return i < 0 ? text : [text.slice(0, i), h("mark", {}, text.slice(i, i + q.length)), text.slice(i + q.length)];
}
function openPalette() {
  if (document.querySelector(".palette")) return;
  const inp = h("input", { placeholder: "Jump to a view, run an action…", "aria-label": "Search", autocomplete: "off" });
  const list = h("div", { class: "list", role: "listbox" });
  const pal = h("div", { class: "palette", role: "dialog", "aria-label": "Command palette" }, inp, list,
    h("div", { class: "foot" }, h("span", {}, h("kbd", {}, "↑"), " ", h("kbd", {}, "↓"), " to move"), h("span", {}, h("kbd", {}, "Enter"), " to open"), h("span", {}, h("kbd", {}, "Esc"), " to close")));
  const scrim = h("div", { class: "scrim" });
  document.body.append(scrim, pal);
  let items = [], sel = 0;
  const close = () => { pal.remove(); scrim.remove(); };
  scrim.onclick = close;
  function draw() {
    const q = inp.value.trim();
    const views = VIEWS.map((v) => ({ name: v.title, icon: v.icon, grp: "Views", hint: v.group, run: () => (location.hash = "#/" + v.id), sc: score(q, v.title + " " + (v.keywords || "")) }));
    const acts = ACTIONS.map((a) => ({ ...a, grp: "Actions", sc: score(q, a.name) }));
    items = [...views, ...acts].filter((x) => x.sc > 0).sort((a, b) => (q ? b.sc - a.sc : 0));
    if (sel >= items.length) sel = Math.max(0, items.length - 1);
    clear(list);
    let grp = null;
    items.forEach((it, i) => {
      if (!q && it.grp !== grp) { grp = it.grp; list.append(h("div", { class: "grp" }, grp)); }
      const row = h("div", { class: "it" + (i === sel ? " on" : ""), role: "option", "aria-selected": String(i === sel) }, icon(it.icon), h("span", {}, mark(it.name, q)), it.hint ? h("span", { class: "h" }, it.hint) : null);
      row.onclick = () => { close(); it.run(); };
      row.onmousemove = () => { if (sel !== i) { sel = i; draw(); } };
      list.append(row);
    });
    if (!items.length) list.append(h("div", { class: "empty" }, "Nothing matches"));
    list.querySelector(".on")?.scrollIntoView({ block: "nearest" });
  }
  inp.oninput = () => { sel = 0; draw(); };
  inp.onkeydown = (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(items.length - 1, sel + 1); draw(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(0, sel - 1); draw(); }
    else if (e.key === "Enter") { e.preventDefault(); const it = items[sel]; if (it) { close(); it.run(); } }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
  };
  draw();
  inp.focus();
}

function shortcuts() {
  const rows = [["Ctrl K, /", "Search views and actions"], ["g then o", "Overview"], ["g then t", "Topology"], ["g then c", "Cluster"], ["g then p", "Performance"], ["g then q", "Queries"], ["g then l", "Logs"], ["g then a", "Alerts"], ["g then k", "Access"], ["g then s", "System"], ["p", "Pause or resume"], ["t", "Light or dark"], ["[", "Collapse the sidebar"], ["Esc", "Close a panel"]];
  modal({ title: "Keyboard shortcuts", body: h("dl", { class: "kv" }, rows.flatMap(([k, v]) => [h("dt", {}, k.split(", ").map((x, i) => [i ? " or " : "", ...x.split(" then ").map((y, j) => [j ? " then " : "", h("kbd", {}, y)])])), h("dd", {}, v)])) });
}

let gPending = 0;
const G = { o: "overview", t: "topology", c: "cluster", p: "performance", q: "queries", l: "logs", a: "alerts", k: "access", s: "system", w: "sql", b: "builder", d: "schema", i: "import", r: "relevance" };
document.addEventListener("keydown", (e) => {
  const typing = e.target.closest?.("input,textarea,select,[contenteditable=true]");
  if ((e.key === "k" || e.key === "K") && (e.ctrlKey || e.metaKey)) { e.preventDefault(); openPalette(); return; }
  if (typing || e.ctrlKey || e.metaKey || e.altKey || document.querySelector(".modal,.palette,.drawer")) return;
  if (gPending && Date.now() - gPending < 1200 && G[e.key]) { gPending = 0; location.hash = "#/" + G[e.key]; e.preventDefault(); return; }
  gPending = 0;
  if (e.key === "g") gPending = Date.now();
  else if (e.key === "/") { e.preventDefault(); openPalette(); }
  else if (e.key === "?") shortcuts();
  else if (e.key === "t") toggleTheme();
  else if (e.key === "p") setPaused(!state.paused);
  else if (e.key === "[") toggleSide();
});

// ---------------------------------------------------------------- start
start();
route();
api("/v1/health").catch(() => {});
