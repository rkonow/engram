// Cluster: change the cluster from here. Each shard with the slots it holds
// and its copies (role, term, last transaction, whether each answers), the
// moves of slots under way; add a shard or a copy, remove one, hand a
// primary over, drain a shard, even the slots out. A cluster that the
// Kubernetes operator runs is scaled through its StrataCluster instead,
// and the operator makes each change. A server that is not in a cluster
// shows how to make one, on machines of your own or on Kubernetes.

import { h, icon, clear, fmt, card, api, every, toast, confirm, modal, pill, empty, copy, failure, bus, seg } from "../core.js";
import { state, isCoordinator, refreshCluster } from "../data.js";

export default { id: "cluster", title: "Cluster", icon: "layers", group: "Admin", keywords: "cluster shards replicas copies scale add remove promote primary drain rebalance move slots kubernetes k8s operator deploy", render };

const SLOTS = 4096;
const short = (url) => (url || "").replace(/^https?:\/\//, "");

/** Calls an admin route, says how it went, and polls the cluster again. */
async function act(what, path, opts) {
  try {
    const j = await api(path, opts);
    toast(what, "ok");
    refreshCluster();
    return j;
  } catch (e) {
    toast(`${what}: ${e.message}`, "bad", 7000);
    return null;
  }
}

/** The 4,096 slots as a strip, the runs a shard holds filled in. */
function slotStrip(sh, total) {
  const runs = Array.isArray(sh.slots) ? sh.slots : [];
  const strip = h("div", { style: { position: "relative", height: "8px", borderRadius: "4px", background: "var(--surface-3)", overflow: "hidden" }, title: runs.map(([a, b]) => `${a}–${b - 1}`).join(", ") });
  for (const [a, b] of runs) strip.append(h("div", { style: { position: "absolute", top: 0, bottom: 0, left: (a / total) * 100 + "%", width: Math.max(0.15, ((b - a) / total) * 100) + "%", background: "var(--accent)" } }));
  const held = sh.slots_held ?? 0;
  return h("div", { style: { margin: "2px 0 8px" } }, strip,
    h("div", { class: "small muted", style: { marginTop: "4px" } }, `${fmt.pct(held / total)} of the slots, in ${runs.length} run${runs.length === 1 ? "" : "s"}`));
}

function render(page) {
  const kubeBox = h("div"), sumBox = h("div"), shardsBox = h("div"), movesBox = h("div"), addBox = h("div"), setupBox = h("div");
  let kube = null, moves = null, movesErr = null;
  page.append(
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Cluster"), h("p", {}, "The shards, their copies and the slots each holds: add, remove, hand over and move them while the cluster serves."))),
    kubeBox, sumBox, shardsBox, movesBox, addBox, setupBox);

  const managed = () => !!kube?.managed;
  const byOperator = "The Kubernetes operator keeps this cluster as its StrataCluster says: change its size above, and the operator makes the change.";

  // ------------------------------------------------------------ Kubernetes
  async function loadKube() {
    if (!isCoordinator()) return;
    const was = managed();
    try { kube = await api("/v1/kube"); } catch (e) { kube = { managed: false, error: e }; }
    drawKube();
    if (managed() !== was) { draw(); drawSetup(); }
  }

  // The form keeps what is typed into it: it is built again only when the
  // spec changes and nothing was typed; the status beside it follows.
  let kubeForm = null, kubeSpec = null, typed = false;
  const kubeStatus = h("div");
  function drawKube() {
    if (!managed()) {
      kubeForm = null;
      clear(kubeBox, kube?.error && kube.error.status !== 404 ? h("div", { class: "error", style: { marginBottom: "14px" } }, "The Kubernetes link: " + kube.error.message) : null);
      return;
    }
    const sp = kube.spec || {}, st = kube.status || {};
    const phase = st.phase || "Pending";
    const lv = { Ready: "ok", Degraded: "degraded", Failed: "bad", Scaling: "info", Creating: "info" }[phase] || "info";
    const shardsAt = Array.isArray(st.shards) ? st.shards.length : null;
    clear(kubeStatus, h("div", { class: "row", style: { gap: "10px", alignItems: "flex-start" } },
      pill(lv, phase),
      h("div", { class: "grow" }, h("div", { style: { fontWeight: 600 } }, st.message || "waiting for the operator"),
        h("div", { class: "small muted" }, [shardsAt != null ? `the operator sees ${shardsAt} shard${shardsAt === 1 ? "" : "s"}` : null, kube.settled ? null : "it has not seen the latest spec yet"].filter(Boolean).join(" · ")))));
    const spec = JSON.stringify(sp);
    if (kubeForm && (typed || spec === kubeSpec)) return;
    kubeSpec = spec; typed = false;
    const num = (v, min, max) => h("input", { class: "input", type: "number", min, max, value: v, style: { width: "90px" }, oninput: () => (typed = true) });
    const shards = num(sp.shards ?? 1, 1, SLOTS), replicas = num(sp.replicas ?? 3, 1, 64), coords = num(sp.coordinators ?? 1, 1, 64);
    const apply = h("button", { class: "btn primary" }, icon("check"), "Apply");
    apply.onclick = async () => {
      const want = { shards: +shards.value, replicas: +replicas.value, coordinators: +coords.value };
      const was = { shards: sp.shards ?? 1, replicas: sp.replicas ?? 3, coordinators: sp.coordinators ?? 1 };
      const changes = Object.entries(want).filter(([k, v]) => v !== was[k]);
      if (!changes.length) { toast("Nothing to change"); return; }
      const said = changes.map(([k, v]) => `${{ shards: "shards", replicas: "copies per shard", coordinators: "coordinators" }[k]}: ${was[k]} → ${v}`);
      const fewer = want.shards < was.shards || want.replicas < was.replicas;
      const ok = await confirm({
        title: `Change ${kube.namespace}/${kube.name}`,
        body: h("div", { class: "stack" }, h("ul", { style: { margin: 0 } }, said.map((x) => h("li", {}, x))),
          h("p", { class: "dim", style: { margin: 0 } }, fewer
            ? "The operator drains a shard before it goes (its slots move to the others) and hands a primary over before its copy goes; the volumes of what goes are deleted."
            : "The operator starts the new copies, which join their shards and are sent a copy of the primary; new shards join and slots move to them.")),
        ok: "Apply", danger: fewer,
      });
      if (!ok) return;
      const j = await act("Asked the operator to change the cluster", "/v1/admin/kube", { body: want });
      if (j) { kube = j; typed = false; kubeForm = null; drawKube(); }
    };
    kubeForm = h("div", { class: "stack" }, kubeStatus,
      h("div", { class: "row", style: { gap: "14px", alignItems: "flex-end", flexWrap: "wrap" } },
        h("label", { class: "f" }, "Shards", shards), h("label", { class: "f" }, "Copies per shard", replicas), h("label", { class: "f" }, "Coordinators", coords), apply),
      h("dl", { class: "kv" }, [["image", sp.image || "strata-server:dev"], ["volume", sp.storage?.size || "10Gi"], ["failover after", sp.failoverAfterMs === 0 ? "by hand" : fmt.ms(sp.failoverAfterMs ?? 3000)], ["write acks", sp.writeAcks || "all"], ["token", sp.tokenSecret ? "Secret " + sp.tokenSecret : "none (open to the cluster's network)"]].flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])));
    clear(kubeBox, card({ title: "Kubernetes", sub: `StrataCluster ${kube.namespace}/${kube.name}: the operator makes what you change here, one step at a time`, attrs: { style: { marginBottom: "14px" } }, body: kubeForm }));
  }

  // ------------------------------------------------------------ the shards
  function drawSummary() {
    const c = state.cluster;
    if (!c || !isCoordinator()) { clear(sumBox); return; }
    const shards = c.shards || [];
    const copies = shards.reduce((n, s) => n + (s.replicas || []).length, 0);
    const down = shards.flatMap((s) => (s.replicas || []).filter((r) => r.status === "down"));
    const kpi = (lbl, val, sub) => card({ body: h("div", { class: "kpi", style: { padding: 0 } }, h("div", { class: "lbl" }, lbl), h("div", { class: "val" }, val), sub ? h("div", { class: "delta" }, sub) : null) });
    clear(sumBox, h("div", { class: "grid g4", style: { marginBottom: "14px" } },
      kpi("Shards", fmt.n(shards.length), c.status === "ok" ? "every one answers" : c.status),
      kpi("Copies", fmt.n(copies), down.length ? `${down.length} not answering` : "all answering"),
      kpi("Cluster map", "epoch " + fmt.n(c.slot_epoch), `${fmt.n(c.slot_count || SLOTS)} slots`),
      kpi("Changes", c.changes?.ok ? "allowed" : "not now", c.changes?.ok ? "the map can change" : c.changes?.why || "")));
  }

  function drawShards() {
    const c = state.cluster;
    if (!isCoordinator()) { clear(shardsBox); return; }
    if (!c) { clear(shardsBox, empty("layers", "Asking the shards", "")); return; }
    const shards = c.shards || [];
    const can = c.changes?.ok && !managed();
    const why = managed() ? byOperator : c.changes?.why;
    const btn = (label, ic, f, { danger = false, off = false, title } = {}) => h("button", { class: "btn sm" + (danger ? " danger" : ""), disabled: off, title, onclick: f }, ic ? icon(ic) : null, label);
    clear(shardsBox, h("div", { class: "grid g2", style: { marginBottom: "14px" } }, shards.map((sh) => {
      const held = sh.slots_held ?? 0;
      const rows = (sh.replicas || []).slice().sort((a, b) => (b.takes_writes ? 1 : 0) - (a.takes_writes ? 1 : 0));
      const primary = rows.find((r) => r.takes_writes);
      const lag = Object.fromEntries((primary?.detail?.health?.replication?.replicas || []).map((x) => [x.url, x]));
      const copyRow = (r) => {
        const rep = r.detail?.health?.replication || {};
        const f = lag[r.url];
        const state_ = r.status === "down" ? pill("down", "down") : r.takes_writes ? pill("ok", "primary") : f ? pill(f.in_sync ? "ok" : "warn", f.in_sync ? "in sync" : f.copying ? "being copied" : `${fmt.n(f.behind)} behind`) : pill("info", r.role);
        const tools = [];
        if (!r.takes_writes && r.status !== "down" && rows.length > 1) tools.push(btn("Make primary", "up", () => promote(sh, r), { title: "Hand the shard's primary to this copy (a majority of the copies vote)" }));
        if (rows.length > 1) tools.push(btn("Remove", "trash", () => removeCopy(sh, r), { danger: true, off: !can, title: can ? "Take this copy out of the shard" : why }));
        return h("div", { class: "row nw", style: { gap: "10px", padding: "7px 0", borderTop: "1px solid var(--line)" } },
          h("div", { class: "grow", style: { minWidth: 0 } },
            h("div", { class: "row nw", style: { gap: "8px" } }, state_, h("span", { class: "mono ellipsis", title: r.url }, short(r.url))),
            h("div", { class: "small muted", style: { marginTop: "2px" } }, r.status === "down" ? (r.error || "not answering") : `term ${fmt.n(r.term ?? rep.term)} · transaction ${fmt.n(rep.seq_no ?? r.txn)} · ${fmt.n(r.docs)} rows · ${fmt.ms(r.rtt_ms)}${r.avoided ? " · avoided after a failure" : ""}`)),
          h("div", { class: "row nw", style: { gap: "6px" } }, tools));
      };
      const last = shards.length > 1;
      return card({
        title: `Shard ${sh.shard}`, sub: `${fmt.n(held)} slots`,
        tools: [pill(sh.status === "ok" ? "ok" : sh.status === "down" ? "down" : "degraded", sh.status)],
        body: h("div", {}, slotStrip(sh, c.slot_count || SLOTS), rows.map(copyRow)),
        foot: h("div", { class: "row", style: { gap: "6px" } },
          btn("Add a copy", "plus", () => addCopy(sh), { off: !can, title: can ? "A new server joins this shard as a replica" : why }),
          btn("Drain", "down", () => drain(sh), { off: !can || !last || held === 0, title: !can ? why : !last ? "the only shard has nowhere to move its slots" : held === 0 ? "it holds no slot" : "Move its slots to the other shards, before removing it" }),
          btn("Remove shard", "trash", () => removeShard(sh), { danger: true, off: !can || held > 0 || !last, title: !can ? why : held > 0 ? "drain it first: it holds slots" : "Take this shard out of the cluster" })),
      });
    })));
  }

  async function promote(sh, r) {
    const ok = await confirm({ title: `Make ${short(r.url)} shard ${sh.shard}'s primary`, body: "The copy asks the shard's copies for their votes; the primary that serves now grants its vote, steps down and follows it. Writes pause for a moment. It is refused while another copy holds a later transaction.", ok: "Hand it over" });
    if (ok) await act(`${short(r.url)} is shard ${sh.shard}'s primary`, `/v1/admin/shards/${sh.shard}/promote`, { body: { url: r.url } });
  }
  async function removeCopy(sh, r) {
    const ok = await confirm({ title: `Remove ${short(r.url)} from shard ${sh.shard}`, body: r.takes_writes ? "It is the shard's primary: hand the primary to another copy first." : "The copy stops receiving the shard's writes; the server keeps its database, and can join a shard again only as a new server.", ok: "Remove", danger: true });
    if (ok) await act(`${short(r.url)} left shard ${sh.shard}`, `/v1/admin/shards/${sh.shard}/replicas?url=${encodeURIComponent(r.url)}`, { method: "DELETE" });
  }
  function addCopy(sh) {
    const url = h("input", { class: "input", placeholder: "http://10.0.0.7:7701", style: { width: "100%" } });
    const force = h("input", { type: "checkbox" });
    const go = h("button", { class: "btn primary" }, icon("plus"), "Add");
    const m = modal({ title: `A new copy of shard ${sh.shard}`, body: [
      h("p", { class: "dim", style: { margin: 0 } }, "A server started empty with the cluster's first table (strata-server --db … --seq-no --grpc ADDRESS --schema …) joins as a replica: it is sent a copy of the primary, then follows its writes."),
      h("label", { class: "f" }, "Its gRPC URL", url),
      h("label", { class: "row nw small dim", style: { gap: "6px" } }, force, "It holds rows already (they are replaced by the copy)"),
    ], foot: [h("button", { class: "btn", onclick: () => m.close() }, "Cancel"), go] });
    go.onclick = async () => {
      if (!url.value.trim()) { url.focus(); return; }
      if (await act(`${url.value.trim()} joins shard ${sh.shard}`, `/v1/admin/shards/${sh.shard}/replicas`, { body: { url: url.value.trim(), force: force.checked } })) m.close();
    };
  }
  async function drain(sh) {
    const ok = await confirm({ title: `Drain shard ${sh.shard}`, body: `Its ${fmt.n(sh.slots_held)} slots move to the other shards, a run at a time, while the cluster serves. Then it can be removed.`, ok: "Drain" });
    if (ok) { await act(`Draining shard ${sh.shard}`, `/v1/admin/shards/${sh.shard}/drain`, { body: {} }); loadMoves(); }
  }
  async function removeShard(sh) {
    const ok = await confirm({ title: `Remove shard ${sh.shard}`, body: "It holds no slot: the cluster stops asking it. Its servers keep their databases.", ok: "Remove", danger: true });
    if (ok) await act(`Shard ${sh.shard} left the cluster`, `/v1/admin/shards/${sh.shard}`, { method: "DELETE" });
  }

  // ------------------------------------------------------------ moves
  async function loadMoves() {
    if (!isCoordinator()) return;
    try { moves = await api("/v1/admin/moves"); movesErr = null; } catch (e) { moves = null; movesErr = e; }
    drawMoves();
  }

  function drawMoves() {
    if (!isCoordinator()) { clear(movesBox); return; }
    const c = state.cluster;
    const counts = (c?.shards || []).map((s) => s.slots_held ?? 0);
    const spread = counts.length ? Math.max(...counts) - Math.min(...counts) : 0;
    const job = moves?.job;
    const running = job && job.state === "running";
    const rebalance = h("button", { class: "btn sm", disabled: !!running || managed() || counts.length < 2 || spread <= 1, title: managed() ? byOperator : spread <= 1 ? "every shard holds the same number of slots, give or take one" : "Move slots until every shard holds the same number" }, icon("layers"), "Even out the slots");
    rebalance.onclick = async () => { if (await confirm({ title: "Even out the slots", body: "Slots move from the shards holding the most to those holding the fewest, a run at a time, while the cluster serves.", ok: "Move them" })) { await act("Evening out the slots", "/v1/admin/rebalance", { body: {} }); loadMoves(); } };
    const stop = h("button", { class: "btn sm danger", onclick: async () => { await act("The moves stop after the one under way", "/v1/admin/moves/stop", { body: {} }); loadMoves(); } }, icon("stop"), "Stop");
    let body;
    if (movesErr) body = failure(movesErr, "The moves");
    else if (!job) body = empty("layers", "No move of slots under way", spread > 1 ? `The shards hold ${Math.min(...counts)} to ${Math.max(...counts)} slots.` : "Every shard holds the same number of slots, give or take one.");
    else {
      body = h("div", { class: "stack" },
        h("div", { class: "row", style: { gap: "10px" } }, pill(job.state === "done" ? "ok" : job.state === "failed" ? "bad" : job.state === "stopped" ? "warn" : "info", job.state), h("b", {}, { rebalance: "Evening out", drain: "Draining", move: "Moving" }[job.kind] || job.kind),
          h("span", { class: "dim" }, `${fmt.n(job.done)} of ${fmt.n(job.of)} moves · ${fmt.n(job.slots_moved)} slots · ${fmt.n(job.rows_moved)} rows`), job.error ? h("span", { class: "small", style: { color: "var(--crit-ink)" } }, job.error) : null),
        h("div", { style: { maxHeight: "220px", overflow: "auto" } }, (job.moves || []).map((m) => h("div", { class: "row nw small", style: { gap: "8px", padding: "3px 0" } },
          pill(m.phase === "done" ? "ok" : m.phase === "failed" || m.phase === "given up" ? "bad" : m.phase === "planned" ? "disabled" : "info", m.phase), h("span", {}, `${fmt.n(m.count)} slots, shard ${m.from} → ${m.to}`),
          h("span", { class: "muted" }, [m.rows ? `${fmt.n(m.rows)} rows` : null, m.held_ms ? `writes held ${fmt.ms(m.held_ms)}` : null, m.ms ? fmt.ms(m.ms) : null, m.error].filter(Boolean).join(" · "))))));
    }
    clear(movesBox, card({ title: "Moves of slots", sub: "a shard serves while its slots move: each run is copied, caught up, then switched over while its writes wait a moment", tools: [running ? stop : null, rebalance], body, attrs: { style: { marginBottom: "14px" } } }));
  }

  // ------------------------------------------------------------ a new shard
  let addGo = null;
  function drawAdd() {
    if (!isCoordinator() || managed()) { addGo = null; clear(addBox); return; }
    const c = state.cluster;
    if (addGo) { addGo.disabled = !c?.changes?.ok; addGo.title = c?.changes?.ok ? "" : c?.changes?.why || ""; return; }
    const servers = h("textarea", { class: "input", rows: 3, placeholder: "http://10.0.0.8:7701\nhttp://10.0.0.9:7701" });
    let acks = "all";
    const acksSeg = seg([["all", "every copy in sync"], ["majority", "a majority"], ["primary", "the primary"]], acks, (v) => (acks = v));
    const go = addGo = h("button", { class: "btn primary", disabled: !c?.changes?.ok, title: c?.changes?.ok ? "" : c?.changes?.why || "" }, icon("plus"), "Add the shard");
    go.onclick = async () => {
      const list = servers.value.split(/[\s,]+/).map((x) => x.trim()).filter(Boolean);
      if (!list.length) { servers.focus(); return; }
      const j = await act(`Shard added, served by ${list.map(short).join(", ")}`, "/v1/admin/shards", { body: { servers: list, write_acks: acks } });
      if (j) {
        servers.value = "";
        if (await confirm({ title: "Move slots to it?", body: "The new shard holds no slot yet, so it serves nothing. Even out the slots now?", ok: "Even them out" })) { await act("Evening out the slots", "/v1/admin/rebalance", { body: {} }); loadMoves(); }
      }
    };
    clear(addBox, card({
      title: "Add a shard", sub: "servers started empty with the cluster's first table and --seq-no; the first is its primary, the others its replicas",
      attrs: { style: { marginBottom: "14px" } },
      body: h("div", { class: "stack" },
        h("label", { class: "f" }, "Its servers' gRPC URLs, one a line", servers),
        h("div", { class: "row", style: { gap: "10px" } }, h("span", { class: "small muted" }, "A write is answered once it is on"), acksSeg, h("div", { class: "grow" }), go)),
    }));
  }

  // ------------------------------------------------------------ setting one up
  // A server holding a shard of a cluster has its place in the map.
  const kind = () => !state.health ? null : isCoordinator() ? "coordinator" : state.health.slots?.shard != null ? "in a cluster" : "alone";
  let setupFor = null, setupGen = 0;
  async function drawSetup() {
    const hl = state.health || {};
    const gen = ++setupGen;
    setupFor = kind();
    if (managed()) { clear(setupBox); return; }
    if (setupFor === "in a cluster") {
      // A shard's server: the cluster is changed from a coordinator.
      const sl = hl.slots || {}, rep = hl.replication || {};
      clear(setupBox, card({
        title: "This server holds a shard", sub: "the cluster is changed from a coordinator's command center",
        body: h("div", { class: "stack" },
          h("dl", { class: "kv" }, [["shard", sl.shard != null ? `${sl.shard} of ${sl.of}` : "–"], ["cluster map", sl.epoch != null ? "epoch " + sl.epoch : "–"], ["role", rep.role || "standalone"], ["term", rep.term ?? "–"], ["last transaction", fmt.n(rep.seq_no ?? hl.txn)]].flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, String(v))])),
          h("p", { class: "dim", style: { margin: 0 } }, "Open the Cluster view of a coordinator over this cluster (strata-server --shards with any of its servers) to add or remove shards and copies, hand primaries over and move slots.")),
      }));
      return;
    }
    let schema = null;
    try { schema = (await api("/v1/schema")).short; } catch { /* the demo then */ }
    if (gen !== setupGen) return; // drawn again meanwhile
    const coordinator = isCoordinator();
    const name = h("input", { class: "input", value: "strata", style: { width: "140px" } });
    const ns = h("input", { class: "input", value: "strata", style: { width: "140px" } });
    const n = (v, min, max) => h("input", { class: "input", type: "number", value: v, min, max, style: { width: "80px" } });
    const shards = n(Math.max(2, state.cluster?.shards?.length || 2), 1, SLOTS), copies = n(3, 1, 64), size = h("input", { class: "input", value: "10Gi", style: { width: "90px" } });
    let first = schema ? "schema" : "demo";
    const firstSeg = seg([["schema", "this server's table", "the first table's schema, from this server"], ["demo", "demo products"]], first, (v) => { first = v; out(); });
    const token = h("input", { type: "checkbox", checked: true });
    const pre = h("pre", { class: "code", style: { maxHeight: "380px", overflow: "auto", margin: 0 } });
    const machines = h("pre", { class: "code", style: { maxHeight: "260px", overflow: "auto", margin: 0 } });
    const yaml = () => {
      const nm = name.value.trim() || "strata", nsv = ns.value.trim() || "strata";
      const t = token.checked;
      return [
        "# 1. The operator (once per Kubernetes cluster), from the Strata repository:",
        "#    kubectl apply -f strata-server/deploy/k8s/operator/operator.yaml",
        t ? `# 2. A token every server and coordinator requires:\n#    kubectl create namespace ${nsv}\n#    kubectl -n ${nsv} create secret generic ${nm}-token --from-literal=token="$(openssl rand -hex 32)"` : "# 2. (No token: anything that reaches the pods may read and write.)",
        "# 3. This file: kubectl apply -f cluster.yaml",
        "apiVersion: v1", "kind: Namespace", "metadata:", `  name: ${nsv}`, "---",
        "apiVersion: strata.chloetech.ai/v1alpha1", "kind: StrataCluster", "metadata:", `  name: ${nm}`, `  namespace: ${nsv}`, "spec:",
        `  shards: ${+shards.value || 1}`, `  replicas: ${+copies.value || 1}`, "  coordinators: 1", "  storage:", `    size: ${size.value.trim() || "10Gi"}`,
        first === "schema" && schema ? `  schema: ${JSON.stringify(schema)}` : "  demo: 10000",
        "  failoverAfterMs: 3000", "  writeAcks: all", t ? `  tokenSecret: ${nm}-token` : null,
        "# Then: kubectl -n " + nsv + " get strata " + nm + " -o wide; kubectl -n " + nsv + " port-forward svc/" + nm + " 7700",
      ].filter((x) => x != null).join("\n");
    };
    const script = () => {
      const k = +shards.value || 1, r = +copies.value || 1;
      const auth = token.checked ? " --token \"$STRATA_TOKEN\"" : " --no-token";
      const serve = "strata-server --db /data/s.strata --seq-no --listen 0.0.0.0:7700 --grpc 0.0.0.0:7701 --failover-after-ms 3000" + auth;
      const lines = first === "demo" && k > 0
        ? [`# On the machine of shard I's first copy (shardI-0), demo products split over ${k}:`, `#   ${serve} --demo 10000 --demo-shard I/${k}`, "# and on each other copy's machine (shardI-K), the demo's schema (GET /v1/schema of a first copy, its \"schema\"):", `#   ${serve} --schema schema.json`]
        : ["# On each machine of shard I, copy K (shardI-K: a database each, the same schema file):", `#   ${serve} --schema schema.json`];
      lines.push("# Then one coordinator over the first copy of each shard (it lays the cluster out):");
      const urls = Array.from({ length: k }, (_, i) => `http://shard${i}-0:7701`).join(",");
      lines.push(`strata-server --shards ${urls} --listen 0.0.0.0:7700${auth}`);
      if (r > 1) {
        lines.push("# and the other copies join their shards, from this page or:");
        for (let i = 0; i < k; i++) for (let c = 1; c < r; c++) lines.push(`curl -X POST ${token.checked ? "-H \"authorization: Bearer $STRATA_TOKEN\" " : ""}http://coordinator:7700/v1/admin/shards/${i}/replicas -d '{"url": "http://shard${i}-${c}:7701"}'`);
      }
      if (schema && first === "schema") lines.push("# schema.json:", "# " + JSON.stringify(schema));
      return lines.join("\n");
    };
    const out = () => { pre.textContent = yaml(); machines.textContent = script(); };
    for (const e of [name, ns, shards, copies, size]) e.oninput = out;
    token.onchange = out;
    out();
    clear(setupBox, card({
      title: coordinator ? "Run it on Kubernetes" : "Make a cluster",
      sub: coordinator ? "the operator makes the same cluster and keeps it as its file says" : "this server holds one database; a cluster spreads tables over shards of a few copies each, behind coordinators",
      body: h("div", { class: "stack" },
        h("div", { class: "row", style: { gap: "14px", alignItems: "flex-end", flexWrap: "wrap" } },
          h("label", { class: "f" }, "Name", name), h("label", { class: "f" }, "Namespace", ns), h("label", { class: "f" }, "Shards", shards), h("label", { class: "f" }, "Copies per shard", copies), h("label", { class: "f" }, "Volume", size),
          h("div", { class: "f" }, h("span", { class: "small muted" }, "First table"), firstSeg), h("label", { class: "row nw small dim", style: { gap: "6px" } }, token, "require a token")),
        h("div", { class: "row", style: { justifyContent: "space-between" } }, h("b", {}, "On Kubernetes"), h("button", { class: "btn sm", onclick: () => copy(pre.textContent, "Copied the StrataCluster") }, icon("copy"), "Copy")), pre,
        coordinator ? null : h("div", { class: "row", style: { justifyContent: "space-between" } }, h("b", {}, "On machines of your own"), h("button", { class: "btn sm", onclick: () => copy(machines.textContent, "Copied the commands") }, icon("copy"), "Copy")),
        coordinator ? null : machines),
    }));
  }

  const draw = () => { drawSummary(); drawShards(); drawMoves(); drawAdd(); };
  draw();
  drawSetup();
  loadKube();
  // What this server is (a coordinator, a shard's, one on its own) is
  // known once the first health poll is.
  let wasCoordinator = isCoordinator();
  const offs = [
    bus.on("cluster", draw),
    bus.on("health", () => {
      if (isCoordinator() !== wasCoordinator) { wasCoordinator = isCoordinator(); draw(); loadKube(); }
      if (kind() !== setupFor) drawSetup();
    }),
    every(5000, loadKube, false),
    every(2000, loadMoves),
  ];
  return () => offs.forEach((f) => f());
}
