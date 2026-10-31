// The views, in the order the navigation shows them.

import overview from "./overview.js";
import topology from "./topology.js";
import performance from "./performance.js";
import alerts from "./alerts.js";
import queries from "./queries.js";
import profile from "./profile.js";
import logs from "./logs.js";
import cluster from "./cluster.js";
import access from "./access.js";
import system from "./system.js";
import sql from "./sql.js";
import builder from "./builder.js";
import schema from "./schema.js";
import importer from "./import.js";
import relevance from "./relevance.js";
import recsys from "./recsys.js";
import sid from "./sid.js";
import autocomplete from "./autocomplete.js";
import media from "./media.js";
import graph from "./graph.js";
import classic from "./classic.js";

export const GROUPS = ["Monitor", "Workload", "Data", "Tune", "Explore", "Admin", "More"];
export const VIEWS = [overview, topology, performance, alerts, queries, profile, logs, sql, builder, schema, importer, relevance, recsys, sid, autocomplete, media, graph, cluster, access, system, classic];
