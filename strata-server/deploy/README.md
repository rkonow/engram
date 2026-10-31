# strata-server on Kubernetes

An operator that runs Strata clusters on Kubernetes and changes them as
their resource says (below), a Docker image, the manifests of a fixed
layout of shards behind a coordinator, and a script that runs them on a
local multi-node cluster
([kind](https://kind.sigs.k8s.io/), Kubernetes in Docker) to test the
distributed behaviour: whether the shards answer as one database would,
what the scatter-gather costs, what happens when a shard or one of its
replicas dies, how to move to more shards, and what network latency does.

| file | what |
|---|---|
| `Dockerfile` | strata-server and strata-server-bench on Debian slim (157 MB); build from the repository root |
| `k8s/shards.yaml` | a set of N shards × R replicas: a headless Service, and per shard a StatefulSet of R pods on different nodes; replica r of shard i serves `/data/<collection>.shard<i>of<N>.strata` (r = 0) or its own copy `...shard<i>of<N>.r<r>.strata` |
| `k8s/coordinator.yaml` | the coordinator, a Deployment pointed at one set of shards (`--shards`, replicas of a shard joined by `\|`) |
| `k8s/single.yaml` | one database holding the whole collection, to compare answers and latency with |
| `k8s/operator/operator.yaml` | the operator: its namespace (`strata-system`), service account, cluster role (which includes making each cluster's Role for its coordinators) and Deployment |
| `k8s/operator/crd.json` | the `StrataCluster` resource definition, which the operator installs when it starts (`strata-server operator --print-crd` writes it) |
| `k8s/operator/cluster.yaml` | a cluster for it to make: 2 shards of 3 copies, 10,000 demo products |
| `kind-test.sh` | creates the kind cluster and runs each step below |

## The operator

`strata-server operator` keeps each `StrataCluster` as its spec says. It
makes a StatefulSet of copies per shard (each copy a pod with a volume of
its own), their headless Service, and the coordinators (a Deployment
behind the Service named as the cluster, port 7700: the API and the
command center), then lays the cluster out and changes it through the
coordinator's admin calls, one step every few seconds, writing what it
found in the resource's status:

```bash
kubectl apply -f strata-server/deploy/k8s/operator/operator.yaml
kubectl apply -f strata-server/deploy/k8s/operator/cluster.yaml
kubectl -n strata get strata -o wide          # NAME SHARDS COPIES PHASE MESSAGE
kubectl -n strata patch strata demo --type merge -p '{"spec": {"shards": 3, "replicas": 2}}'
kubectl -n strata port-forward svc/demo 7700  # http://127.0.0.1:7700
```

| spec | default | |
|---|---|---|
| `shards` | 1 | shards, each holding a part of the 4,096 slots |
| `replicas` | 3 | copies of each shard: a primary and its replicas, on different nodes when there are |
| `coordinators` | 1 | coordinators behind the cluster's Service |
| `schema` or `demo` | | the first table's schema (as `--schema` takes it), or this many demo products split over the shards the cluster starts with |
| `storage` | `{size: 10Gi}` | each copy's volume (`size`, `storageClassName`) |
| `failoverAfterMs` | 3000 | every server's `--failover-after-ms`: a shard elects a new primary on its own when its primary's pod dies |
| `writeAcks` | `all` | every primary's `--write-acks` |
| `tokenSecret` | | a Secret whose key `token` every server and coordinator requires; without it they run with `--no-token`, open to whatever reaches them |
| `image`, `imagePullPolicy`, `resources`, `coordinatorResources`, `args` | `strata-server:dev` | the servers' image, containers and further arguments |

What a change does:

- **More shards**: a StatefulSet for each new shard; once its first copy
  is ready the shard joins the cluster, then slots move to it until each
  shard holds the same number (give or take one), while the cluster
  serves. **Fewer**: the newest shard is drained (its slots move to the
  others), leaves the cluster, and its StatefulSet and volumes are deleted.
- **More copies**: the StatefulSets grow, and each new pod, started empty
  with the schema, joins its shard as a replica and is sent a copy of the
  primary. **Fewer**: the highest pod of each set leaves its shard first
  (when it is the primary, the operator hands the primary over to a replica
  that answers), then the set shrinks and the pod's volume is deleted.
- A pod that comes back without its volume (its node was replaced) leaves
  its shard and joins again, and is sent a copy.
- The coordinators' `--shards` list follows the cluster's servers, so that
  a coordinator that starts again finds the cluster; their number follows
  `coordinators`.

The coordinators run as the service account `<cluster>-coordinator`, which
a Role lets read and patch their own `StrataCluster` and nothing else: the
command center's **Cluster** view (`kubectl -n strata port-forward svc/demo
7700`, then `http://127.0.0.1:7700/#/cluster`) shows the operator's phase
and message and changes the shards, copies per shard and coordinators,
which the operator then makes as above.

The operator does not choose primaries: the copies of a shard elect one
among themselves ([failover](../../docs/strata/server.md#failover)), so a
dead pod costs a few seconds of writes to its shard whether the operator
runs or not. It takes no step while slots are moving or the cluster map
cannot change, and the status says why (`Creating`, `Scaling`, `Ready`,
`Degraded` with the servers that do not answer, or `Failed` with what is
wrong with the spec). Deleting a `StrataCluster` deletes its pods and keeps
its volumes (`data-<cluster>-s<shard>-<copy>`).

It reaches each cluster's coordinators through their Service, so it runs
in the cluster; `--namespace` limits it to one namespace, and `--no-crd`
leaves the resource definition to you. On kind:

```bash
strata-server/deploy/kind-test.sh up          # no STRATA_WORK or STRATA_DATA needed for the operator
strata-server/deploy/kind-test.sh image
strata-server/deploy/kind-test.sh operator    # follows the cluster until it is Ready
strata-server/deploy/kind-test.sh scale 3 3   # a third shard
strata-server/deploy/kind-test.sh scale 2 2   # back to two, a copy fewer each
```

Its tests: `tests/operator.rs` runs the same steps with server processes
standing in for pods (a cluster made, grown, healed and shrunk, every row
kept), and `tests/operator_kube.rs` makes a cluster's objects on a real API
server (`STRATA_ENVTEST_DIR`: a directory with `etcd` and `kube-apiserver`,
as `setup-envtest` downloads them).

## The fixed layout

The rest of this page runs Strata from the manifests under `k8s/` without
the operator, to test what sharding costs.

The databases are files on a volume: the manifests read them from a
`hostPath` at `/data`, which `kind-test.sh up` mounts from the Mac into every
node, so a shard pod finds its file on whichever node it lands. On a real
cluster, replace the `hostPath` with a PersistentVolume per shard.

## The test

From the repository root, with Docker running, `kind` and `kubectl`
installed, and Quora prepared by `strata/scripts/prepare_beir.py`:

```bash
export STRATA_DATA=/path/to/beir STRATA_WORK=/path/outside/the/repo
cd strata-server
# The unsplit database and its splits into 2, 3 and 4 shards (natively; minutes each).
for n in 2 3 4; do cargo run --release --bin strata-server-bench -- \
    --collection quora --work "$STRATA_WORK" --shards $n --reuse --build-only; done
cd ..
strata-server/deploy/kind-test.sh up          # 1 control plane + 3 workers
strata-server/deploy/kind-test.sh image
strata-server/deploy/kind-test.sh single
strata-server/deploy/kind-test.sh shards 2           # 2 shards, 1 replica each (shards 2 2: two each)
strata-server/deploy/kind-test.sh coordinator 2      # coordinator 2 2 for two replicas each
# Time both and compare their answers, from a client pod in the cluster.
strata-server/deploy/kind-test.sh bench two --collection quora --out /data/results \
    --single http://strata-single:7700 --coordinator http://strata-coordinator:7700 --label "2 shards"
```

Then, from the same client, the failure and resharding tests (each a
per-second timeline while one action happens; pin the client with
`BENCH_NODE` to a node you are not about to freeze):

```bash
T=strata-server/deploy/kind-test.sh
Q="--collection quora --out /data/results --concurrency 4"      # bash (word splitting)
C="--coordinator http://strata-coordinator:7700"
BENCH_NODE=strata-worker $T bench crash $Q $C --watch 40 --label "watch crash" &
sleep 15; $T crash strata-shards-2-s1-0; wait       # also: delete POD, pause POD 20
$T shards 3                                         # the new set beside the old
$T bench warm $Q --single http://strata-shards-3-s0-0.strata-shards-3:7700 --watch 20 --label warm   # warm each new shard
$T coordinator 3                                    # move the coordinator (under a --watch run)
$T delay strata-shards-3-s0-0 5                     # +5 ms to every packet sent to that shard; 0 removes it
$T coordinator 2 2 --partial --shard-timeout-ms=300 # answer without a shard that has no replica left
```

The `--watch` output prints one line a second (answers, errors, p50, p99,
the first error). Results of a full run on a Mac are in
[`../results/k8s-mac/README.md`](../results/k8s-mac/README.md).

## Security

These manifests serve a test cluster open: every server runs with
`--no-token`, which a server listening beyond loopback needs when it has no
token. Anything that can reach a pod can read and write. On a real cluster,
put a cluster token in a Secret and give it to every server, coordinators
included (they send it to each other, and only it makes the calls between
servers), give the coordinators an admin token of their own, drop
`--no-token`, and serve TLS. Clients get API keys made on a coordinator,
which every server then holds:

```bash
kubectl -n strata create secret generic strata-cluster-token --from-literal=token="$(openssl rand -hex 32)"
kubectl -n strata create secret generic strata-token --from-literal=token="$(openssl rand -hex 32)"
```

```yaml
          env:
            - name: STRATA_CLUSTER_TOKEN
              valueFrom: {secretKeyRef: {name: strata-cluster-token, key: token}}
            # the coordinators only:
            - name: STRATA_TOKEN
              valueFrom: {secretKeyRef: {name: strata-token, key: token}}
```

With `--tls-cert` and `--tls-key` on the shards (a certificate for their
Service names, from cert-manager say), list them as `https://` in the
coordinator's `--shards` and give it the CA with `--shard-ca`. A
NetworkPolicy that lets only the coordinator reach the shards' gRPC port
is worth adding too.
