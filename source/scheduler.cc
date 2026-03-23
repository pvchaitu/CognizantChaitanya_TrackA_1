#include "scheduler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <sys/resource.h>

#include "mini_json.h"
#include "render_table.h"
#include "srs_ann.h"

namespace systema {

// -----------------------------
// Helpers
// -----------------------------

static inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

static inline int64_t tile_counts(int64_t W, int64_t H, int64_t w, int64_t h) {
  return ceil_div(W, w) * ceil_div(H, h);
}

static inline int64_t slice_bytes(int64_t h, int64_t w) {
  // Contest model treats area directly (no element-size factor).
  return h * w;
}

static inline std::tuple<int64_t, int64_t, int64_t> matmul_dims(const mlsys::Problem& p, const mlsys::Op& op) {
  // op.inputs[0]=LHS (HxK), op.inputs[1]=RHS (KxW), output is HxW
  size_t lhs = op.inputs.at(0);
  size_t rhs = op.inputs.at(1);
  int64_t H = p.tensors[lhs].height;
  int64_t K = p.tensors[lhs].width;
  int64_t W = p.tensors[rhs].width;
  return {H, W, K};
}

static inline int64_t op_working_set(const mlsys::Problem& p, const mlsys::Op& op, int64_t w, int64_t h, int64_t k) {
  if (op.op_type == "Pointwise") {
    // need input slice + output slice
    return slice_bytes(h, w) + slice_bytes(h, w);
  }
  // MatMul
  auto [H, W, K] = matmul_dims(p, op);
  int64_t kk = std::min<int64_t>(k, K);
  int64_t lhs = slice_bytes(h, kk);
  int64_t rhs = slice_bytes(kk, w);
  int64_t out = slice_bytes(h, w);
  return lhs + rhs + out;
}

struct Boundary {
  std::vector<size_t> boundary_in;
  std::vector<size_t> boundary_out;
  std::unordered_set<size_t> produced;
  std::unordered_set<size_t> consumed;
};

static Boundary subgraph_boundary(const mlsys::Problem& p, const std::vector<size_t>& ops) {
  Boundary b;
  for (size_t oi : ops) {
    const auto& op = p.ops[oi];
    for (size_t t : op.outputs) b.produced.insert(t);
    for (size_t t : op.inputs) b.consumed.insert(t);
  }
  for (size_t t : b.consumed) if (!b.produced.count(t)) b.boundary_in.push_back(t);
  for (size_t t : b.produced) if (!b.consumed.count(t)) b.boundary_out.push_back(t);
  std::sort(b.boundary_in.begin(), b.boundary_in.end());
  std::sort(b.boundary_out.begin(), b.boundary_out.end());
  return b;
}

// Simple topo by tensor dependencies.
static std::vector<size_t> topo_order(const mlsys::Problem& p) {
  size_t n = p.ops.size();
  std::unordered_map<size_t, size_t> produces;
  produces.reserve(n * 2);
  for (size_t oi = 0; oi < n; ++oi) {
    for (size_t t : p.ops[oi].outputs) produces[t] = oi;
  }
  std::vector<int> indeg(n, 0);
  std::vector<std::vector<size_t>> adj(n);
  for (size_t j = 0; j < n; ++j) {
    for (size_t t : p.ops[j].inputs) {
      auto it = produces.find(t);
      if (it != produces.end()) {
        size_t i = it->second;
        adj[i].push_back(j);
        indeg[j] += 1;
      }
    }
  }
  std::vector<size_t> q;
  for (size_t i = 0; i < n; ++i) if (indeg[i] == 0) q.push_back(i);
  std::vector<size_t> out;
  while (!q.empty()) {
    size_t i = q.back();
    q.pop_back();
    out.push_back(i);
    for (size_t j : adj[i]) {
      indeg[j] -= 1;
      if (indeg[j] == 0) q.push_back(j);
    }
  }
  if (out.size() != n) {
    out.clear();
    for (size_t i = 0; i < n; ++i) out.push_back(i);
  }
  return out;
}

static bool fits_gran(const mlsys::Problem& p,
                      const std::vector<size_t>& ops,
                      int64_t w, int64_t h, int64_t k,
                      const std::unordered_set<size_t>& retain_in) {
  int64_t op_ws = 0;
  for (size_t oi : ops) {
    op_ws = std::max<int64_t>(op_ws, op_working_set(p, p.ops[oi], w, h, k));
  }
  Boundary b = subgraph_boundary(p, ops);
  int64_t ws2 = 0;
  for (size_t t : b.boundary_in) {
    if (retain_in.count(t)) continue;
    ws2 += slice_bytes(h, w);
  }
  for (size_t t : b.boundary_out) {
    ws2 += slice_bytes(h, w);
  }
  return (ws2 + op_ws) <= p.fast_memory_capacity;
}

static mlsys::Granularity choose_granularity_for_subgraph(const mlsys::Problem& p,
                                                         const std::vector<size_t>& ops,
                                                         const std::unordered_set<size_t>& retain_in,
                                                         const std::unordered_set<size_t>& /*retain_out*/) {
  // Start with native w,h and for k use full K or smaller if needed.
  int64_t w = p.native_granularity.width;
  int64_t h = p.native_granularity.height;

  Boundary b = subgraph_boundary(p, ops);

  // Conservative: include slices for all boundary inputs that are NOT retained from prev.
  int64_t ws = 0;
  for (size_t t : b.boundary_in) {
    if (retain_in.count(t)) continue;
    ws += slice_bytes(h, w);
  }
  // include boundary outputs
  for (size_t t : b.boundary_out) ws += slice_bytes(h, w);

  bool has_mm = false;
  int64_t k = 1;
  int64_t minK = std::numeric_limits<int64_t>::max();
  for (size_t oi : ops) {
    if (p.ops[oi].op_type == "MatMul") {
      has_mm = true;
      auto [Hm, Wm, Km] = matmul_dims(p, p.ops[oi]);
      (void)Hm; (void)Wm;
      minK = std::min<int64_t>(minK, Km);
    }
  }
  if (has_mm) {
    k = minK;
    while (k > 1) {
      int64_t op_ws = 0;
      for (size_t oi : ops) op_ws = std::max<int64_t>(op_ws, op_working_set(p, p.ops[oi], w, h, k));
      if (ws + op_ws <= p.fast_memory_capacity) break;
      k = std::max<int64_t>(1, k / 2);
    }
  }

  // final check; if still OOM, reduce w/h
  while (!fits_gran(p, ops, w, h, k, retain_in)) {
    if (w > 1 && w >= h) w = std::max<int64_t>(1, w / 2);
    else if (h > 1) h = std::max<int64_t>(1, h / 2);
    else break;
  }

  mlsys::Granularity g;
  g.width = w;
  g.height = h;
  g.depth = std::max<int64_t>(1, k);
  return g;
}

static double subgraph_latency(const mlsys::Problem& p,
                              const std::vector<size_t>& ops,
                              const mlsys::Granularity& g,
                              const std::unordered_set<size_t>& retain_in,
                              const std::unordered_set<size_t>& retain_out) {
  int64_t w = g.width, h = g.height, k = g.depth;
  Boundary b = subgraph_boundary(p, ops);

  // Use output tensor of the last op for tiling grid size proxy.
  size_t last_out = p.ops[ops.back()].outputs.at(0);
  int64_t OW = p.tensors[last_out].width;
  int64_t OH = p.tensors[last_out].height;
  int64_t tiles = tile_counts(OW, OH, w, h);

  double compute = 0.0;
  double mem_in_per_tile = 0.0;
  double mem_out_per_tile = 0.0;
  double B = double(p.slow_memory_bandwidth);

  for (size_t t : b.boundary_in) {
    if (retain_in.count(t)) continue;
    mem_in_per_tile += double(slice_bytes(h, w)) / B;
  }

  for (size_t t : b.boundary_out) {
    if (retain_out.count(t)) continue;
    mem_out_per_tile += double(slice_bytes(h, w)) / B;
  }

  for (size_t oi : ops) {
    const auto& op = p.ops[oi];
    if (op.op_type == "Pointwise") {
      compute += double(op.base_cost);
    } else {
      auto [Hm, Wm, Km] = matmul_dims(p, op);
      (void)Hm; (void)Wm;
      int64_t kk = std::max<int64_t>(1, std::min<int64_t>(k, Km));
      int64_t splitk = ceil_div(Km, kk);
      compute += double(op.base_cost) * double(splitk);
    }
  }

  double per_tile = std::max(compute, mem_in_per_tile + mem_out_per_tile);
  return per_tile * double(tiles);
}

// Deterministic "semantic geometry" vector for ranking fusions.
static std::vector<float> embed_subgraph(const mlsys::Problem& p,
                                        const std::vector<size_t>& ops,
                                        const mlsys::Granularity& g,
                                        double lat) {
  int64_t w = g.width, h = g.height, k = g.depth;
  float n_ops = float(ops.size());
  float n_mm = 0.f;
  for (size_t oi : ops) if (p.ops[oi].op_type == "MatMul") n_mm += 1.f;
  float n_pw = n_ops - n_mm;

  int64_t max_ws = 0;
  for (size_t oi : ops) max_ws = std::max<int64_t>(max_ws, op_working_set(p, p.ops[oi], w, h, k));
  float slack = 0.f;
  if (p.fast_memory_capacity > 0) {
    slack = float(std::max<double>(0.0, (double(p.fast_memory_capacity) - double(max_ws)) / double(p.fast_memory_capacity)));
  }

  std::vector<float> v = {
      n_ops / 10.0f,
      n_mm / 10.0f,
      n_pw / 10.0f,
      float(std::log1p(lat) / 10.0),
      float(double(w) / double(p.native_granularity.width)),
      float(double(h) / double(p.native_granularity.height)),
      float(std::log1p(double(k)) / 10.0),
      slack,
  };
  return v;
}

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double da = 0, db = 0, dot = 0;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    dot += double(a[i]) * double(b[i]);
    da += double(a[i]) * double(a[i]);
    db += double(b[i]) * double(b[i]);
  }
  da = std::sqrt(da) + 1e-12;
  db = std::sqrt(db) + 1e-12;
  return float(dot / (da * db));
}

static void atomic_write(const std::string& path, const std::string& content) {
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    f.write(content.data(), (std::streamsize)content.size());
  }
  std::remove(path.c_str());
  std::rename(tmp.c_str(), path.c_str());
}

// -----------------------------
// JSON IO
// -----------------------------

mlsys::Problem ReadProblemJson(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("Cannot open input.json: " + path);
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  mini_json::Value root = mini_json::parse(s);
  if (!root.is_object()) throw std::runtime_error("Problem JSON must be an object");
  const auto& o = root.as_object();

  auto get_arr = [&](const std::string& k) -> const mini_json::Array& {
    auto it = o.find(k);
    if (it == o.end() || !it->second.is_array()) throw std::runtime_error("Missing/invalid array: " + k);
    return it->second.as_array();
  };
  auto get_num = [&](const std::string& k) -> int64_t {
    auto it = o.find(k);
    if (it == o.end() || !it->second.is_number()) throw std::runtime_error("Missing/invalid number: " + k);
    return (int64_t)std::llround(it->second.as_number());
  };

  const auto& widths = get_arr("widths");
  const auto& heights = get_arr("heights");
  if (widths.size() != heights.size()) throw std::runtime_error("widths/heights mismatch");

  mlsys::Problem p;
  p.tensors.resize(widths.size());
  for (size_t i = 0; i < widths.size(); ++i) {
    p.tensors[i].width = (int64_t)std::llround(widths[i].as_number());
    p.tensors[i].height = (int64_t)std::llround(heights[i].as_number());
  }

  const auto& op_types = get_arr("op_types");
  const auto& inputs = get_arr("inputs");
  const auto& outputs = get_arr("outputs");
  const auto& base_costs = get_arr("base_costs");
  if (!(op_types.size() == inputs.size() && inputs.size() == outputs.size() && outputs.size() == base_costs.size())) {
    throw std::runtime_error("op arrays size mismatch");
  }

  p.ops.resize(op_types.size());
  for (size_t i = 0; i < p.ops.size(); ++i) {
    p.ops[i].op_type = op_types[i].as_string();
    // inputs[i] is array of tensor indices
    for (const auto& v : inputs[i].as_array()) p.ops[i].inputs.push_back((size_t)std::llround(v.as_number()));
    for (const auto& v : outputs[i].as_array()) p.ops[i].outputs.push_back((size_t)std::llround(v.as_number()));
    p.ops[i].base_cost = (int64_t)std::llround(base_costs[i].as_number());
  }

  p.fast_memory_capacity = get_num("fast_memory_capacity");
  p.slow_memory_bandwidth = get_num("slow_memory_bandwidth");

  const auto& ng = get_arr("native_granularity");
  if (ng.size() != 2) throw std::runtime_error("native_granularity must have 2 elements");
  p.native_granularity.width = (int64_t)std::llround(ng[0].as_number());
  p.native_granularity.height = (int64_t)std::llround(ng[1].as_number());
  p.native_granularity.depth = 1;

  return p;
}

void WriteScheduleJson(const std::string& path, const ScheduleJSON& sch) {
  // Output format is specified in PROBLEM.md: parallel lists.
  mini_json::Object root;

  // subgraphs
  mini_json::Array subgraphs;
  for (const auto& sg : sch.subgraphs) {
    mini_json::Array a;
    for (size_t oi : sg) a.emplace_back((double)oi);
    subgraphs.emplace_back(a);
  }
  root["subgraphs"] = subgraphs;

  // granularities
  mini_json::Array grans;
  for (const auto& g : sch.granularities) {
    mini_json::Array a;
    a.emplace_back((double)g.width);
    a.emplace_back((double)g.height);
    a.emplace_back((double)g.depth);
    grans.emplace_back(a);
  }
  root["granularities"] = grans;

  // tensors_to_retain
  mini_json::Array retains;
  for (const auto& r : sch.tensors_to_retain) {
    mini_json::Array a;
    for (size_t t : r) a.emplace_back((double)t);
    retains.emplace_back(a);
  }
  root["tensors_to_retain"] = retains;

  // traversal_orders
  mini_json::Array travs;
  for (const auto& t : sch.traversal_orders) {
    if (!t.has_value()) {
      travs.emplace_back(nullptr);
    } else {
      mini_json::Array a;
      for (int64_t x : t.value()) a.emplace_back((double)x);
      travs.emplace_back(a);
    }
  }
  root["traversal_orders"] = travs;

  // subgraph_latencies
  mini_json::Array lats;
  for (double x : sch.subgraph_latencies) lats.emplace_back(x);
  root["subgraph_latencies"] = lats;

  std::string content = mini_json::dumps(mini_json::Value(root));
  atomic_write(path, content);
}

// -----------------------------
// Baseline schedule + improvements (SRS + HNSW guided adjacency fusions)
// -----------------------------

static ScheduleJSON baseline_schedule(const mlsys::Problem& p) {
  std::vector<size_t> order = topo_order(p);
  ScheduleJSON sch;
  for (size_t oi : order) sch.subgraphs.push_back({oi});
  sch.tensors_to_retain.assign(sch.subgraphs.size(), {});
  sch.traversal_orders.assign(sch.subgraphs.size(), std::nullopt);

  sch.granularities.clear();
  sch.subgraph_latencies.clear();

  std::unordered_set<size_t> retain_in;
  for (const auto& sg : sch.subgraphs) {
    mlsys::Granularity g = choose_granularity_for_subgraph(p, sg, retain_in, {});
    sch.granularities.push_back(g);
    sch.subgraph_latencies.push_back(subgraph_latency(p, sg, g, retain_in, {}));
    retain_in.clear();
  }
  return sch;
}

static double total_latency(const ScheduleJSON& sch) {
  return std::accumulate(sch.subgraph_latencies.begin(), sch.subgraph_latencies.end(), 0.0);
}

static ScheduleJSON compute_schedule_for_subgraphs(const mlsys::Problem& p, const std::vector<std::vector<size_t>>& sgs) {
  ScheduleJSON sch;
  sch.subgraphs = sgs;
  sch.tensors_to_retain.assign(sgs.size(), {});
  sch.traversal_orders.assign(sgs.size(), std::nullopt);

  std::unordered_set<size_t> retain_in;
  for (const auto& sg : sgs) {
    mlsys::Granularity g = choose_granularity_for_subgraph(p, sg, retain_in, {});
    sch.granularities.push_back(g);
    sch.subgraph_latencies.push_back(subgraph_latency(p, sg, g, retain_in, {}));
    retain_in.clear();
  }
  return sch;
}

static double try_fuse_adjacent_anytime(const mlsys::Problem& p,
                                       ScheduleJSON& best,
                                       const std::string& output_path,
                                       std::chrono::steady_clock::time_point deadline) {
  // Prototype vector for "good": more ops fused, high slack, lower latency.
  // Used as fallback when HNSW is unavailable or empty.
  const std::vector<float> proto = {0.8f, 0.6f, 0.2f, 0.1f, 1.0f, 1.0f, 0.2f, 0.8f};

  srs::SRSAnnIndex ann((int)proto.size(), 5000);
  ann.add(proto, "seed_proto");

  double best_total = total_latency(best);

  // Anytime: baseline is already written by caller.
  bool improved = true;
  while (improved && std::chrono::steady_clock::now() < deadline) {
    improved = false;

    struct Candidate {
      float score;
      size_t i;
      std::vector<size_t> fused;
      mlsys::Granularity g;
      double lat;
      std::vector<float> vec;
    };

    std::vector<Candidate> candidates;
    for (size_t i = 0; i + 1 < best.subgraphs.size(); ++i) {
      std::vector<size_t> fused = best.subgraphs[i];
      fused.insert(fused.end(), best.subgraphs[i + 1].begin(), best.subgraphs[i + 1].end());

      mlsys::Granularity g = choose_granularity_for_subgraph(p, fused, {}, {});
      double lat = subgraph_latency(p, fused, g, {}, {});
      std::vector<float> vec = embed_subgraph(p, fused, g, lat);

      float ann_sim = ann.query_best_similarity(vec, 3);
      float score = (std::isinf(ann_sim) ? cosine(vec, proto) : ann_sim);
      candidates.push_back({score, i, std::move(fused), g, lat, std::move(vec)});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return a.score > b.score;
    });

    size_t K = std::min<size_t>(6, candidates.size());
    for (size_t c = 0; c < K && std::chrono::steady_clock::now() < deadline; ++c) {
      const auto& cand = candidates[c];
      std::vector<std::vector<size_t>> new_sgs;
      new_sgs.reserve(best.subgraphs.size() - 1);
      for (size_t j = 0; j < cand.i; ++j) new_sgs.push_back(best.subgraphs[j]);
      new_sgs.push_back(cand.fused);
      for (size_t j = cand.i + 2; j < best.subgraphs.size(); ++j) new_sgs.push_back(best.subgraphs[j]);

      ScheduleJSON trial = compute_schedule_for_subgraphs(p, new_sgs);
      double new_total = total_latency(trial);
      if (new_total < best_total) {
        best = std::move(trial);
        best_total = new_total;
        improved = true;

        // Online learning: add accepted fusion embedding to ANN.
        ann.add(cand.vec, "accepted_fusion");

        // Overwrite after every improvement (kill-safe).
        WriteScheduleJson(output_path, best);
        break;
      }
    }
  }

  return best_total;
}

static std::chrono::steady_clock::time_point infer_deadline() {
  // Track A harness kills the binary after benchmark-specific timeout.
  // We do NOT receive timeout as an argument.
  // Best effort: if RLIMIT_CPU is set, use it as an upper bound.
  // Otherwise, fall back to a conservative local budget.
  rlimit lim;
  bool have = (getrlimit(RLIMIT_CPU, &lim) == 0);
  double seconds = 1.0;  // fallback for local runs
  if (have && lim.rlim_cur != RLIM_INFINITY) {
    seconds = std::max<double>(0.05, double(lim.rlim_cur) * 0.98);
  }
  auto start = std::chrono::steady_clock::now();
  return start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(seconds));
}

void SolveOne(const std::string& input_path, const std::string& output_path) {
  auto start = std::chrono::steady_clock::now();
  auto deadline = infer_deadline();

  mlsys::Problem p = ReadProblemJson(input_path);

  // 1) Write a valid baseline schedule immediately.
  ScheduleJSON best = baseline_schedule(p);
  WriteScheduleJson(output_path, best);

  // 2) Try greedy local improvements (safe fusions + k-tuning) until time budget.
  double total = try_fuse_adjacent_anytime(p, best, output_path, deadline);

  // Optional local debug print (disabled by default).
  (void)total;
  (void)start;
}

}  // namespace systema
