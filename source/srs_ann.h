#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Optional ANN accelerator: hnswlib (C++).
// This mirrors the Python runner:
// - SRS decides what constitutes a reusable decision (embedding) and when it is safe to apply.
// - HNSW provides fast nearest-neighbour retrieval over stored "good" embeddings.
// If hnswlib is unavailable, we fall back to brute-force cosine over stored winners.

#if __has_include("third_party/hnswlib/hnswlib.h")
  #include "third_party/hnswlib/hnswlib.h"
  #define MLSYS_HAS_HNSWLIB 1
#else
  #define MLSYS_HAS_HNSWLIB 0
#endif

namespace srs {

inline float dot(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0.f;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline void l2_normalize(std::vector<float>& v) {
  double ss = 0.0;
  for (float x : v) ss += double(x) * double(x);
  double inv = 1.0 / (std::sqrt(ss) + 1e-12);
  for (float& x : v) x = float(x * inv);
}

class SRSAnnIndex {
 public:
  explicit SRSAnnIndex(int dim, int max_elements = 10000)
      : dim_(dim), max_elements_(max_elements) {
#if MLSYS_HAS_HNSWLIB
    space_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(space_.get(), max_elements_);
    // Default ef; may be tuned.
    index_->setEf(50);
#endif
  }

  size_t size() const { return next_id_; }

  void add(std::vector<float> vec, const std::string& meta = "") {
    // For cosine similarity with InnerProductSpace, store normalized vectors.
    l2_normalize(vec);

#if MLSYS_HAS_HNSWLIB
    index_->addPoint(vec.data(), (hnswlib::labeltype)next_id_);
#else
    winners_.push_back(std::move(vec));
#endif
    meta_.push_back(meta);
    next_id_++;
  }

  float query_best_similarity(std::vector<float> vec, int k = 1) const {
    if (next_id_ == 0) return -std::numeric_limits<float>::infinity();
    l2_normalize(vec);

#if MLSYS_HAS_HNSWLIB
    // In hnswlib InnerProductSpace, the returned distance is (1 - dot).
    // So similarity ≈ dot = 1 - distance.
    auto res = index_->searchKnn(vec.data(), std::min<int>(k, (int)next_id_));
    if (res.empty()) return -std::numeric_limits<float>::infinity();
    float best_dist = res.top().first;  // smallest is best
    float sim = 1.0f - best_dist;
    return sim;
#else
    float best = -std::numeric_limits<float>::infinity();
    for (const auto& w : winners_) best = std::max(best, dot(vec, w));
    return best;
#endif
  }

 private:
  int dim_;
  int max_elements_;
  size_t next_id_ = 0;
  std::vector<std::string> meta_;

#if MLSYS_HAS_HNSWLIB
  std::unique_ptr<hnswlib::InnerProductSpace> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
#else
  std::vector<std::vector<float>> winners_;
#endif
};

}  // namespace srs
