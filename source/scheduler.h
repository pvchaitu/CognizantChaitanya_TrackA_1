#pragma once
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "mlsys.h"

namespace systema {

// -----------------------------
// Ported baseline scheduler from systema_runner (Python).
// -----------------------------

struct ScheduleJSON {
  std::vector<std::vector<size_t>> subgraphs;
  std::vector<mlsys::Granularity> granularities;
  std::vector<std::vector<size_t>> tensors_to_retain;
  std::vector<std::optional<mlsys::TraversalOrder>> traversal_orders;
  std::vector<double> subgraph_latencies;
};

mlsys::Problem ReadProblemJson(const std::string& path);
void WriteScheduleJson(const std::string& path, const ScheduleJSON& sch);

// Solve one benchmark: reads problem and writes a schedule.
void SolveOne(const std::string& input_path, const std::string& output_path);

}  // namespace systema
