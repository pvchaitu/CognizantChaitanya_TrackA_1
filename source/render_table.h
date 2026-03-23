#pragma once
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// render_table: minimal table printer similar to unified_sdnf_experiment_hybrid_v8.render_table
// We keep this only for readability / local debugging; it is NOT used by the judge.

inline std::string render_table(const std::vector<std::string>& headers,
                                const std::vector<std::vector<std::string>>& rows,
                                const std::string& title = "") {
  std::vector<size_t> widths(headers.size(), 0);
  for (size_t i = 0; i < headers.size(); ++i) widths[i] = headers[i].size();
  for (const auto& r : rows) {
    for (size_t i = 0; i < headers.size() && i < r.size(); ++i) {
      widths[i] = std::max(widths[i], r[i].size());
    }
  }
  auto line = [&](char ch) {
    std::ostringstream os;
    os << "+";
    for (size_t i = 0; i < widths.size(); ++i) {
      os << std::string(widths[i] + 2, ch) << "+";
    }
    return os.str();
  };
  auto fmt_row = [&](const std::vector<std::string>& r) {
    std::ostringstream os;
    os << "|";
    for (size_t i = 0; i < widths.size(); ++i) {
      std::string cell = (i < r.size() ? r[i] : "");
      os << " " << cell << std::string(widths[i] - cell.size(), ' ') << " |";
    }
    return os.str();
  };

  std::ostringstream out;
  if (!title.empty()) out << title << "\n";
  out << line('-') << "\n";
  out << fmt_row(headers) << "\n";
  out << line('=') << "\n";
  for (const auto& r : rows) out << fmt_row(r) << "\n" << line('-') << "\n";
  return out.str();
}
