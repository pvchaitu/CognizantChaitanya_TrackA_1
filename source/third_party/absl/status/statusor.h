#ifndef THIRD_PARTY_ABSL_STATUS_STATUSOR_H_
#define THIRD_PARTY_ABSL_STATUS_STATUSOR_H_
#include <utility>
#include "third_party/absl/status/status.h"

namespace absl {

template <typename T>
class StatusOr {
 public:
  StatusOr(const Status& s) : status_(s), has_value_(false) {}
  StatusOr(Status&& s) : status_(std::move(s)), has_value_(false) {}
  StatusOr(const T& v) : status_(Status::OK()), value_(v), has_value_(true) {}
  StatusOr(T&& v) : status_(Status::OK()), value_(std::move(v)), has_value_(true) {}

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

  const T& value() const { return value_; }
  T& value() { return value_; }

  // Convenience for older absl style.
  const T& operator*() const { return value_; }
  T& operator*() { return value_; }

 private:
  Status status_;
  T value_{};
  bool has_value_;
};

}  // namespace absl

#endif  // THIRD_PARTY_ABSL_STATUS_STATUSOR_H_
