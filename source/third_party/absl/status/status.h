#ifndef THIRD_PARTY_ABSL_STATUS_STATUS_H_
#define THIRD_PARTY_ABSL_STATUS_STATUS_H_
#include <string>

namespace absl {

class Status {
 public:
  enum Code {
    kOk = 0,
    kUnknown = 2,
    kInvalidArgument = 3,
    kInternal = 13,
  };

  Status() : code_(kOk) {}
  Status(Code code, std::string msg) : code_(code), msg_(std::move(msg)) {}

  static Status OK() { return Status(); }

  bool ok() const { return code_ == kOk; }
  Code code() const { return code_; }
  const std::string& message() const { return msg_; }

 private:
  Code code_;
  std::string msg_;
};

}  // namespace absl

#endif  // THIRD_PARTY_ABSL_STATUS_STATUS_H_
