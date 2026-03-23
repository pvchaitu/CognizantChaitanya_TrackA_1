#pragma once
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// mini_json: small, self-contained JSON parser sufficient for MLSys contest I/O.
// Supports: null, bool, number (as double), string (basic escapes), array, object.
// This is intentionally tiny to keep the submission self-contained.

namespace mini_json {

struct Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

struct Value {
  using Storage = std::variant<std::nullptr_t, bool, double, std::string, Array, Object>;
  Storage v;

  Value() : v(nullptr) {}
  Value(std::nullptr_t) : v(nullptr) {}
  Value(bool b) : v(b) {}
  Value(double d) : v(d) {}
  Value(int64_t i) : v(double(i)) {}
  Value(std::string s) : v(std::move(s)) {}
  Value(const char* s) : v(std::string(s)) {}
  Value(Array a) : v(std::move(a)) {}
  Value(Object o) : v(std::move(o)) {}

  bool is_null() const { return std::holds_alternative<std::nullptr_t>(v); }
  bool is_bool() const { return std::holds_alternative<bool>(v); }
  bool is_number() const { return std::holds_alternative<double>(v); }
  bool is_string() const { return std::holds_alternative<std::string>(v); }
  bool is_array() const { return std::holds_alternative<Array>(v); }
  bool is_object() const { return std::holds_alternative<Object>(v); }

  const Array& as_array() const { return std::get<Array>(v); }
  const Object& as_object() const { return std::get<Object>(v); }
  const std::string& as_string() const { return std::get<std::string>(v); }
  double as_number() const { return std::get<double>(v); }
  bool as_bool() const { return std::get<bool>(v); }

  Array& as_array() { return std::get<Array>(v); }
  Object& as_object() { return std::get<Object>(v); }
  std::string& as_string() { return std::get<std::string>(v); }
};

class Parser {
 public:
  explicit Parser(std::string s) : s_(std::move(s)) {}

  Value parse() {
    i_ = 0;
    skip_ws();
    Value v = parse_value();
    skip_ws();
    if (i_ != s_.size()) throw std::runtime_error("Trailing characters in JSON");
    return v;
  }

 private:
  std::string s_;
  size_t i_ = 0;

  void skip_ws() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) i_++;
  }

  char peek() const { return (i_ < s_.size()) ? s_[i_] : '\0'; }
  char get() { return (i_ < s_.size()) ? s_[i_++] : '\0'; }

  void expect(char c) {
    if (get() != c) throw std::runtime_error(std::string("Expected '") + c + "'");
  }

  Value parse_value() {
    skip_ws();
    char c = peek();
    if (c == 'n') return parse_null();
    if (c == 't' || c == 'f') return parse_bool();
    if (c == '"') return parse_string();
    if (c == '[') return parse_array();
    if (c == '{') return parse_object();
    // number
    return parse_number();
  }

  Value parse_null() {
    if (s_.compare(i_, 4, "null") != 0) throw std::runtime_error("Invalid token");
    i_ += 4;
    return Value(nullptr);
  }

  Value parse_bool() {
    if (s_.compare(i_, 4, "true") == 0) { i_ += 4; return Value(true); }
    if (s_.compare(i_, 5, "false") == 0) { i_ += 5; return Value(false); }
    throw std::runtime_error("Invalid bool");
  }

  Value parse_number() {
    size_t start = i_;
    if (peek() == '-') get();
    if (!std::isdigit(static_cast<unsigned char>(peek()))) throw std::runtime_error("Invalid number");
    while (std::isdigit(static_cast<unsigned char>(peek()))) get();
    if (peek() == '.') {
      get();
      while (std::isdigit(static_cast<unsigned char>(peek()))) get();
    }
    if (peek() == 'e' || peek() == 'E') {
      get();
      if (peek() == '+' || peek() == '-') get();
      while (std::isdigit(static_cast<unsigned char>(peek()))) get();
    }
    double val = std::strtod(s_.c_str() + start, nullptr);
    return Value(val);
  }

  Value parse_string() {
    expect('"');
    std::string out;
    while (true) {
      if (i_ >= s_.size()) throw std::runtime_error("Unterminated string");
      char c = get();
      if (c == '"') break;
      if (c == '\\') {
        if (i_ >= s_.size()) throw std::runtime_error("Bad escape");
        char e = get();
        switch (e) {
          case '"': out.push_back('"'); break;
          case '\\': out.push_back('\\'); break;
          case '/': out.push_back('/'); break;
          case 'b': out.push_back('\b'); break;
          case 'f': out.push_back('\f'); break;
          case 'n': out.push_back('\n'); break;
          case 'r': out.push_back('\r'); break;
          case 't': out.push_back('\t'); break;
          // We ignore \uXXXX for simplicity; contest inputs don't need it.
          default: out.push_back(e); break;
        }
      } else {
        out.push_back(c);
      }
    }
    return Value(std::move(out));
  }

  Value parse_array() {
    expect('[');
    skip_ws();
    Array arr;
    if (peek() == ']') { get(); return Value(std::move(arr)); }
    while (true) {
      arr.push_back(parse_value());
      skip_ws();
      char c = get();
      if (c == ']') break;
      if (c != ',') throw std::runtime_error("Expected ',' or ']'");
      skip_ws();
    }
    return Value(std::move(arr));
  }

  Value parse_object() {
    expect('{');
    skip_ws();
    Object obj;
    if (peek() == '}') { get(); return Value(std::move(obj)); }
    while (true) {
      if (peek() != '"') throw std::runtime_error("Expected string key");
      std::string key = parse_string().as_string();
      skip_ws();
      expect(':');
      skip_ws();
      obj.emplace(std::move(key), parse_value());
      skip_ws();
      char c = get();
      if (c == '}') break;
      if (c != ',') throw std::runtime_error("Expected ',' or '}'");
      skip_ws();
    }
    return Value(std::move(obj));
  }
};

inline Value parse(const std::string& s) {
  return Parser(s).parse();
}

// Writer: compact JSON for contest output.
inline void dump_string(std::string& out, const std::string& s) {
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out.push_back(c); break;
    }
  }
  out.push_back('"');
}

inline void dump(std::string& out, const Value& v);

inline void dump_array(std::string& out, const Array& a) {
  out.push_back('[');
  for (size_t i = 0; i < a.size(); ++i) {
    if (i) out.push_back(',');
    dump(out, a[i]);
  }
  out.push_back(']');
}

inline void dump_object(std::string& out, const Object& o) {
  out.push_back('{');
  bool first = true;
  for (const auto& kv : o) {
    if (!first) out.push_back(',');
    first = false;
    dump_string(out, kv.first);
    out.push_back(':');
    dump(out, kv.second);
  }
  out.push_back('}');
}

inline void dump(std::string& out, const Value& v) {
  if (v.is_null()) { out += "null"; return; }
  if (v.is_bool()) { out += (v.as_bool() ? "true" : "false"); return; }
  if (v.is_number()) {
    // Use a stable-ish representation.
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10g", v.as_number());
    out += buf;
    return;
  }
  if (v.is_string()) { dump_string(out, v.as_string()); return; }
  if (v.is_array()) { dump_array(out, v.as_array()); return; }
  dump_object(out, v.as_object());
}

inline std::string dumps(const Value& v) {
  std::string out;
  dump(out, v);
  return out;
}

}  // namespace mini_json
