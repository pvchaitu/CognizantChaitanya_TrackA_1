#include <iostream>
#include <string>

#include "scheduler.h"

// Track A interface:
//   ./mlsys <path_to_input.json> <path_to_output.json>
// Timeout is enforced externally by the organizer harness.

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./mlsys <input.json> <output.json>\n";
    return 2;
  }
  try {
    systema::SolveOne(argv[1], argv[2]);
  } catch (const std::exception& e) {
    // Always try to fail loudly; harness will treat missing output as invalid.
    std::cerr << "Fatal: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
