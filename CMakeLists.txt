cmake_minimum_required(VERSION 3.16)
project(mlsys_solution CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mlsys
  source/main.cc
  source/scheduler.cc
)

# Include your own source directory
target_include_directories(mlsys PRIVATE source)

# Add hnswlib headers (assuming you cloned hnswlib into project root)
target_include_directories(mlsys PRIVATE ${PROJECT_SOURCE_DIR}/hnswlib)

# Compiler optimizations
target_compile_options(mlsys PRIVATE -O3 -DNDEBUG)

# For static-ish builds you can try:
# target_link_options(mlsys PRIVATE -static -static-libstdc++ -static-libgcc)
# Note: fully static glibc builds may require a musl toolchain.
