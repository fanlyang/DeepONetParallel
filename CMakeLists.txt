cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(DEAL_II_DIR "/opt/dealii-9.7.1-codim/lib/cmake/deal.II"
    CACHE PATH "Path to deal.IIConfig.cmake in fanyoung/dealii-deeponet")
find_package(deal.II 9.7.1 REQUIRED HINTS
  "${DEAL_II_DIR}" "$ENV{DEAL_II_DIR}")
deal_ii_initialize_cached_variables()
project(torus_mg_jacobi LANGUAGES CXX)
if(DEAL_II_WITH_CXX20)
  message(FATAL_ERROR
    "The selected deal.II install was built with C++20. "
    "Use the image's C++17 install at "
    "/opt/dealii-9.7.1-codim/lib/cmake/deal.II")
endif()
if(NOT DEAL_II_WITH_MPI)
  message(FATAL_ERROR "This benchmark requires an MPI-enabled deal.II build.")
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(torus_mg_jacobi src/main.cpp)
target_compile_features(torus_mg_jacobi PRIVATE cxx_std_17)
target_link_libraries(torus_mg_jacobi MPI::MPI_CXX)
deal_ii_setup_target(torus_mg_jacobi)
