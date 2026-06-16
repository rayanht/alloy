cmake_minimum_required(VERSION 3.21)
project(alloy_metal_ext LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

nanobind_add_module(_metal_ext csrc/alloy_metal.mm csrc/alloy_core.mm)
target_link_libraries(_metal_ext PRIVATE
    ${FOUNDATION_LIBRARY}
    ${METAL_FRAMEWORK}
)

install(TARGETS _metal_ext LIBRARY DESTINATION alloy/_runtime)
