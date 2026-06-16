cmake_minimum_required(VERSION 3.21)
project(alloy LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)

# The code lives in the workspace packages one level up. This recipe assembles
# them into a single `alloy` wheel WITHOUT moving anything in the repo — the
# packages/ layout is the dev/source-of-truth; this is purely a packaging view.
set(WS "${CMAKE_CURRENT_SOURCE_DIR}/../packages")

# Build the Metal extension fresh (so a stale dev-built .so in the source tree
# is never the one that ships — we exclude .so from the python copy below).
nanobind_add_module(_metal_ext ${WS}/alloy-metal/csrc/alloy_metal.mm ${WS}/alloy-metal/csrc/alloy_core.mm)
target_link_libraries(_metal_ext PRIVATE ${FOUNDATION_LIBRARY} ${METAL_FRAMEWORK})
install(TARGETS _metal_ext LIBRARY DESTINATION alloy/_runtime)

# The four import packages (alloy = kernels, alloy_torch = backend,
# alloy_server = serving/generation stack, alloy_cli = CLI). FILES_MATCHING
# copies sources + data (configs, models.json) and skips .so / __pycache__
# implicitly.
install(DIRECTORY ${WS}/alloy-metal/src/alloy/         DESTINATION alloy
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi" PATTERN "*.json" PATTERN "*.jinja")
install(DIRECTORY ${WS}/alloy-torch/src/alloy_torch/   DESTINATION alloy_torch
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi" PATTERN "*.json" PATTERN "*.jinja")
install(DIRECTORY ${WS}/alloy-server/src/alloy_server/ DESTINATION alloy_server
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi" PATTERN "*.json" PATTERN "*.jinja")
install(DIRECTORY ${WS}/alloy-cli/src/alloy_cli/       DESTINATION alloy_cli
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi" PATTERN "*.json" PATTERN "*.jinja")
