// Alloy Metal runtime core — nanobind-free.
//
// The device, buffer table, pipeline cache, plan registry, and the plan
// register/dispatch execution path. Shared SOURCE between two binaries that
// never share runtime state: the Python nanobind extension (alloy_metal.mm
// shims parse nb::list → these plain structs → call here) and the on-device
// Swift engine (app/AlloyEngine wraps these in a C API). One implementation of
// the dispatch core; each binary gets its own Metal state.
//
// Nothing here includes nanobind, so it compiles in Xcode without the Python
// toolchain.

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace alloycore {

// --- buffers ---------------------------------------------------------------

struct AllocatedBuffer {
  void *ptr;               // data pointer (== metal_buf.contents + mtl_offset)
  size_t nbytes;           // requested size (not rounded)
  size_t aligned_size;     // Metal buffer length (rounded to 16KB)
  id<MTLBuffer> metal_buf; // Metal-allocated buffer — the source of truth
  int64_t handle;
  bool released;     // true = handle released, Metal backing dropped
  size_t mtl_offset; // byte offset of ptr within metal_buf (pool slices)
  bool slice;        // true = view into a pool; release must not purge
  bool vm_owned;     // true = pool backed by our mach_vm reservation
};

// --- plans -----------------------------------------------------------------

struct PlanDispatch {
  id<MTLComputePipelineState> pso;
  std::vector<uint32_t> buf_indices; // index into Plan.slots
  std::vector<int64_t> buf_offsets;  // byte offset per buffer
  MTLSize grid, tg;
};

struct PlanSlot {
  enum Type { INPUT = 0, WEIGHT = 1, INTERMEDIATE = 2 };
  Type type;
  int input_arg_idx;       // for INPUT: which arg
  id<MTLBuffer> metal_buf; // resolved at dispatch time
  int64_t base_offset;     // resolved at dispatch time
  int64_t handle;          // alloy buffer handle (direct Metal buffer lookup)
  uintptr_t ptr;           // raw pointer for profiled get_buffer path
  size_t nbytes;           // buffer size
};

struct Plan {
  std::vector<std::vector<PlanDispatch>> groups; // dependency groups
  std::vector<PlanSlot> slots;
  std::unordered_set<int> written_slots; // slots written by any dispatch
  std::vector<std::pair<int, int>> flat_to_group; // flat dispatch idx -> (group, j)
};

// --- plain specs the nb shims / C API marshal into --------------------------

struct SlotSpec {
  int type;             // PlanSlot::Type
  int input_arg_idx;
  int64_t handle_or_ptr; // WEIGHT/INTERMEDIATE: alloy buffer handle; INPUT: 0
  size_t nbytes;
};

struct DispatchSpec {
  int64_t pso_handle;
  std::vector<uint32_t> buf_indices;
  std::vector<int64_t> buf_offsets;
  int grid[3];
  int tg[3];
};

struct InputUpdate {
  int slot_idx;
  int64_t handle;
  int64_t offset;
};

struct PreCopy {
  id<MTLBuffer> src_buf;
  int64_t src_offset;
  id<MTLBuffer> dst_buf;
  int64_t dst_offset;
  int64_t nbytes;
};

struct PreCopySpec {
  int64_t dst_handle, dst_offset, src_handle, src_offset, nbytes;
};

struct GridUpdate {
  int flat_idx, gx, gy, gz;
};

struct DeviceInfo {
  std::string name, gpu_family;
  int max_threads_per_threadgroup;
  int max_threadgroup_memory_length;
  int64_t recommended_max_working_set_size;
  bool has_bfloat16;
};

struct PlanTiming {
  double gpu_ms, encode_ms, wait_ms;
  int n_groups;
};

// --- shared global Metal state (defined in alloy_core.mm) -------------------

extern id<MTLDevice> g_device;
extern id<MTLCommandQueue> g_queue;
extern std::string g_device_name;
extern std::string g_gpu_family;

extern std::unordered_map<int64_t, id<MTLComputePipelineState>> g_pipelines;
extern std::unordered_map<int64_t, id<MTLLibrary>> g_libraries;
extern std::unordered_map<int64_t, std::string> g_pipeline_names;
extern int64_t g_next_handle;
extern std::mutex g_cache_mutex;

extern std::unordered_map<uintptr_t, id<MTLBuffer>> g_buffer_cache;
extern std::mutex g_buffer_mutex;

extern std::unordered_map<int64_t, AllocatedBuffer> g_alloy_buffers;
extern std::unordered_set<uintptr_t> g_alloy_ptrs;
extern std::unordered_map<uintptr_t, int64_t> g_ptr_to_handle;
extern int64_t g_next_buf_handle;
extern std::mutex g_buf_mutex;

extern std::unordered_map<int64_t, Plan> g_plans;
extern int64_t g_next_plan_handle;
extern std::mutex g_plan_mutex;

extern id<MTLCommandBuffer> g_pending_cb;

// --- core API (nanobind-free) ----------------------------------------------

void ensure_device();
DeviceInfo device_info();

int64_t buf_alloc(size_t nbytes);
void buf_release(int64_t handle);
uintptr_t buf_ptr(int64_t handle);
int64_t buf_handle_for_ptr(uintptr_t ptr);
size_t buf_nbytes(int64_t handle);

int64_t compile_msl(const std::string &source, const std::string &function_name);
int64_t compile_metallib(const std::string &path, const std::string &function_name);

NSUInteger plan_buffer_offset(const PlanSlot &slot, const PlanDispatch &dispatch,
                              size_t i);
void update_plan_input_slots(Plan *plan, const std::vector<InputUpdate> &updates);
void update_plan_dispatch_grids(Plan *plan, const std::vector<GridUpdate> &updates);

int64_t register_plan(const std::vector<DispatchSpec> &dispatches,
                      const std::vector<SlotSpec> &slots,
                      const std::vector<std::vector<int>> &groups,
                      const std::vector<int> &written_slot_indices);

void release_plan(int64_t plan_handle);

PlanTiming dispatch_plan(int64_t plan_handle,
                         const std::vector<InputUpdate> &input_arrays,
                         bool serialized = false, bool defer_wait = false,
                         const std::vector<PreCopySpec> &pre_copies = {},
                         const std::vector<GridUpdate> &grid_updates = {});

void gpu_sync();

} // namespace alloycore
