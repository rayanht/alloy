// Alloy Metal runtime core — implementation (nanobind-free). See alloy_core.h.

#include "alloy_core.h"

#include <chrono>
#include <stdexcept>

#include <mach/mach.h>
// mach/mach_vm.h is unsupported on the iOS SDK; vm_deallocate (mach/mach.h)
// covers the only use here (returning a paged-pool reservation's VA) on both
// platforms. The engine never creates vm_owned pool buffers, but the branch
// must still compile for iOS.

namespace alloycore {

// --- global Metal state ----------------------------------------------------

id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
std::string g_device_name;
std::string g_gpu_family;
static std::once_flag g_init_flag;

std::unordered_map<int64_t, id<MTLComputePipelineState>> g_pipelines;
std::unordered_map<int64_t, id<MTLLibrary>> g_libraries;
std::unordered_map<int64_t, std::string> g_pipeline_names;
int64_t g_next_handle = 1;
std::mutex g_cache_mutex;

std::unordered_map<uintptr_t, id<MTLBuffer>> g_buffer_cache;
std::mutex g_buffer_mutex;

std::unordered_map<int64_t, AllocatedBuffer> g_alloy_buffers;
std::unordered_set<uintptr_t> g_alloy_ptrs;
std::unordered_map<uintptr_t, int64_t> g_ptr_to_handle;
int64_t g_next_buf_handle = 1;
std::mutex g_buf_mutex;

std::unordered_map<int64_t, Plan> g_plans;
int64_t g_next_plan_handle = 1;
std::mutex g_plan_mutex;

id<MTLCommandBuffer> g_pending_cb = nil;

// --- device ----------------------------------------------------------------

void ensure_device() {
  std::call_once(g_init_flag, [] {
    @autoreleasepool {
      g_device = MTLCreateSystemDefaultDevice();
      if (!g_device)
        throw std::runtime_error("No Metal device found");
      g_queue = [g_device newCommandQueue];
      if (!g_queue)
        throw std::runtime_error("Failed to create command queue");
      g_device_name = std::string([g_device.name UTF8String]);

      std::string best = "unknown";
      for (int i = 1; i <= 20; i++)
        if ([g_device supportsFamily:(MTLGPUFamily)(1000 + i)])
          best = "apple" + std::to_string(i);
      g_gpu_family = best;
    }
  });
}

DeviceInfo device_info() {
  ensure_device();
  DeviceInfo d;
  d.name = g_device_name;
  d.gpu_family = g_gpu_family;
  d.max_threads_per_threadgroup = (int)g_device.maxThreadsPerThreadgroup.width;
  d.max_threadgroup_memory_length = (int)g_device.maxThreadgroupMemoryLength;
  d.recommended_max_working_set_size =
      (int64_t)g_device.recommendedMaxWorkingSetSize;
  int gen = 0;
  if (g_gpu_family.rfind("apple", 0) == 0)
    gen = std::atoi(g_gpu_family.c_str() + 5);
  d.has_bfloat16 = gen >= 9;
  return d;
}

// --- buffers ---------------------------------------------------------------

int64_t buf_alloc(size_t nbytes) {
  ensure_device();
  size_t aligned = ((nbytes + 16383) / 16384 + 1) * 16384; // +1 page guard
  if (aligned == 0)
    aligned = 16384;

  @autoreleasepool {
    id<MTLBuffer> metal_buf =
        [g_device newBufferWithLength:aligned
                              options:MTLResourceStorageModeShared];
    if (!metal_buf)
      throw std::runtime_error("Failed to allocate Metal buffer");
    void *ptr = metal_buf.contents;

    std::lock_guard<std::mutex> lock(g_buf_mutex);
    int64_t h = g_next_buf_handle++;
    g_alloy_buffers[h] = {ptr, nbytes, aligned, metal_buf, h, false};
    g_alloy_ptrs.insert((uintptr_t)ptr);
    g_ptr_to_handle[(uintptr_t)ptr] = h;
    return h;
  }
}

void buf_release(int64_t handle) {
  @autoreleasepool {
    std::lock_guard<std::mutex> lock(g_buf_mutex);
    auto it = g_alloy_buffers.find(handle);
    if (it == g_alloy_buffers.end() || it->second.released)
      return;
    g_alloy_ptrs.erase((uintptr_t)it->second.ptr);
    g_ptr_to_handle.erase((uintptr_t)it->second.ptr);
    {
      std::lock_guard<std::mutex> lock2(g_buffer_mutex);
      g_buffer_cache.erase((uintptr_t)it->second.ptr);
    }
    if (it->second.slice) {
      // View into a pool: the pool owns the pages.
    } else if (it->second.vm_owned) {
      vm_deallocate(mach_task_self(), (vm_address_t)it->second.ptr,
                    it->second.aligned_size);
    } else {
      [it->second.metal_buf setPurgeableState:MTLPurgeableStateEmpty];
    }
    it->second.metal_buf = nil;
    it->second.ptr = nullptr;
    it->second.released = true;
  }
}

uintptr_t buf_ptr(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  auto it = g_alloy_buffers.find(handle);
  if (it == g_alloy_buffers.end())
    throw std::runtime_error("Invalid buffer handle");
  if (it->second.released)
    throw std::runtime_error("Buffer already released");
  return (uintptr_t)it->second.ptr;
}

int64_t buf_handle_for_ptr(uintptr_t ptr) {
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  auto it = g_ptr_to_handle.find(ptr);
  return (it != g_ptr_to_handle.end()) ? it->second : -1;
}

size_t buf_nbytes(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  auto it = g_alloy_buffers.find(handle);
  if (it == g_alloy_buffers.end())
    throw std::runtime_error("Invalid buffer handle");
  return it->second.nbytes;
}

// --- pipelines -------------------------------------------------------------

int64_t compile_msl(const std::string &source,
                    const std::string &function_name) {
  ensure_device();
  int64_t handle;
  @autoreleasepool {
    NSString *src = [NSString stringWithUTF8String:source.c_str()];
    NSError *err = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:src
                                                options:nil
                                                  error:&err];
    if (!lib) {
      std::string msg =
          err ? std::string([err.localizedDescription UTF8String]) : "unknown";
      throw std::runtime_error("MSL compilation failed: " + msg);
    }
    NSString *fname = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> func = [lib newFunctionWithName:fname];
    if (!func)
      throw std::runtime_error("Function '" + function_name + "' not found");
    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&err];
    if (!pso) {
      std::string msg =
          err ? std::string([err.localizedDescription UTF8String]) : "unknown";
      throw std::runtime_error("Pipeline creation failed: " + msg);
    }
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    handle = g_next_handle++;
    g_pipelines[handle] = pso;
    g_libraries[handle] = lib;
    g_pipeline_names[handle] = function_name;
  }
  return handle;
}

int64_t compile_metallib(const std::string &path,
                         const std::string &function_name) {
  ensure_device();
  int64_t handle;
  @autoreleasepool {
    NSString *nspath = [NSString stringWithUTF8String:path.c_str()];
    NSURL *url = [NSURL fileURLWithPath:nspath];
    NSError *err = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithURL:url error:&err];
    if (!lib)
      throw std::runtime_error("Failed to load metallib: " + path);
    NSString *fname = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> func = [lib newFunctionWithName:fname];
    if (!func)
      throw std::runtime_error("Function '" + function_name + "' not found");
    id<MTLComputePipelineState> pso =
        [g_device newComputePipelineStateWithFunction:func error:&err];
    if (!pso)
      throw std::runtime_error("Pipeline creation failed");
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    handle = g_next_handle++;
    g_pipelines[handle] = pso;
    g_libraries[handle] = lib;
    g_pipeline_names[handle] = function_name;
  }
  return handle;
}

// --- plan registration + dispatch ------------------------------------------

NSUInteger plan_buffer_offset(const PlanSlot &slot, const PlanDispatch &dispatch,
                              size_t i) {
  int64_t offset = slot.base_offset + dispatch.buf_offsets[i];
  if (offset < 0)
    throw std::runtime_error("Negative resolved plan buffer offset");
  return (NSUInteger)offset;
}

void update_plan_input_slots(Plan *plan,
                             const std::vector<InputUpdate> &updates) {
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  for (auto &slot : plan->slots) {
    if (slot.type == PlanSlot::INPUT) {
      slot.metal_buf = nil;
      slot.base_offset = 0;
      slot.handle = 0;
      slot.ptr = 0;
    }
  }
  for (const auto &u : updates) {
    int slot_idx = u.slot_idx;
    int64_t buf_handle = u.handle;
    int64_t offset = u.offset;
    if (slot_idx < 0 || (size_t)slot_idx >= plan->slots.size())
      throw std::runtime_error("Invalid input slot index " +
                               std::to_string(slot_idx));
    auto &slot = plan->slots[slot_idx];
    if (slot.type != PlanSlot::INPUT)
      throw std::runtime_error("Input update targeted non-input slot " +
                               std::to_string(slot_idx));
    auto it = g_alloy_buffers.find(buf_handle);
    if (it == g_alloy_buffers.end())
      throw std::runtime_error("Invalid input buffer handle " +
                               std::to_string(buf_handle) + " for input slot " +
                               std::to_string(slot_idx));
    if (it->second.released || it->second.metal_buf == nil)
      throw std::runtime_error("Released input buffer handle " +
                               std::to_string(buf_handle) + " for input slot " +
                               std::to_string(slot_idx));
    if (offset < 0 || (size_t)offset > it->second.aligned_size)
      throw std::runtime_error("Input buffer offset out of range for slot " +
                               std::to_string(slot_idx));
    slot.metal_buf = it->second.metal_buf;
    slot.base_offset = offset + (int64_t)it->second.mtl_offset;
    slot.handle = buf_handle;
    slot.ptr = (uintptr_t)it->second.ptr;
    slot.nbytes = it->second.nbytes;
  }
  for (size_t slot_idx = 0; slot_idx < plan->slots.size(); slot_idx++) {
    auto &slot = plan->slots[slot_idx];
    if (slot.type == PlanSlot::INPUT && !slot.metal_buf)
      throw std::runtime_error("Missing input update for input slot " +
                               std::to_string(slot_idx));
  }
}

void update_plan_dispatch_grids(Plan *plan,
                                const std::vector<GridUpdate> &updates) {
  for (const auto &u : updates) {
    int flat_idx = u.flat_idx;
    if (flat_idx < 0 || (size_t)flat_idx >= plan->flat_to_group.size())
      throw std::runtime_error("Grid-update dispatch index out of range: " +
                               std::to_string(flat_idx));
    const auto &loc = plan->flat_to_group[flat_idx];
    if (loc.first < 0)
      throw std::runtime_error("Grid-update dispatch index not in any group: " +
                               std::to_string(flat_idx));
    plan->groups[loc.first][loc.second].grid =
        MTLSizeMake((NSUInteger)u.gx, (NSUInteger)u.gy, (NSUInteger)u.gz);
  }
}

int64_t register_plan(const std::vector<DispatchSpec> &dispatches,
                      const std::vector<SlotSpec> &slots,
                      const std::vector<std::vector<int>> &groups,
                      const std::vector<int> &written_slot_indices) {
  ensure_device();
  Plan plan;

  plan.slots.reserve(slots.size());
  for (const auto &s : slots) {
    PlanSlot slot;
    slot.type = (PlanSlot::Type)s.type;
    slot.input_arg_idx = s.input_arg_idx;
    int64_t handle_or_ptr = s.handle_or_ptr;
    slot.nbytes = s.nbytes;
    slot.handle = 0;
    slot.ptr = 0;
    slot.metal_buf = nil;
    slot.base_offset = 0;
    if (slot.type != PlanSlot::INPUT && handle_or_ptr > 0) {
      slot.handle = handle_or_ptr;
      std::lock_guard<std::mutex> lock(g_buf_mutex);
      auto it = g_alloy_buffers.find(handle_or_ptr);
      if (it == g_alloy_buffers.end())
        throw std::runtime_error("Invalid plan buffer handle " +
                                 std::to_string(handle_or_ptr));
      if (it->second.released || it->second.metal_buf == nil)
        throw std::runtime_error("Released plan buffer handle " +
                                 std::to_string(handle_or_ptr));
      slot.metal_buf = it->second.metal_buf;
      slot.base_offset = (int64_t)it->second.mtl_offset;
      slot.ptr = (uintptr_t)it->second.ptr;
    }
    plan.slots.push_back(slot);
  }

  std::vector<PlanDispatch> all_dispatches;
  all_dispatches.reserve(dispatches.size());
  for (const auto &item : dispatches) {
    PlanDispatch pd;
    {
      std::lock_guard<std::mutex> lock(g_cache_mutex);
      auto it = g_pipelines.find(item.pso_handle);
      if (it == g_pipelines.end())
        throw std::runtime_error("Invalid pipeline handle");
      pd.pso = it->second;
    }
    pd.buf_indices = item.buf_indices;
    pd.buf_offsets = item.buf_offsets;
    pd.grid = MTLSizeMake(item.grid[0], item.grid[1], item.grid[2]);
    pd.tg = MTLSizeMake(item.tg[0], item.tg[1], item.tg[2]);
    all_dispatches.push_back(std::move(pd));
  }

  plan.flat_to_group.assign(all_dispatches.size(), std::make_pair(-1, -1));
  plan.groups.reserve(groups.size());
  for (size_t gi = 0; gi < groups.size(); gi++) {
    const auto &group_indices = groups[gi];
    std::vector<PlanDispatch> group;
    group.reserve(group_indices.size());
    for (size_t j = 0; j < group_indices.size(); j++) {
      int di = group_indices[j];
      group.push_back(all_dispatches[di]);
      if (di >= 0 && (size_t)di < plan.flat_to_group.size())
        plan.flat_to_group[di] = std::make_pair((int)gi, (int)j);
    }
    plan.groups.push_back(std::move(group));
  }

  for (int si : written_slot_indices)
    plan.written_slots.insert(si);

  std::lock_guard<std::mutex> lock(g_plan_mutex);
  int64_t handle = g_next_plan_handle++;
  g_plans[handle] = std::move(plan);
  return handle;
}

void release_plan(int64_t plan_handle) {
  std::lock_guard<std::mutex> lock(g_plan_mutex);
  g_plans.erase(plan_handle); // destroys the Plan → drops its slots' MTLBuffer refs
}

static std::vector<PreCopy>
resolve_pre_copies(const std::vector<PreCopySpec> &specs) {
  std::vector<PreCopy> out;
  if (specs.empty())
    return out;
  out.reserve(specs.size());
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  auto resolve = [](int64_t handle) -> id<MTLBuffer> {
    auto it = g_alloy_buffers.find(handle);
    if (it == g_alloy_buffers.end() || it->second.released ||
        it->second.metal_buf == nil)
      throw std::runtime_error("Invalid pre_copy buffer handle " +
                               std::to_string(handle));
    return it->second.metal_buf;
  };
  for (const auto &s : specs) {
    PreCopy pc;
    pc.dst_buf = resolve(s.dst_handle);
    pc.dst_offset = s.dst_offset;
    pc.src_buf = resolve(s.src_handle);
    pc.src_offset = s.src_offset;
    pc.nbytes = s.nbytes;
    out.push_back(pc);
  }
  return out;
}

// g_pending_cb lifecycle is portable across ARC (app build) and MRC (the Python
// extension's historical build flags): under ARC the strong global auto-manages.
#if __has_feature(objc_arc)
#define AE_CB_RETAIN(cb) (cb)
#define AE_CB_RELEASE(p) ((p) = nil)
#else
#define AE_CB_RETAIN(cb) ([(cb) retain])
#define AE_CB_RELEASE(p)                                                        \
  do {                                                                          \
    [(p) release];                                                             \
    (p) = nil;                                                                  \
  } while (0)
#endif

void gpu_sync() {
  if (g_pending_cb != nil) {
    [g_pending_cb waitUntilCompleted];
    if (g_pending_cb.status == MTLCommandBufferStatusError) {
      NSError *err = g_pending_cb.error;
      std::string msg =
          err ? std::string([err.localizedDescription UTF8String]) : "unknown";
      AE_CB_RELEASE(g_pending_cb);
      throw std::runtime_error("GPU error (async): " + msg);
    }
    AE_CB_RELEASE(g_pending_cb);
  }
}

PlanTiming dispatch_plan(int64_t plan_handle,
                         const std::vector<InputUpdate> &input_arrays,
                         bool serialized, bool defer_wait,
                         const std::vector<PreCopySpec> &pre_copies,
                         const std::vector<GridUpdate> &grid_updates) {
  ensure_device();

  Plan *plan;
  {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto it = g_plans.find(plan_handle);
    if (it == g_plans.end())
      throw std::runtime_error("Invalid plan handle");
    plan = &it->second;
  }

  std::vector<PreCopy> pre_copy_ops = resolve_pre_copies(pre_copies);
  PlanTiming timing{0.0, 0.0, 0.0, 0};

  @autoreleasepool {
    auto t0 = std::chrono::high_resolution_clock::now();

    update_plan_input_slots(plan, input_arrays);
    if (!grid_updates.empty())
      update_plan_dispatch_grids(plan, grid_updates);

    double gpu_ms = 0.0;
    auto t_encode = std::chrono::high_resolution_clock::now();
    auto t_wait = t_encode;

    if (serialized) {
      for (auto &group : plan->groups) {
        for (auto &pd : group) {
          MTLCommandBufferDescriptor *desc =
              [[MTLCommandBufferDescriptor alloc] init];
          desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
          id<MTLCommandBuffer> cb = [g_queue commandBufferWithDescriptor:desc];
          if (!cb)
            throw std::runtime_error("Failed to create command buffer");
          id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
          [enc setComputePipelineState:pd.pso];
          for (size_t i = 0; i < pd.buf_indices.size(); i++) {
            auto &slot = plan->slots[pd.buf_indices[i]];
            [enc setBuffer:slot.metal_buf
                    offset:plan_buffer_offset(slot, pd, i)
                   atIndex:i];
          }
          [enc dispatchThreadgroups:pd.grid threadsPerThreadgroup:pd.tg];
          [enc endEncoding];
          [cb commit];
          [cb waitUntilCompleted];
          if (cb.status == MTLCommandBufferStatusError) {
            NSError *err = cb.error;
            std::string msg =
                err ? std::string([err.localizedDescription UTF8String])
                    : "unknown";
            throw std::runtime_error("GPU error (serialized): " + msg);
          }
          gpu_ms += (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        }
      }
      t_encode = std::chrono::high_resolution_clock::now();
      t_wait = t_encode;
    } else {
      id<MTLCommandBuffer> cb;
      {
        MTLCommandBufferDescriptor *desc =
            [[MTLCommandBufferDescriptor alloc] init];
        desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        cb = [g_queue commandBufferWithDescriptor:desc];
      }
      if (!cb)
        throw std::runtime_error("Failed to create command buffer");

      if (!pre_copy_ops.empty()) {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        for (const auto &pc : pre_copy_ops) {
          [blit copyFromBuffer:pc.src_buf
                  sourceOffset:(NSUInteger)pc.src_offset
                      toBuffer:pc.dst_buf
             destinationOffset:(NSUInteger)pc.dst_offset
                          size:(NSUInteger)pc.nbytes];
        }
        [blit endEncoding];
      }

      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (size_t gi = 0; gi < plan->groups.size(); gi++) {
        if (gi > 0)
          [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        for (auto &pd : plan->groups[gi]) {
          [enc setComputePipelineState:pd.pso];
          for (size_t i = 0; i < pd.buf_indices.size(); i++) {
            auto &slot = plan->slots[pd.buf_indices[i]];
            [enc setBuffer:slot.metal_buf
                    offset:plan_buffer_offset(slot, pd, i)
                   atIndex:i];
          }
          [enc dispatchThreadgroups:pd.grid threadsPerThreadgroup:pd.tg];
        }
      }
      [enc endEncoding];
      t_encode = std::chrono::high_resolution_clock::now();
      [cb commit];

      if (defer_wait) {
        g_pending_cb = AE_CB_RETAIN(cb);
        t_wait = t_encode;
        gpu_ms = 0.0;
      } else {
        [cb waitUntilCompleted];
        t_wait = std::chrono::high_resolution_clock::now();
        if (cb.status == MTLCommandBufferStatusError) {
          NSError *err = cb.error;
          std::string msg =
              err ? std::string([err.localizedDescription UTF8String])
                  : "unknown";
          throw std::runtime_error("GPU error: " + msg);
        }
        gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
      }
    }

    auto to_ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    timing.gpu_ms = gpu_ms;
    timing.encode_ms = to_ms(t0, t_encode);
    timing.wait_ms = to_ms(t_encode, t_wait);
    timing.n_groups = (int)plan->groups.size();
  }

  return timing;
}

} // namespace alloycore
