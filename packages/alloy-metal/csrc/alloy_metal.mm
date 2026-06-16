// Alloy Metal runtime — nanobind C++/ObjC++ extension.
// All Metal state lives in C++ statics. Python passes numpy arrays,
// gets results via shared memory.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
// No numpy C API — use CPython + nanobind for array access
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <algorithm>
#include <atomic>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>

namespace nb = nanobind;

// The device, buffers, pipelines, and plan registry live in the shared,
// nanobind-free core (alloy_core), linked by the on-device Swift engine too.
// This translation unit keeps only the nb shims + Python-only paths (handler
// dispatch, paged pool, profiling).

#include "alloy_core.h"

using namespace alloycore;

// Training mode: when true, dispatch_plan_profiled clears the get_buffer cache
// on each call so optimizer.step() mutations on external storage are observed.
// Production dispatch_plan resolves through alloy buffer handles and is
// unaffected — it never reads g_training_mode.
static bool g_training_mode = false;

// buf_alloc / buf_release now live in alloy_core (shared with the Swift engine).

// Live Metal-buffer accounting (debug/memory audit): count + total aligned
// bytes of non-released alloy buffers, so leaked plan pools / scratch show up
// without vmmap region guesswork.
static int64_t region_resident_bytes(uintptr_t addr, size_t len) {
  // Sum resident pages over [addr, addr+len) via the VM region map — one
  // recurse call per region, so cheap even for multi-GB demand-paged buffers.
  int64_t resident = 0;
  mach_vm_address_t a = addr;
  mach_vm_address_t end = addr + len;
  while (a < end) {
    mach_vm_address_t r_addr = a;
    mach_vm_size_t r_size = 0;
    vm_region_submap_info_data_64_t info;
    mach_msg_type_number_t cnt = VM_REGION_SUBMAP_INFO_COUNT_64;
    natural_t depth = 0;
    kern_return_t kr = mach_vm_region_recurse(
        mach_task_self(), &r_addr, &r_size, &depth,
        (vm_region_recurse_info_t)&info, &cnt);
    if (kr != KERN_SUCCESS || r_addr >= end)
      break;
    mach_vm_address_t lo = r_addr > addr ? r_addr : addr;
    mach_vm_address_t hi = (r_addr + r_size) < end ? (r_addr + r_size) : end;
    if (hi > lo) {
      double frac = (double)(hi - lo) / (double)r_size;
      resident += (int64_t)((double)info.pages_resident * 16384.0 * frac);
    }
    a = r_addr + r_size;
  }
  return resident;
}

static nb::dict buffer_stats() {
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  int64_t n = 0, reserved = 0, resident = 0;
  for (auto &[h, ab] : g_alloy_buffers) {
    if (ab.released || ab.slice)
      continue; // skip pool slices: their pages are counted under the pool
    n++;
    reserved += (int64_t)ab.aligned_size;
    resident += region_resident_bytes((uintptr_t)ab.ptr, ab.aligned_size);
  }
  nb::dict d;
  d["count"] = n;
  d["bytes"] = reserved;
  d["resident"] = resident;
  return d;
}

// Per-buffer (handle, reserved, resident) for the top-N by resident bytes —
// memory-audit dump to see what the live Metal buffers actually are.
static nb::list buffer_dump(int64_t top) {
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  std::vector<std::tuple<int64_t, int64_t, int64_t>> rows;
  for (auto &[h, ab] : g_alloy_buffers) {
    if (ab.released)
      continue;
    int64_t res = region_resident_bytes((uintptr_t)ab.ptr, ab.aligned_size);
    rows.emplace_back(h, (int64_t)ab.aligned_size, res);
  }
  std::sort(rows.begin(), rows.end(),
            [](auto &a, auto &b) { return std::get<2>(a) > std::get<2>(b); });
  nb::list out;
  for (size_t i = 0; i < rows.size() && (int64_t)i < top; i++) {
    nb::dict r;
    r["handle"] = std::get<0>(rows[i]);
    r["reserved"] = std::get<1>(rows[i]);
    r["resident"] = std::get<2>(rows[i]);
    out.append(r);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Paged KV pool — one mach_vm reservation wrapped in a single bytesNoCopy
// MTLBuffer; slices register interior ranges as first-class alloy buffers so
// every pointer/handle path (handler dispatch, compiled-plan binding,
// buf_handle_for_ptr) resolves them with the right Metal offset. Pages commit
// on first touch and return to the kernel via pool_reclaim (E0/E1-proven).
// ---------------------------------------------------------------------------

// Memory-pressure level, polled from Python between requests (reclaim must
// only run while the GPU is quiescent — the serve loop is serialized, so a
// flag + between-request sweep keeps that invariant without a C++-side race
// against the allocator). 0 = normal, 1 = warn, 2 = critical.
static std::atomic<int> g_memory_pressure{0};
static dispatch_source_t g_pressure_source = nil;

static void ensure_pressure_source() {
  if (g_pressure_source)
    return;
  g_pressure_source = dispatch_source_create(
      DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
      DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
          DISPATCH_MEMORYPRESSURE_CRITICAL,
      dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
  dispatch_source_set_event_handler(g_pressure_source, ^{
    unsigned long flags = dispatch_source_get_data(g_pressure_source);
    if (flags & DISPATCH_MEMORYPRESSURE_CRITICAL)
      g_memory_pressure.store(2);
    else if (flags & DISPATCH_MEMORYPRESSURE_WARN)
      g_memory_pressure.store(1);
    else
      g_memory_pressure.store(0);
  });
  dispatch_resume(g_pressure_source);
}

static int memory_pressure_level() { return g_memory_pressure.load(); }

static int64_t pool_max_bytes() {
  ensure_device();
  return (int64_t)g_device.maxBufferLength;
}

// Reserve virtual address space only — NO Metal buffer over the pool. Metal
// wires a buffer's FULL residency at first encoder use (the one-shot
// intermediate pool learned this the hard way: see
// update_plan_intermediate_slots), so a single working-set-sized pool buffer
// would force-commit the entire reservation the first time any slice is
// bound. Each pool_slice instead wraps its own page range in its own
// bytesNoCopy MTLBuffer — wiring stays per-tensor, exactly like buf_alloc.
static int64_t pool_create(size_t nbytes) {
  ensure_device();
  ensure_pressure_source();
  size_t aligned = ((nbytes + 16383) / 16384 + 1) * 16384; // +1 page guard
  mach_vm_address_t addr = 0;
  kern_return_t kr =
      mach_vm_allocate(mach_task_self(), &addr, aligned, VM_FLAGS_ANYWHERE);
  if (kr != KERN_SUCCESS)
    throw std::runtime_error("pool_create: mach_vm_allocate failed: " +
                             std::to_string(kr));
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  int64_t h = g_next_buf_handle++;
  g_alloy_buffers[h] = {(void *)addr, nbytes, aligned, nil, h,
                        false,        0,      false,   true};
  // No ptr-map registration: the pool itself is never a tensor storage and
  // must never resolve through get_buffer (it has no Metal backing).
  return h;
}

// Register [offset, offset+nbytes) of a pool as a standalone alloy buffer
// with its own bytesNoCopy MTLBuffer. Offset must be page-aligned (torch
// storage bases; pool_reclaim works on whole pages). The Metal buffer length
// is padded by one page for cooperative-load overshoot, like buf_alloc — the
// pool's trailing guard page covers the last slice.
static int64_t pool_slice(int64_t pool_handle, size_t offset, size_t nbytes) {
  ensure_device();
  @autoreleasepool {
    std::lock_guard<std::mutex> lock(g_buf_mutex);
    auto it = g_alloy_buffers.find(pool_handle);
    if (it == g_alloy_buffers.end() || it->second.released ||
        !it->second.vm_owned)
      throw std::runtime_error("pool_slice: invalid pool handle");
    if (offset % 16384 != 0)
      throw std::runtime_error("pool_slice: offset must be page-aligned");
    if (offset + nbytes > it->second.nbytes)
      throw std::runtime_error("pool_slice: range exceeds pool");
    size_t aligned = ((nbytes + 16383) / 16384 + 1) * 16384; // +1 page guard
    aligned = std::min(aligned, it->second.aligned_size - offset);
    if (aligned > g_device.maxBufferLength)
      throw std::runtime_error("pool_slice: slice exceeds maxBufferLength");
    void *ptr = (char *)it->second.ptr + offset;
    id<MTLBuffer> metal_buf =
        [g_device newBufferWithBytesNoCopy:ptr
                                    length:aligned
                                   options:MTLResourceStorageModeShared
                               deallocator:nil];
    if (!metal_buf)
      throw std::runtime_error("pool_slice: newBufferWithBytesNoCopy failed");
    int64_t h = g_next_buf_handle++;
    g_alloy_buffers[h] = {ptr,   nbytes, aligned, metal_buf, h,
                          false, 0,      true,    false};
    g_alloy_ptrs.insert((uintptr_t)ptr);
    g_ptr_to_handle[(uintptr_t)ptr] = h;
    return h;
  }
}

// Return committed pages in [offset, offset+len) of `handle` to the kernel
// (footprint accounting drops now; residency under later memory pressure).
// Page-granular: the range is shrunk inward to whole pages. Returns the
// number of bytes madvised. Caller guarantees the GPU is quiescent on the
// range and never assumes the content survives OR zeros (E1: stale bytes can
// persist) — for KV both are fine, rows past cumulative_length are dead.
static int64_t pool_reclaim(int64_t handle, size_t offset, size_t len) {
  void *base;
  {
    std::lock_guard<std::mutex> lock(g_buf_mutex);
    auto it = g_alloy_buffers.find(handle);
    if (it == g_alloy_buffers.end() || it->second.released)
      throw std::runtime_error("pool_reclaim: invalid handle");
    if (offset + len > it->second.nbytes)
      throw std::runtime_error("pool_reclaim: range exceeds buffer");
    base = (char *)it->second.ptr + offset;
  }
  uintptr_t start = ((uintptr_t)base + 16383) & ~(uintptr_t)16383;
  uintptr_t end = ((uintptr_t)base + len) & ~(uintptr_t)16383;
  if (end <= start)
    return 0;
  if (madvise((void *)start, end - start, MADV_FREE_REUSABLE) != 0)
    return 0; // advisory: a failed reclaim is a no-op, never an error
  return (int64_t)(end - start);
}

// buf_handle_for_ptr / buf_ptr now live in alloy_core.

static nb::ndarray<nb::numpy> buf_numpy(int64_t handle, nb::tuple shape,
                                        const std::string &dtype_str) {
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  auto it = g_alloy_buffers.find(handle);
  if (it == g_alloy_buffers.end())
    throw std::runtime_error("Invalid buffer handle");
  if (it->second.released)
    throw std::runtime_error("Buffer already released");

  void *ptr = it->second.ptr;
  size_t ndim = nb::len(shape);
  std::vector<size_t> shape_vec(ndim);
  for (size_t i = 0; i < ndim; i++)
    shape_vec[i] = nb::cast<size_t>(shape[i]);

  nb::dlpack::dtype dt;
  if (dtype_str == "f32" || dtype_str == "float32") {
    dt = {(uint8_t)nb::dlpack::dtype_code::Float, 32, 1};
  } else if (dtype_str == "f16" || dtype_str == "float16") {
    dt = {(uint8_t)nb::dlpack::dtype_code::Float, 16, 1};
  } else if (dtype_str == "i32" || dtype_str == "int32") {
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 32, 1};
  } else if (dtype_str == "i16" || dtype_str == "int16") {
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 16, 1};
  } else if (dtype_str == "u8" || dtype_str == "uint8") {
    dt = {(uint8_t)nb::dlpack::dtype_code::UInt, 8, 1};
  } else if (dtype_str == "i8" || dtype_str == "int8") {
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 8, 1};
  } else if (dtype_str == "u32" || dtype_str == "uint32") {
    dt = {(uint8_t)nb::dlpack::dtype_code::UInt, 32, 1};
  } else if (dtype_str == "i64" || dtype_str == "int64") {
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 64, 1};
  } else if (dtype_str == "u16" || dtype_str == "uint16") {
    dt = {(uint8_t)nb::dlpack::dtype_code::UInt, 16, 1};
  } else if (dtype_str == "u64" || dtype_str == "uint64") {
    dt = {(uint8_t)nb::dlpack::dtype_code::UInt, 64, 1};
  } else {
    throw std::runtime_error("Unsupported dtype: " + dtype_str);
  }

  // No-op owner: lifetime is managed by the Alloy buffer handle.
  nb::capsule owner(ptr, [](void *) noexcept {});
  return nb::ndarray<nb::numpy>(ptr, ndim, shape_vec.data(), owner, nullptr,
                                dt);
}

// buf_nbytes / ensure_device now live in alloy_core.

// Pre-wire each buffer's full VA resident via a trivial GPU dispatch (lazy
// first-encoder-use), with NO MTLResidencySet / requestResidency. Metal wires
// a bytesNoCopy buffer's whole VA on first encoder use (~14ms/GB); doing it
// here moves that one-time cost off the request path for paged free slices.
// Unlike requestResidency, a dispatch-wire does NOT count against
// phys_footprint (E1: GPU-faulted bytesNoCopy pages are invisible to it), so
// the slices stay demand-paged for memory. Reads one element per buffer into a
// shared scratch — the target buffers are never modified.
static id<MTLComputePipelineState> g_wire_pso = nil;

static void wire_buffers(nb::list handles) {
  ensure_device();
  @autoreleasepool {
    if (!g_wire_pso) {
      const char *src =
          "#include <metal_stdlib>\nusing namespace metal;\n"
          "kernel void wire_touch(device const uint* b [[buffer(0)]],\n"
          "  device uint* o [[buffer(1)]], uint t [[thread_position_in_grid]])\n"
          "{ o[0] = b[0]; }";
      NSError *err = nil;
      id<MTLLibrary> lib =
          [g_device newLibraryWithSource:@(src) options:nil error:&err];
      if (!lib)
        throw std::runtime_error("wire_buffers: library compile failed");
      g_wire_pso = [g_device
          newComputePipelineStateWithFunction:[lib newFunctionWithName:@"wire_touch"]
                                        error:&err];
      if (!g_wire_pso)
        throw std::runtime_error("wire_buffers: pipeline failed");
    }
    // Resolve handles → (buffer, offset) under the lock, then dispatch without
    // it (the wait must not block allocations).
    std::vector<std::pair<id<MTLBuffer>, NSUInteger>> targets;
    {
      std::lock_guard<std::mutex> lock(g_buf_mutex);
      for (size_t i = 0; i < handles.size(); i++) {
        int64_t h = nb::cast<int64_t>(handles[i]);
        auto it = g_alloy_buffers.find(h);
        if (it == g_alloy_buffers.end() || it->second.released ||
            !it->second.metal_buf)
          continue;
        targets.emplace_back(it->second.metal_buf,
                             (NSUInteger)it->second.mtl_offset);
      }
    }
    if (targets.empty())
      return;
    id<MTLBuffer> scratch =
        [g_device newBufferWithLength:16 options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:g_wire_pso];
    for (auto &[buf, offset] : targets) {
      [enc setBuffer:buf offset:offset atIndex:0];
      [enc setBuffer:scratch offset:0 atIndex:1];
      [enc dispatchThreads:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }
}

// ---------------------------------------------------------------------------
// Python API — functions only, no classes with Metal pointers
// ---------------------------------------------------------------------------

// nb shim over alloycore::device_info. The working-set budget is the figure to
// size KV-cache fill against on Apple Silicon's unified memory.
static nb::dict py_device_info() {
  alloycore::DeviceInfo i = alloycore::device_info();
  nb::dict d;
  d["name"] = i.name;
  d["gpu_family"] = i.gpu_family;
  d["max_threads_per_threadgroup"] = i.max_threads_per_threadgroup;
  d["max_threadgroup_memory_length"] = i.max_threadgroup_memory_length;
  d["recommended_max_working_set_size"] = i.recommended_max_working_set_size;
  d["has_bfloat16"] = i.has_bfloat16;
  return d;
}

// compile_msl / compile_metallib now live in alloy_core.

static int pipeline_max_threads(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  auto it = g_pipelines.find(handle);
  if (it == g_pipelines.end())
    throw std::runtime_error("Invalid pipeline handle");
  return (int)it->second.maxTotalThreadsPerThreadgroup;
}

static int pipeline_thread_width(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  auto it = g_pipelines.find(handle);
  if (it == g_pipelines.end())
    throw std::runtime_error("Invalid pipeline handle");
  return (int)it->second.threadExecutionWidth;
}

static int pipeline_static_threadgroup_memory(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_cache_mutex);
  auto it = g_pipelines.find(handle);
  if (it == g_pipelines.end())
    throw std::runtime_error("Invalid pipeline handle");
  return (int)it->second.staticThreadgroupMemoryLength;
}

// GPU-trace capture for `alloy profile --capture`. Captures g_device — the exact
// device alloy dispatches on — into an Xcode .gputrace document, returning "" on
// success or an error string. MTL_CAPTURE_ENABLED=1 must be set in the env before
// the device is created (the CLI re-execs to guarantee it).
static std::string py_capture_start(const std::string &out_path) {
  ensure_device();
  MTLCaptureManager *mgr = [MTLCaptureManager sharedCaptureManager];
  if (![mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument])
    return "this device/OS can't write GPU trace documents";
  MTLCaptureDescriptor *desc = [[MTLCaptureDescriptor alloc] init];
  desc.captureObject = g_device;
  desc.destination = MTLCaptureDestinationGPUTraceDocument;
  desc.outputURL = [NSURL fileURLWithPath:@(out_path.c_str())];
  NSError *err = nil;
  if (![mgr startCaptureWithDescriptor:desc error:&err]) {
    const char *m = err ? [[err localizedDescription] UTF8String] : nullptr;
    return std::string("startCapture failed: ") + (m ? m : "unknown");
  }
  return "";
}

static void py_capture_stop() {
  [[MTLCaptureManager sharedCaptureManager] stopCapture];
}

static size_t rounded_buffer_length(size_t nbytes) {
  size_t aligned = ((nbytes + 16383) / 16384) * 16384;
  return aligned == 0 ? 16384 : aligned;
}

// Resolve a raw pointer to a Metal buffer for handler/profiled dispatch.
// Alloy-owned pointers reuse their Metal backing; external pointers are
// copied into a cached shared buffer.
static std::pair<id<MTLBuffer>, size_t> get_buffer(void *ptr, size_t nbytes) {
  uintptr_t addr = (uintptr_t)ptr;
  uintptr_t page_base = addr & ~(uintptr_t)0x3FFF; // round down to 16KB
  size_t offset = addr - page_base;
  size_t total = rounded_buffer_length(offset + nbytes);

  // Our own pointers (buf_alloc bases and pool-slice bases, tracked in
  // g_ptr_to_handle) resolve directly to their Metal backing; a pool slice
  // carries its byte offset within the pool buffer.
  {
    std::lock_guard<std::mutex> lock(g_buf_mutex);
    auto ph = g_ptr_to_handle.find(addr);
    if (ph != g_ptr_to_handle.end()) {
      auto &ab = g_alloy_buffers[ph->second];
      if (!ab.released)
        return {ab.metal_buf, ab.mtl_offset};
    }
  }

  {
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    auto it = g_buffer_cache.find(addr);
    if (it != g_buffer_cache.end() && it->second.length >= nbytes) {
      return {it->second, 0};
    }
  }

  // External pointers are copied once into the cache; callers clear
  // g_buffer_cache when the source storage mutates.
  size_t copy_len = rounded_buffer_length(nbytes);
  id<MTLBuffer> buf =
      [g_device newBufferWithLength:copy_len
                            options:MTLResourceStorageModeShared];
  if (!buf)
    throw std::runtime_error("Failed to create Metal buffer");
  memcpy(buf.contents, ptr, nbytes);
  // Zero padding beyond data — kernels may overshoot via cooperative loads
  if (copy_len > nbytes)
    memset((char *)buf.contents + nbytes, 0, copy_len - nbytes);
  {
    std::lock_guard<std::mutex> lock(g_buffer_mutex);
    g_buffer_cache[addr] = buf;
  }
  return {buf, 0};
}

// Dispatch grouped kernels. Barriers separate dependency groups; dispatches
// inside a group are assumed independent.
static nb::dict dispatch(nb::list groups) {
  ensure_device();
  nb::dict result;

  size_t total = 0;
  for (size_t gi = 0; gi < groups.size(); gi++)
    total += nb::len(nb::cast<nb::list>(groups[gi]));
  if (total == 0) {
    result["gpu"] = 0.0;
    result["encode"] = 0.0;
    result["wait"] = 0.0;
    result["copy"] = 0.0;
    return result;
  }

  @autoreleasepool {

    auto t0 = std::chrono::high_resolution_clock::now();

    id<MTLCommandBuffer> cb;
    {
      MTLCommandBufferDescriptor *desc =
          [[MTLCommandBufferDescriptor alloc] init];
      desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
      cb = [g_queue commandBufferWithDescriptor:desc];
    }
    if (!cb)
      throw std::runtime_error("Failed to create command buffer");

    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    for (size_t gi = 0; gi < groups.size(); gi++) {
      if (gi > 0)
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      nb::list group = nb::cast<nb::list>(groups[gi]);
      for (size_t d = 0; d < group.size(); d++) {
        nb::tuple item = nb::cast<nb::tuple>(group[d]);
        int64_t pso_handle = nb::cast<int64_t>(item[0]);
        nb::list buf_entries = nb::cast<nb::list>(item[1]);
        auto grid = nb::cast<std::tuple<int, int, int>>(item[2]);
        auto tg = nb::cast<std::tuple<int, int, int>>(item[3]);

        id<MTLComputePipelineState> pso;
        {
          std::lock_guard<std::mutex> lock(g_cache_mutex);
          auto it = g_pipelines.find(pso_handle);
          if (it == g_pipelines.end())
            throw std::runtime_error("Invalid pipeline handle");
          pso = it->second;
        }

        [enc setComputePipelineState:pso];
        for (size_t i = 0; i < buf_entries.size(); i++) {
          nb::tuple entry = nb::cast<nb::tuple>(buf_entries[i]);
          uintptr_t ptr = (uintptr_t)nb::cast<int64_t>(entry[0]);
          size_t nbytes = nb::cast<size_t>(entry[1]);
          size_t view_offset =
              entry.size() > 2 ? nb::cast<size_t>(entry[2]) : 0;
          auto [buf, offset] = get_buffer((void *)ptr, nbytes);
          [enc setBuffer:buf offset:offset + view_offset atIndex:i];
        }

        [enc dispatchThreadgroups:MTLSizeMake(std::get<0>(grid),
                                              std::get<1>(grid),
                                              std::get<2>(grid))
            threadsPerThreadgroup:MTLSizeMake(std::get<0>(tg), std::get<1>(tg),
                                              std::get<2>(tg))];
      }
    }
    [enc endEncoding];

    auto t_encode = std::chrono::high_resolution_clock::now();

    [cb commit];
    [cb waitUntilCompleted];

    auto t_wait = std::chrono::high_resolution_clock::now();

    if (cb.status == MTLCommandBufferStatusError) {
      NSError *err = cb.error;
      std::string msg =
          err ? std::string([err.localizedDescription UTF8String]) : "unknown";
      throw std::runtime_error("GPU error: " + msg);
    }
    double gpu_ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;

    auto to_ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    result["gpu"] = gpu_ms;
    result["encode"] = to_ms(t0, t_encode);
    result["wait"] = to_ms(t_encode, t_wait);
    result["copy"] = 0.0;
    result["n_groups"] = (int)groups.size();
    result["n_dispatches"] = (int)total;

  } // @autoreleasepool

  return result;
}

// Allocate a typed Metal-owned buffer and return a typed numpy view with its
// handle and pointer.
static nb::tuple alloc_typed(size_t nbytes, nb::tuple shape,
                             const std::string &dtype_str) {
  int64_t handle = buf_alloc(nbytes);
  auto &buf = g_alloy_buffers[handle];
  void *ptr = buf.ptr;
  size_t ndim = nb::len(shape);
  std::vector<size_t> shape_vec(ndim);
  for (size_t i = 0; i < ndim; i++)
    shape_vec[i] = nb::cast<size_t>(shape[i]);

  nb::dlpack::dtype dt;
  if (dtype_str == "f32")
    dt = {(uint8_t)nb::dlpack::dtype_code::Float, 32, 1};
  else if (dtype_str == "f16")
    dt = {(uint8_t)nb::dlpack::dtype_code::Float, 16, 1};
  else if (dtype_str == "i32")
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 32, 1};
  else if (dtype_str == "i64")
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 64, 1};
  else if (dtype_str == "i16")
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 16, 1};
  else if (dtype_str == "i8")
    dt = {(uint8_t)nb::dlpack::dtype_code::Int, 8, 1};
  else if (dtype_str == "u8")
    dt = {(uint8_t)nb::dlpack::dtype_code::UInt, 8, 1};
  else if (dtype_str == "u32")
    dt = {(uint8_t)nb::dlpack::dtype_code::UInt, 32, 1};
  else
    dt = {(uint8_t)nb::dlpack::dtype_code::Float, 32, 1};

  nb::capsule owner(ptr, [](void *) noexcept {});
  auto arr =
      nb::ndarray<nb::numpy>(ptr, ndim, shape_vec.data(), owner, nullptr, dt);
  return nb::make_tuple(arr, handle, (uintptr_t)ptr);
}

// Clear only the Metal buffer cache (pointer → MTLBuffer mapping).
// Called between model compilations to prevent stale buffer bindings
// when torch recycles memory addresses from freed models.
static void clear_buffer_cache() {
  std::lock_guard<std::mutex> lock(g_buffer_mutex);
  g_buffer_cache.clear();
}

// ---------------------------------------------------------------------------
// Graph compiler: pre-planned dispatch
// ---------------------------------------------------------------------------

// PlanDispatch / PlanSlot / Plan, g_plans, plan_buffer_offset, and
// update_plan_input_slots now live in alloy_core.

// nb → struct marshaling for the shims + the Python-only profiled paths.
static std::vector<alloycore::InputUpdate> parse_input_updates(nb::list l) {
  std::vector<alloycore::InputUpdate> out;
  out.reserve(l.size());
  for (size_t i = 0; i < l.size(); i++) {
    nb::tuple e = nb::cast<nb::tuple>(l[i]);
    out.push_back({nb::cast<int>(e[0]), nb::cast<int64_t>(e[1]),
                   nb::cast<int64_t>(e[2])});
  }
  return out;
}

static std::vector<alloycore::GridUpdate> parse_grid_updates(nb::list l) {
  std::vector<alloycore::GridUpdate> out;
  out.reserve(l.size());
  for (size_t i = 0; i < l.size(); i++) {
    nb::tuple e = nb::cast<nb::tuple>(l[i]);
    out.push_back({nb::cast<int>(e[0]), nb::cast<int>(e[1]), nb::cast<int>(e[2]),
                   nb::cast<int>(e[3])});
  }
  return out;
}

// Rebind INTERMEDIATE slots of a registered plan to new alloy buffers. Each
// entry is (slot_idx, handle, nbytes). Used by the one-shot request-bounded
// intermediate pool: M-outer pool buffers are allocated at a high-water prompt
// bound instead of M_MAX (Metal wires FULL buffer residency at first encoder
// use, so a native-M_MAX pool wires ~its whole VA regardless of the shrunk
// grids); when a longer prompt arrives, the Python side reallocates the
// affected pool buffers at the new bound and rebinds them here. Rare path
// (monotone high-water growth), never on the steady-state dispatch.
static void update_plan_intermediate_slots(int64_t plan_handle,
                                           nb::list slot_updates) {
  Plan *plan;
  {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto it = g_plans.find(plan_handle);
    if (it == g_plans.end())
      throw std::runtime_error("Invalid plan handle");
    plan = &it->second;
  }
  std::lock_guard<std::mutex> lock(g_buf_mutex);
  for (size_t i = 0; i < slot_updates.size(); i++) {
    nb::tuple entry = nb::cast<nb::tuple>(slot_updates[i]);
    int slot_idx = nb::cast<int>(entry[0]);
    int64_t buf_handle = nb::cast<int64_t>(entry[1]);
    size_t nbytes = nb::cast<size_t>(entry[2]);
    if (slot_idx < 0 || (size_t)slot_idx >= plan->slots.size()) {
      throw std::runtime_error("Invalid intermediate slot index " +
                               std::to_string(slot_idx));
    }
    auto &slot = plan->slots[slot_idx];
    if (slot.type != PlanSlot::INTERMEDIATE) {
      throw std::runtime_error(
          "Intermediate update targeted non-intermediate slot " +
          std::to_string(slot_idx));
    }
    auto buf_it = g_alloy_buffers.find(buf_handle);
    if (buf_it == g_alloy_buffers.end()) {
      throw std::runtime_error("Invalid intermediate buffer handle " +
                               std::to_string(buf_handle) + " for slot " +
                               std::to_string(slot_idx));
    }
    if (buf_it->second.released || buf_it->second.metal_buf == nil) {
      throw std::runtime_error("Released intermediate buffer handle " +
                               std::to_string(buf_handle) + " for slot " +
                               std::to_string(slot_idx));
    }
    slot.metal_buf = buf_it->second.metal_buf;
    slot.base_offset = (int64_t)buf_it->second.mtl_offset;
    slot.handle = buf_handle;
    slot.ptr = (uintptr_t)buf_it->second.ptr;
    slot.nbytes = nbytes;
  }
}

// Register a compiled plan. Returns an opaque plan handle.
// dispatches: list of (pso_handle, [buf_slot_indices], [buf_offsets], grid_3d, tg_3d)
// slots: list of (type, input_arg_idx, handle_or_ptr, nbytes)
// groups: list of list of dispatch indices
static int64_t py_register_plan(nb::list dispatches, nb::list slots,
                             nb::list groups,
                             nb::list written_slot_indices = nb::list()) {
  std::vector<alloycore::SlotSpec> slot_specs;
  slot_specs.reserve(slots.size());
  for (size_t i = 0; i < slots.size(); i++) {
    nb::tuple s = nb::cast<nb::tuple>(slots[i]);
    slot_specs.push_back({nb::cast<int>(s[0]), nb::cast<int>(s[1]),
                          nb::cast<int64_t>(s[2]), nb::cast<size_t>(s[3])});
  }
  std::vector<alloycore::DispatchSpec> disp_specs;
  disp_specs.reserve(dispatches.size());
  for (size_t d = 0; d < dispatches.size(); d++) {
    nb::tuple item = nb::cast<nb::tuple>(dispatches[d]);
    alloycore::DispatchSpec ds;
    ds.pso_handle = nb::cast<int64_t>(item[0]);
    ds.buf_indices = nb::cast<std::vector<uint32_t>>(item[1]);
    ds.buf_offsets = nb::cast<std::vector<int64_t>>(item[2]);
    auto grid = nb::cast<std::tuple<int, int, int>>(item[3]);
    auto tg = nb::cast<std::tuple<int, int, int>>(item[4]);
    ds.grid[0] = std::get<0>(grid);
    ds.grid[1] = std::get<1>(grid);
    ds.grid[2] = std::get<2>(grid);
    ds.tg[0] = std::get<0>(tg);
    ds.tg[1] = std::get<1>(tg);
    ds.tg[2] = std::get<2>(tg);
    disp_specs.push_back(std::move(ds));
  }
  std::vector<std::vector<int>> group_specs;
  group_specs.reserve(groups.size());
  for (size_t gi = 0; gi < groups.size(); gi++)
    group_specs.push_back(nb::cast<std::vector<int>>(groups[gi]));
  std::vector<int> written;
  for (size_t i = 0; i < written_slot_indices.size(); i++)
    written.push_back(nb::cast<int>(written_slot_indices[i]));
  return alloycore::register_plan(disp_specs, slot_specs, group_specs, written);
}

// g_pending_cb + gpu_sync now live in alloy_core.

// Execute a registered plan with new input arrays — nb shim over
// alloycore::dispatch_plan. pre_copies are (dst_handle, dst_offset, src_handle,
// src_offset, nbytes) GPU-side bulk copies run before the plan's compute (the
// spec-decode DeltaNet recurrent-state rotate). PreCopy resolution + execution
// now live in alloy_core.
static nb::dict py_dispatch_plan(int64_t plan_handle, nb::list input_arrays,
                              bool serialized = false, bool defer_wait = false,
                              nb::list pre_copies = nb::list(),
                              nb::list grid_updates = nb::list()) {
  std::vector<alloycore::PreCopySpec> pcs;
  pcs.reserve(pre_copies.size());
  for (size_t i = 0; i < pre_copies.size(); i++) {
    nb::tuple t = nb::cast<nb::tuple>(pre_copies[i]);
    pcs.push_back({nb::cast<int64_t>(t[0]), nb::cast<int64_t>(t[1]),
                   nb::cast<int64_t>(t[2]), nb::cast<int64_t>(t[3]),
                   nb::cast<int64_t>(t[4])});
  }
  alloycore::PlanTiming timing = alloycore::dispatch_plan(
      plan_handle, parse_input_updates(input_arrays), serialized, defer_wait,
      pcs, parse_grid_updates(grid_updates));
  nb::dict result;
  result["gpu"] = timing.gpu_ms;
  result["encode"] = timing.encode_ms;
  result["wait"] = timing.wait_ms;
  result["copy"] = 0.0;
  result["n_groups"] = timing.n_groups;
  return result;
}

// Dispatch each kernel in its own command buffer for per-dispatch GPU timing.
// Returns list of dicts: [{idx, gpu_us, name, grid, tg}, ...]
static nb::list dispatch_plan_profiled(int64_t plan_handle,
                                       nb::list input_arrays,
                                       nb::list grid_updates = nb::list()) {
  ensure_device();

  Plan *plan;
  {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto it = g_plans.find(plan_handle);
    if (it == g_plans.end())
      throw std::runtime_error("Invalid plan handle");
    plan = &it->second;
  }

  // One-shot prefill profiling: shrink the M-dependent dispatch grids to the
  // real prompt length exactly as dispatch_plan does, so the per-kernel GPU
  // timings reflect the production shrunk launch (not the max-length grid).
  if (grid_updates.size() > 0)
    alloycore::update_plan_dispatch_grids(plan, parse_grid_updates(grid_updates));

  nb::list result;

  @autoreleasepool {

    // Profiled dispatch still resolves raw pointers via get_buffer; training
    // mode clears that cache so external storage mutations are visible.
    if (g_training_mode) {
      std::lock_guard<std::mutex> lock(g_buffer_mutex);
      g_buffer_cache.clear();
    }

    // Resolve raw input pointers for the profiled path.
    for (size_t i = 0; i < input_arrays.size(); i++) {
      nb::tuple entry = nb::cast<nb::tuple>(input_arrays[i]);
      int slot_idx = nb::cast<int>(entry[0]);
      uintptr_t ptr = (uintptr_t)nb::cast<int64_t>(entry[1]);
      size_t nbytes = nb::cast<size_t>(entry[2]);
      plan->slots[slot_idx].ptr = ptr;
      plan->slots[slot_idx].nbytes = nbytes;
    }
    for (size_t i = 0; i < plan->slots.size(); i++) {
      auto &slot = plan->slots[i];
      if (slot.ptr == 0)
        continue;
      auto [buf, offset] = get_buffer((void *)slot.ptr, slot.nbytes);
      slot.metal_buf = buf;
      slot.base_offset = offset;
    }

    int dispatch_idx = 0;
    for (auto &group : plan->groups) {
      for (auto &pd : group) {
        id<MTLCommandBuffer> cb;
        {
          MTLCommandBufferDescriptor *desc =
              [[MTLCommandBufferDescriptor alloc] init];
          desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
          cb = [g_queue commandBufferWithDescriptor:desc];
        }
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
        auto t_commit = std::chrono::high_resolution_clock::now();
        [cb commit];
        [cb waitUntilCompleted];
        auto t_done = std::chrono::high_resolution_clock::now();

        double gpu_us = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
        // Wall around commit+wait: catches driver/scheduling cost that GPU
        // timestamps exclude (it scales with depth on spec-verify plans while
        // gpu_us stays flat — the depth-decay investigation's key signal).
        double wall_us =
            std::chrono::duration<double, std::micro>(t_done - t_commit)
                .count();

        std::string name = "unknown";
        {
          std::lock_guard<std::mutex> lock(g_cache_mutex);
          for (auto &[h, p] : g_pipelines) {
            if (p == pd.pso) {
              auto nit = g_pipeline_names.find(h);
              if (nit != g_pipeline_names.end())
                name = nit->second;
              break;
            }
          }
        }

        nb::dict entry;
        entry["idx"] = dispatch_idx;
        entry["gpu_us"] = gpu_us;
        entry["wall_us"] = wall_us;
        entry["name"] = name;
        entry["grid"] = nb::make_tuple((int)pd.grid.width, (int)pd.grid.height,
                                       (int)pd.grid.depth);
        entry["tg"] = nb::make_tuple((int)pd.tg.width, (int)pd.tg.height,
                                     (int)pd.tg.depth);
        result.append(entry);
        dispatch_idx++;
      }
    }

  } // @autoreleasepool

  return result;
}

// One command buffer per dependency GROUP, committed back-to-back without
// intermediate waits (one queue ⇒ in-order execution + cross-CB write
// visibility, so plan semantics hold). Per-CB GPU timestamps give exact
// per-group attribution INSIDE the production-like pipelined schedule —
// the tool for costs that only manifest when dispatches share a schedule
// (the spec-verify depth scaling: flat per-dispatch, +41ms single-CB).
static nb::list dispatch_plan_group_profiled(int64_t plan_handle,
                                             nb::list input_arrays) {
  ensure_device();

  Plan *plan;
  {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    auto it = g_plans.find(plan_handle);
    if (it == g_plans.end())
      throw std::runtime_error("Invalid plan handle");
    plan = &it->second;
  }

  nb::list result;

  @autoreleasepool {
    alloycore::update_plan_input_slots(plan, parse_input_updates(input_arrays));

    std::vector<id<MTLCommandBuffer>> cbs;
    cbs.reserve(plan->groups.size());
    for (auto &group : plan->groups) {
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      if (!cb)
        throw std::runtime_error("Failed to create command buffer");
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      for (auto &pd : group) {
        [enc setComputePipelineState:pd.pso];
        for (size_t i = 0; i < pd.buf_indices.size(); i++) {
          auto &slot = plan->slots[pd.buf_indices[i]];
          [enc setBuffer:slot.metal_buf
                  offset:plan_buffer_offset(slot, pd, i)
                 atIndex:i];
        }
        [enc dispatchThreadgroups:pd.grid threadsPerThreadgroup:pd.tg];
      }
      [enc endEncoding];
      [cb commit];
      cbs.push_back(cb);
    }
    [cbs.back() waitUntilCompleted];

    for (size_t gi = 0; gi < plan->groups.size(); gi++) {
      id<MTLCommandBuffer> cb = cbs[gi];
      std::string name = "unknown";
      {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        for (auto &[h, p] : g_pipelines) {
          if (p == plan->groups[gi].front().pso) {
            auto nit = g_pipeline_names.find(h);
            if (nit != g_pipeline_names.end())
              name = nit->second;
            break;
          }
        }
      }
      nb::dict entry;
      entry["group"] = (int)gi;
      entry["n"] = (int)plan->groups[gi].size();
      entry["gpu_us"] = (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
      entry["start"] = cb.GPUStartTime;
      entry["end"] = cb.GPUEndTime;
      entry["name"] = name;
      result.append(entry);
    }
  } // @autoreleasepool

  return result;
}

// ---------------------------------------------------------------------------
// nanobind module — functions only
// ---------------------------------------------------------------------------

NB_MODULE(_metal_ext, m) {
  m.doc() =
      "Alloy Metal runtime — function-only API, no Python-side Metal objects";

  m.def("device_info", &py_device_info, "Get Metal device info as dict");
  m.def(
      "set_training_mode", [](bool mode) { g_training_mode = mode; },
      "Enable/disable re-copy of external buffers on each dispatch",
      nb::arg("mode"));
  m.def("compile_msl", &compile_msl, "Compile MSL source → pipeline handle",
        nb::arg("source"), nb::arg("function_name"));
  m.def("compile_metallib", &compile_metallib,
        "Load metallib → pipeline handle", nb::arg("path"),
        nb::arg("function_name"));
  m.def("pipeline_max_threads", &pipeline_max_threads,
        "Max threadgroup size for pipeline", nb::arg("handle"));
  m.def("pipeline_thread_width", &pipeline_thread_width,
        "SIMD width for pipeline", nb::arg("handle"));
  m.def("pipeline_static_threadgroup_memory", &pipeline_static_threadgroup_memory,
        "Static threadgroup memory (bytes) for pipeline", nb::arg("handle"));
  m.def("capture_start", &py_capture_start,
        "Begin a GPU-trace capture of the alloy device → .gputrace document; "
        "returns \"\" on success or an error string", nb::arg("out_path"));
  m.def("capture_stop", &py_capture_stop, "End the GPU-trace capture");
  m.def("dispatch", &dispatch,
        "Dispatch grouped kernels — barriers between groups, not within",
        nb::arg("groups"));
  m.def("clear_buffer_cache", &clear_buffer_cache,
        "Clear Metal buffer cache only");
  m.def("wire_buffers", &wire_buffers,
        "Dispatch-wire each buffer handle's VA resident off the request path "
        "(no residency set; stays off phys_footprint)", nb::arg("handles"));
  m.def("alloc_typed", &alloc_typed,
        "Allocate typed page-aligned buffer → (ndarray, handle, ptr)",
        nb::arg("nbytes"), nb::arg("shape"), nb::arg("dtype"));

  // BufferManager API
  m.def("buf_alloc", &buf_alloc, "Allocate typed page-aligned buffer → handle",
        nb::arg("nbytes"));
  m.def("buf_release", &buf_release, "Release buffer",
        nb::arg("handle"));
  m.def("buf_ptr", &buf_ptr, "Get raw pointer for buffer handle",
        nb::arg("handle"));
  m.def("buf_handle_for_ptr", &buf_handle_for_ptr,
        "Reverse lookup: ptr → handle (-1 if not alloy)", nb::arg("ptr"));
  m.def("buf_numpy", &buf_numpy, "Get numpy view of buffer", nb::arg("handle"),
        nb::arg("shape"), nb::arg("dtype"));
  m.def("buf_nbytes", &buf_nbytes, "Get nbytes for buffer handle",
        nb::arg("handle"));

  // Paged KV pool
  m.def("pool_create", &pool_create,
        "Reserve a vm range wrapped in one bytesNoCopy MTLBuffer → handle",
        nb::arg("nbytes"));
  m.def("pool_slice", &pool_slice,
        "Register a page-aligned pool range as an alloy buffer → handle",
        nb::arg("pool_handle"), nb::arg("offset"), nb::arg("nbytes"));
  m.def("pool_reclaim", &pool_reclaim,
        "MADV_FREE_REUSABLE whole pages in [offset, offset+len) → bytes",
        nb::arg("handle"), nb::arg("offset"), nb::arg("len"));
  m.def("buffer_stats", &buffer_stats,
        "Live alloy Metal-buffer count + total aligned bytes");
  m.def("buffer_dump", &buffer_dump,
        "Top-N live buffers by resident bytes", nb::arg("top") = 20);
  m.def("pool_max_bytes", &pool_max_bytes,
        "Largest pool reservation the device accepts (maxBufferLength)");
  m.def("memory_pressure_level", &memory_pressure_level,
        "Current memory-pressure level: 0 normal, 1 warn, 2 critical");

  // Graph compiler plan API
  m.def("register_plan", &py_register_plan,
        "Register compiled dispatch plan → plan handle", nb::arg("dispatches"),
        nb::arg("slots"), nb::arg("groups"),
        nb::arg("written_slots") = nb::list());
  m.def("dispatch_plan", &py_dispatch_plan,
        "Execute registered plan with new input arrays", nb::arg("plan_handle"),
        nb::arg("input_arrays"), nb::arg("serialized") = false,
        nb::arg("defer_wait") = false, nb::arg("pre_copies") = nb::list(),
        nb::arg("grid_updates") = nb::list());
  m.def("update_plan_intermediate_slots", &update_plan_intermediate_slots,
        "Rebind INTERMEDIATE slots to new alloy buffers (pool growth)",
        nb::arg("plan_handle"), nb::arg("slot_updates"));
  m.def("gpu_sync", &gpu_sync,
        "Wait for any pending async command buffer");
  m.def("dispatch_plan_profiled", &dispatch_plan_profiled,
        "Execute plan with per-dispatch GPU timing", nb::arg("plan_handle"),
        nb::arg("input_arrays"), nb::arg("grid_updates") = nb::list());
  m.def("dispatch_plan_group_profiled", &dispatch_plan_group_profiled,
        "Execute plan with per-dependency-group GPU timing (pipelined CBs)",
        nb::arg("plan_handle"), nb::arg("input_arrays"));
}
