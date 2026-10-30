// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/weight_sync/tiling_utils.h"

#include <sched.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <thread>  // NOLINT
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "hwy/highway.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "tpu_sync/core/numa_thread_pool.h"

namespace tpu_raiden::weight_sync {

bool IsStandardRowMajorTiled(const xla::Shape& shape,
                             const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  if (R < 1) return false;
  if (layout.minor_to_major().size() != R) return false;

  for (int i = 0; i < R; ++i) {
    if (layout.minor_to_major(i) != R - 1 - i) {
      return false;
    }
  }

  if (layout.tiles().empty()) return false;
  for (const auto& t : layout.tiles()) {
    if (t.dimensions().empty()) return false;
    for (int64_t d : t.dimensions()) {
      if (d <= 0) return false;
    }
  }

  const int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  if (layout.tiles().size() == 1) {
    const auto& t0 = layout.tiles(0);
    if (t0.dimensions().size() == 2) return true;
    if (R == 1 && t0.dimensions().size() == 1) return true;
    return false;
  }

  if (layout.tiles().size() == 2) {
    const auto& t0 = layout.tiles(0);
    const auto& t1 = layout.tiles(1);
    if (t0.dimensions().size() == 2 && t1.dimensions().size() == 2) {
      int64_t tile_H = t0.dimension(0);
      int64_t P = t1.dimension(0);
      int64_t sub_W = t1.dimension(1);
      return (sub_W == 1 &&
              ((P == 2 && itemsize == 2) || (P == 4 && itemsize == 1)) &&
              tile_H % P == 0);
    }
    if (R == 1 && t0.dimensions().size() == 1 && t1.dimensions().size() == 2) {
      int64_t P = t1.dimension(0);
      int64_t sub_W = t1.dimension(1);
      return (sub_W == 1 &&
              ((P == 2 && itemsize == 2) || (P == 4 && itemsize == 1)));
    }
    if (R == 1 && t0.dimensions().size() == 1 && t1.dimensions().size() == 1) {
      return (t0.dimension(0) % t1.dimension(0) == 0);
    }
    return false;
  }

  if (layout.tiles().size() == 3) {
    const auto& t0 = layout.tiles(0);
    const auto& t1 = layout.tiles(1);
    const auto& t2 = layout.tiles(2);
    if (R == 1 && t0.dimensions().size() == 1 && t1.dimensions().size() == 1 &&
        t2.dimensions().size() == 2) {
      int64_t outer_W = t0.dimension(0);
      int64_t inner_W = t1.dimension(0);
      int64_t P = t2.dimension(0);
      int64_t sub_W = t2.dimension(1);
      return (sub_W == 1 &&
              ((P == 2 && itemsize == 2) || (P == 4 && itemsize == 1)) &&
              outer_W % inner_W == 0 && (outer_W / inner_W) % P == 0);
    }
    return false;
  }

  return false;
}

bool IsStandardColMajorTiled(const xla::Shape& shape,
                             const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  if (R < 2) return false;
  if (layout.minor_to_major().size() != R) return false;

  if (layout.minor_to_major(0) != R - 2 || layout.minor_to_major(1) != R - 1) {
    return false;
  }
  for (int i = 2; i < R; ++i) {
    if (layout.minor_to_major(i) != R - 1 - i) {
      return false;
    }
  }

  if (layout.tiles().empty()) return false;
  for (const auto& t : layout.tiles()) {
    if (t.dimensions().empty()) return false;
    for (int64_t d : t.dimensions()) {
      if (d <= 0) return false;
    }
  }

  if (layout.tiles().size() == 1) {
    return layout.tiles(0).dimensions().size() == 2;
  }

  if (layout.tiles().size() == 2) {
    const auto& t0 = layout.tiles(0);
    const auto& t1 = layout.tiles(1);
    if (t0.dimensions().size() != 2 || t1.dimensions().size() != 2) {
      return false;
    }
    int64_t tile_H = t0.dimension(0);
    int64_t P = t1.dimension(0);
    int64_t sub_W = t1.dimension(1);
    return (sub_W == 1 && (P == 2 || P == 4) && tile_H % P == 0);
  }

  return false;
}

namespace {

// Tensors below 2 * kParallelizationThresholdBytes (8 MB) are tiled inline
// directly on the calling thread to preserve private L2 cache locality and
// avoid thread pool scheduling overhead. For larger tensors, each parallel
// chunk targets at least kParallelizationThresholdBytes (4 MB).
constexpr int64_t kParallelizationThresholdBytes = 4 * 1024 * 1024;  // 4 MB

// Maximum sub-task chunks per tensor when parallelizing across the thread pool,
// ensuring a single large tensor does not monopolize the pool during concurrent
// layer arrivals while allowing large MoE tensors (>= 32 MB) to scale up to 8
// threads.
constexpr int64_t kMaxChunksPerTensor = 8;

constexpr int64_t kMaxNumThreads = 16;

// Geometry of a standard row-major tiled tensor (see IsStandardRowMajorTiled),
// viewed as |batch_size| matrices of |H| x |W| elements tiled by
// |tile_H| x |tile_W| with an optional minor packing factor. Shared by the
// out-of-place and in-place row-major tilers so both agree on byte offsets.
struct RowMajorTileGeometry {
  int64_t H = 1;
  int64_t W = 1;
  int64_t itemsize = 1;
  int64_t tile_H = 1;
  int64_t tile_W = 1;
  int64_t packing_factor = 1;
  int64_t num_tiles_0 = 0;
  int64_t num_tiles_1 = 0;
  int64_t tile_size_bytes = 0;
  int64_t batch_size = 1;
  int64_t matrix_size_bytes = 0;
  int64_t tiled_matrix_size_bytes = 0;
  bool has_padding = false;

  int64_t total_tiled_bytes() const {
    return batch_size * tiled_matrix_size_bytes;
  }
};

RowMajorTileGeometry ComputeRowMajorTileGeometry(const xla::Shape& shape,
                                                 const xla::Layout& layout) {
  RowMajorTileGeometry g;
  const int R = shape.dimensions().size();
  g.H = (R == 1) ? 1 : shape.dimensions(layout.minor_to_major(1));
  g.W = shape.dimensions(layout.minor_to_major(0));
  g.itemsize = xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  g.tile_H = (tile.dimensions().size() <= 1) ? 1 : tile.dimension(0);
  g.tile_W = (tile.dimensions().size() == 1)   ? tile.dimension(0)
             : (tile.dimensions().size() >= 2) ? tile.dimension(1)
                                               : 1;
  if (layout.tiles().size() >= 2) {
    g.packing_factor = layout.tiles(1).dimension(0);
  }

  g.num_tiles_0 = xla::CeilOfRatio(g.H, g.tile_H);
  g.num_tiles_1 = xla::CeilOfRatio(g.W, g.tile_W);
  g.tile_size_bytes = g.tile_H * g.tile_W * g.itemsize;

  for (int i = 2; i < R; ++i) {
    g.batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  g.matrix_size_bytes = g.H * g.W * g.itemsize;
  g.tiled_matrix_size_bytes = g.num_tiles_0 * g.num_tiles_1 * g.tile_size_bytes;
  g.has_padding = (g.H % g.tile_H != 0) || (g.W % g.tile_W != 0);
  return g;
}

// Geometry of a 1D tensor tiled as {0:T(tile_W)(P,1)} or
// {0:T(outer_W)(tile_W)(P,1)}: |total_groups| packing groups of
// |P| * |tile_W| elements each.
struct Packed1DTileGeometry {
  int64_t P = 1;
  int64_t tile_W = 1;
  int64_t total_groups = 0;
  int64_t group_elems = 0;
  int64_t group_bytes = 0;
};

Packed1DTileGeometry ComputePacked1DTileGeometry(int64_t W, int64_t itemsize,
                                                 const xla::Layout& layout) {
  Packed1DTileGeometry g;
  const int num_tiles = layout.tiles().size();
  g.P = layout.tiles(num_tiles - 1).dimension(0);
  if (num_tiles == 2) {
    g.tile_W = layout.tiles(0).dimension(0);
    g.total_groups = xla::CeilOfRatio(xla::CeilOfRatio(W, g.tile_W), g.P);
  } else {
    const int64_t outer_W = layout.tiles(0).dimension(0);
    g.tile_W = layout.tiles(1).dimension(0);
    g.total_groups =
        xla::CeilOfRatio(W, outer_W) * ((outer_W / g.tile_W) / g.P);
  }
  g.group_elems = g.P * g.tile_W;
  g.group_bytes = g.group_elems * itemsize;
  return g;
}

tpu_raiden::NumaThreadPool* GetThreadPool() {
  static absl::NoDestructor<tpu_raiden::NumaThreadPool> global_pool([]() {
    int64_t hw_threads =
        static_cast<int64_t>(std::thread::hardware_concurrency());
    return static_cast<size_t>(
        hw_threads > 0 ? std::clamp<int64_t>(hw_threads, 4, kMaxNumThreads)
                       : kMaxNumThreads);
  }());
  return global_pool.get();
}

// Splits |total_tasks| into chunks and runs |run_task| on them: one chunk on
// the calling thread, the rest as helper tasks on the shared pool pinned to
// |numa_node| when set.
template <typename TaskFn>
void ExecuteParallelTasks(int64_t total_tasks, int64_t num_tiles_0,
                          int64_t desired_chunks, std::optional<int> numa_node,
                          TaskFn&& run_task) {
  tpu_raiden::NumaThreadPool* target_pool = GetThreadPool();
  int64_t pool_threads = static_cast<int64_t>(target_pool->num_threads());
  int64_t max_threads =
      std::min<int64_t>({kMaxChunksPerTensor, pool_threads, desired_chunks});

  if (max_threads > 1 && total_tasks >= max_threads) {
    int64_t num_workers = max_threads - 1;
    int64_t step = std::max<int64_t>(1, total_tasks / (max_threads * 8));
    std::atomic<int64_t> next_idx(0);
    std::atomic<int64_t> remaining_workers(num_workers);

    auto work_loop = [&]() {
      while (true) {
        int64_t begin = next_idx.fetch_add(step, std::memory_order_relaxed);
        if (begin >= total_tasks) {
          break;
        }
        int64_t end = std::min(begin + step, total_tasks);
        for (int64_t i = begin; i < end; ++i) {
          run_task(i / num_tiles_0, i % num_tiles_0);
        }
      }
    };

    // Schedule num_workers helper tasks to the pool
    for (int64_t t = 0; t < num_workers; ++t) {
      target_pool->Schedule(numa_node, [&work_loop, &next_idx,
                                        &remaining_workers, total_tasks]() {
        if (next_idx.load(std::memory_order_relaxed) < total_tasks) {
          work_loop();
        }
        remaining_workers.fetch_sub(1, std::memory_order_release);
      });
    }

    // Execute directly on the calling thread via shared dynamic work-stealing
    work_loop();

    // While waiting for helper tasks to complete, drain pending pool tasks
    while (remaining_workers.load(std::memory_order_acquire) > 0) {
      if (target_pool->ExecuteOneTask()) {
        continue;
      }
      sched_yield();
    }
  } else {
    for (int64_t i = 0; i < total_tasks; ++i) {
      run_task(i / num_tiles_0, i % num_tiles_0);
    }
  }
}

template <typename F>
decltype(auto) DispatchByPackingFactor(int64_t packing_factor, F&& f) {
  switch (packing_factor) {
    case 1:
      return f(std::integral_constant<int64_t, 1>{});
    case 2:
      return f(std::integral_constant<int64_t, 2>{});
    case 4:
      return f(std::integral_constant<int64_t, 4>{});
    default:
      return f(std::integral_constant<int64_t, 1>{});
  }
}

// Dispatches row copy operations to compile-time specialized fixed-width
// vector copy loops for common TPU tile row sizes (FP8, BF16, FP32 with tile_W
// 128 or 8), falling back to generic std::memcpy for arbitrary dimensions.
template <typename F>
decltype(auto) DispatchByRowBytes(int64_t row_bytes, F&& f) {
  switch (row_bytes) {
    case 8:
      return f(std::integral_constant<size_t, 8>{});
    case 16:
      return f(std::integral_constant<size_t, 16>{});
    case 32:
      return f(std::integral_constant<size_t, 32>{});
    case 64:
      return f(std::integral_constant<size_t, 64>{});
    case 128:
      return f(std::integral_constant<size_t, 128>{});
    case 256:
      return f(std::integral_constant<size_t, 256>{});
    case 512:
      return f(std::integral_constant<size_t, 512>{});
    default:
      return f(std::integral_constant<size_t, 0>{});
  }
}

namespace hn = hwy::HWY_NAMESPACE;

// Copies valid_bytes from src to dst and zero-fills the rest of total_row_bytes
// using portable Highway SIMD vector operations. Never reads past
// src + valid_bytes or writes past dst + total_row_bytes.
inline void CopyRowWithPaddingHighway(const uint8_t* src, uint8_t* dst,
                                      size_t valid_bytes,
                                      size_t total_row_bytes) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);
  size_t offset = 0;

  for (; offset + N <= valid_bytes; offset += N) {
    const auto v = hn::LoadU(d, src + offset);
    hn::StoreU(v, d, dst + offset);
  }

  if (offset < valid_bytes) {
    // LoadN zeroes the lanes past the valid tail, which pads the row.
    const auto v = hn::LoadN(d, src + offset, valid_bytes - offset);
    const size_t num_stored = std::min(N, total_row_bytes - offset);
    hn::StoreN(v, d, dst + offset, num_stored);
    offset += num_stored;
  }

  const auto zeros = hn::Zero(d);
  for (; offset + N <= total_row_bytes; offset += N) {
    hn::StoreU(zeros, d, dst + offset);
  }

  if (offset < total_row_bytes) {
    hn::StoreN(zeros, d, dst + offset, total_row_bytes - offset);
  }
}

// Detiles valid_bytes from src to dst using Highway SIMD vectors. Never reads
// past src + valid_bytes or writes past dst + valid_bytes.
inline void DetileRowWithPaddingHighway(const uint8_t* src, uint8_t* dst,
                                        size_t valid_bytes) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);
  size_t offset = 0;

  for (; offset + N <= valid_bytes; offset += N) {
    const auto v = hn::LoadU(d, src + offset);
    hn::StoreU(v, d, dst + offset);
  }

  if (offset < valid_bytes) {
    const size_t rem = valid_bytes - offset;
    hn::StoreN(hn::LoadN(d, src + offset, rem), d, dst + offset, rem);
  }
}

// Zeroes total_row_bytes using Highway SIMD vectors.
inline void ZeroRowHighway(uint8_t* dst, size_t total_row_bytes) {
  const hn::ScalableTag<uint8_t> d;
  const size_t N = hn::Lanes(d);
  const auto zeros = hn::Zero(d);
  size_t offset = 0;
  for (; offset + N <= total_row_bytes; offset += N) {
    hn::StoreU(zeros, d, dst + offset);
  }
  if (offset < total_row_bytes) {
    hn::StoreN(zeros, d, dst + offset, total_row_bytes - offset);
  }
}

// Copies a single tile from the source batch pointer to the destination tile
// pointer when no padding is needed. Template-specialized on compile-time
// packing factor and row byte width to enable inlined SIMD load/store vector
// instructions.
template <int64_t kPackingFactor, size_t kRowBytes>
void CopyTilePackedNoPadding(const uint8_t* src_batch_ptr,
                             uint8_t* dst_tile_ptr, int64_t tile_row,
                             int64_t tile_col, int64_t tile_H, int64_t tile_W,
                             int64_t W, int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  if constexpr (kPackingFactor == 1) {
    int64_t row_stride = (kRowBytes > 0) ? kRowBytes : (tile_W * itemsize);
    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      const uint8_t* src_row_ptr =
          src_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
      uint8_t* dst_row_ptr = dst_tile_ptr + r * row_stride;

      if constexpr (kRowBytes > 0) {
        std::memcpy(dst_row_ptr, src_row_ptr, kRowBytes);
      } else {
        std::memcpy(dst_row_ptr, src_row_ptr, row_stride);
      }
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);
    const hn::ScalableTag<uint16_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      const uint16_t* src0 = reinterpret_cast<const uint16_t*>(
          src_batch_ptr + (r0 * W + logical_col_start) * sizeof(uint16_t));
      const uint16_t* src1 = reinterpret_cast<const uint16_t*>(
          src_batch_ptr + (r1 * W + logical_col_start) * sizeof(uint16_t));
      uint16_t* dst_group =
          reinterpret_cast<uint16_t*>(dst_tile_ptr + g * group_stride_bytes);

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        auto v0 = hn::LoadU(d, src0 + c);
        auto v1 = hn::LoadU(d, src1 + c);
        hn::StoreInterleaved2(v0, v1, d, dst_group + c * 2);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst_group[2 * c + 0] = src0[c];
        dst_group[2 * c + 1] = src1[c];
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      const uint8_t* src0 = src_batch_ptr + (r0 * W + logical_col_start);
      const uint8_t* src1 = src_batch_ptr + ((r0 + 1) * W + logical_col_start);
      const uint8_t* src2 = src_batch_ptr + ((r0 + 2) * W + logical_col_start);
      const uint8_t* src3 = src_batch_ptr + ((r0 + 3) * W + logical_col_start);
      uint8_t* dst_group = dst_tile_ptr + g * group_stride_bytes;

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        auto v0 = hn::LoadU(d, src0 + c);
        auto v1 = hn::LoadU(d, src1 + c);
        auto v2 = hn::LoadU(d, src2 + c);
        auto v3 = hn::LoadU(d, src3 + c);
        hn::StoreInterleaved4(v0, v1, v2, v3, d, dst_group + c * 4);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst_group[4 * c + 0] = src0[c];
        dst_group[4 * c + 1] = src1[c];
        dst_group[4 * c + 2] = src2[c];
        dst_group[4 * c + 3] = src3[c];
      }
    }
  }
}

// Copies a single tile from the source batch pointer to the destination tile
// pointer, handling padding if the tile is partially or fully out of bounds.
// Out-of-bounds elements in the destination tile are zero-initialized.
template <int64_t kPackingFactor>
void CopyTilePackedWithPadding(const uint8_t* src_batch_ptr,
                               uint8_t* dst_tile_ptr, int64_t tile_row,
                               int64_t tile_col, int64_t tile_H, int64_t tile_W,
                               int64_t H, int64_t W, int64_t itemsize,
                               int64_t tile_size_bytes) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t valid_elements = std::min(tile_W, W - logical_col_start);
  if (valid_elements <= 0) {
    ZeroRowHighway(dst_tile_ptr, static_cast<size_t>(tile_size_bytes));
    return;
  }

  if constexpr (kPackingFactor == 1) {
    size_t valid_bytes = static_cast<size_t>(valid_elements * itemsize);
    size_t total_row_bytes = static_cast<size_t>(tile_W * itemsize);

    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      uint8_t* dst_row_ptr = dst_tile_ptr + r * total_row_bytes;
      if (logical_row >= H) {
        ZeroRowHighway(dst_row_ptr, total_row_bytes);
        continue;
      }
      const uint8_t* src_row_ptr =
          src_batch_ptr + (logical_row * W + logical_col_start) * itemsize;

      CopyRowWithPaddingHighway(src_row_ptr, dst_row_ptr, valid_bytes,
                                total_row_bytes);
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      uint16_t* dst_group =
          reinterpret_cast<uint16_t*>(dst_tile_ptr + g * group_stride_bytes);

      if (r0 >= H && r1 >= H) {
        ZeroRowHighway(reinterpret_cast<uint8_t*>(dst_group),
                       static_cast<size_t>(group_stride_bytes));
        continue;
      }

      const uint16_t* src0 =
          (r0 < H) ? reinterpret_cast<const uint16_t*>(
                         src_batch_ptr +
                         (r0 * W + logical_col_start) * sizeof(uint16_t))
                   : nullptr;
      const uint16_t* src1 =
          (r1 < H) ? reinterpret_cast<const uint16_t*>(
                         src_batch_ptr +
                         (r1 * W + logical_col_start) * sizeof(uint16_t))
                   : nullptr;

      for (int64_t c = 0; c < tile_W; ++c) {
        if (c < valid_elements) {
          dst_group[2 * c + 0] = src0 ? src0[c] : 0;
          dst_group[2 * c + 1] = src1 ? src1[c] : 0;
        } else {
          dst_group[2 * c + 0] = 0;
          dst_group[2 * c + 1] = 0;
        }
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      uint8_t* dst_group = dst_tile_ptr + g * group_stride_bytes;

      if (r0 >= H) {
        ZeroRowHighway(dst_group, static_cast<size_t>(group_stride_bytes));
        continue;
      }

      const uint8_t* src[4];
      for (int p = 0; p < 4; ++p) {
        src[p] = (r0 + p < H)
                     ? (src_batch_ptr + ((r0 + p) * W + logical_col_start))
                     : nullptr;
      }

      for (int64_t c = 0; c < tile_W; ++c) {
        if (c < valid_elements) {
          for (int p = 0; p < 4; ++p) {
            dst_group[4 * c + p] = src[p] ? src[p][c] : 0;
          }
        } else {
          for (int p = 0; p < 4; ++p) {
            dst_group[4 * c + p] = 0;
          }
        }
      }
    }
  }
}

template <int64_t kPackingFactor, size_t kRowBytes>
void DetileSingleTilePackedNoPadding(const uint8_t* src_tile_ptr,
                                     uint8_t* dst_batch_ptr, int64_t tile_row,
                                     int64_t tile_col, int64_t tile_H,
                                     int64_t tile_W, int64_t W,
                                     int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  if constexpr (kPackingFactor == 1) {
    int64_t row_stride = (kRowBytes > 0) ? kRowBytes : (tile_W * itemsize);
    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      uint8_t* dst_row_ptr =
          dst_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
      const uint8_t* src_row_ptr = src_tile_ptr + r * row_stride;

      if constexpr (kRowBytes > 0) {
        std::memcpy(dst_row_ptr, src_row_ptr, kRowBytes);
      } else {
        std::memcpy(dst_row_ptr, src_row_ptr, row_stride);
      }
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);
    const hn::ScalableTag<uint16_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      uint16_t* dst0 = reinterpret_cast<uint16_t*>(
          dst_batch_ptr + (r0 * W + logical_col_start) * sizeof(uint16_t));
      uint16_t* dst1 = reinterpret_cast<uint16_t*>(
          dst_batch_ptr + (r1 * W + logical_col_start) * sizeof(uint16_t));
      const uint16_t* src_group = reinterpret_cast<const uint16_t*>(
          src_tile_ptr + g * group_stride_bytes);

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        hn::Vec<decltype(d)> v0, v1;
        hn::LoadInterleaved2(d, src_group + c * 2, v0, v1);
        hn::StoreU(v0, d, dst0 + c);
        hn::StoreU(v1, d, dst1 + c);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst0[c] = src_group[2 * c + 0];
        dst1[c] = src_group[2 * c + 1];
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);
    const hn::ScalableTag<uint8_t> d;
    const size_t N = hn::Lanes(d);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      uint8_t* dst0 = dst_batch_ptr + (r0 * W + logical_col_start);
      uint8_t* dst1 = dst_batch_ptr + ((r0 + 1) * W + logical_col_start);
      uint8_t* dst2 = dst_batch_ptr + ((r0 + 2) * W + logical_col_start);
      uint8_t* dst3 = dst_batch_ptr + ((r0 + 3) * W + logical_col_start);
      const uint8_t* src_group = src_tile_ptr + g * group_stride_bytes;

      size_t c = 0;
      for (; c + N <= static_cast<size_t>(tile_W); c += N) {
        hn::Vec<decltype(d)> v0, v1, v2, v3;
        hn::LoadInterleaved4(d, src_group + c * 4, v0, v1, v2, v3);
        hn::StoreU(v0, d, dst0 + c);
        hn::StoreU(v1, d, dst1 + c);
        hn::StoreU(v2, d, dst2 + c);
        hn::StoreU(v3, d, dst3 + c);
      }
      for (; c < static_cast<size_t>(tile_W); ++c) {
        dst0[c] = src_group[4 * c + 0];
        dst1[c] = src_group[4 * c + 1];
        dst2[c] = src_group[4 * c + 2];
        dst3[c] = src_group[4 * c + 3];
      }
    }
  }
}

template <int64_t kPackingFactor>
void DetileSingleTilePackedWithPadding(const uint8_t* src_tile_ptr,
                                       uint8_t* dst_batch_ptr, int64_t tile_row,
                                       int64_t tile_col, int64_t tile_H,
                                       int64_t tile_W, int64_t H, int64_t W,
                                       int64_t itemsize) {
  int64_t logical_col_start = tile_col * tile_W;
  int64_t valid_elements = std::min(tile_W, W - logical_col_start);
  if (valid_elements <= 0) {
    return;
  }

  if constexpr (kPackingFactor == 1) {
    size_t valid_bytes = static_cast<size_t>(valid_elements * itemsize);
    size_t total_row_bytes = static_cast<size_t>(tile_W * itemsize);

    for (int64_t r = 0; r < tile_H; ++r) {
      int64_t logical_row = tile_row * tile_H + r;
      if (logical_row >= H) {
        continue;
      }
      uint8_t* dst_row_ptr =
          dst_batch_ptr + (logical_row * W + logical_col_start) * itemsize;
      const uint8_t* src_row_ptr = src_tile_ptr + r * total_row_bytes;

      DetileRowWithPaddingHighway(src_row_ptr, dst_row_ptr, valid_bytes);
    }
  } else if constexpr (kPackingFactor == 2) {
    int64_t num_groups = tile_H / 2;
    int64_t group_stride_bytes = tile_W * 2 * sizeof(uint16_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 2;
      int64_t r1 = r0 + 1;
      if (r0 >= H && r1 >= H) {
        continue;
      }
      uint16_t* dst0 = (r0 < H)
                           ? reinterpret_cast<uint16_t*>(
                                 dst_batch_ptr + (r0 * W + logical_col_start) *
                                                     sizeof(uint16_t))
                           : nullptr;
      uint16_t* dst1 = (r1 < H)
                           ? reinterpret_cast<uint16_t*>(
                                 dst_batch_ptr + (r1 * W + logical_col_start) *
                                                     sizeof(uint16_t))
                           : nullptr;
      const uint16_t* src_group = reinterpret_cast<const uint16_t*>(
          src_tile_ptr + g * group_stride_bytes);

      for (int64_t c = 0; c < valid_elements; ++c) {
        if (dst0) dst0[c] = src_group[2 * c + 0];
        if (dst1) dst1[c] = src_group[2 * c + 1];
      }
    }
  } else if constexpr (kPackingFactor == 4) {
    int64_t num_groups = tile_H / 4;
    int64_t group_stride_bytes = tile_W * 4 * sizeof(uint8_t);

    for (int64_t g = 0; g < num_groups; ++g) {
      int64_t r0 = tile_row * tile_H + g * 4;
      if (r0 >= H) {
        continue;
      }
      uint8_t* dst[4];
      for (int p = 0; p < 4; ++p) {
        dst[p] = (r0 + p < H)
                     ? (dst_batch_ptr + ((r0 + p) * W + logical_col_start))
                     : nullptr;
      }
      const uint8_t* src_group = src_tile_ptr + g * group_stride_bytes;

      for (int64_t c = 0; c < valid_elements; ++c) {
        for (int p = 0; p < 4; ++p) {
          if (dst[p]) dst[p][c] = src_group[4 * c + p];
        }
      }
    }
  }
}

// Tiles a 1D buffer when the outer tile is 1D (e.g. {0:T(128)},
// {0:T(1024)(128)}, {0:T(128)(P,1)}, or {0:T(1024)(128)(P,1)}).
absl::Status TileBuffer1DOptimized(const uint8_t* src_linear,
                                   uint8_t* dst_tiled, const xla::Shape& shape,
                                   const xla::Layout& layout) {
  const int64_t W = shape.dimensions(0);
  const int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const int num_tiles = layout.tiles().size();
  const auto& last_tile = layout.tiles(num_tiles - 1);

  // Unpacked 1D tiling: {0:T(T0)} or {0:T(T0)(T1)}.
  if (last_tile.dimensions().size() == 1) {
    const int64_t outer_W = layout.tiles(0).dimension(0);
    const int64_t total_physical_elements =
        xla::CeilOfRatio(W, outer_W) * outer_W;
    if (W > 0) {
      std::memcpy(dst_tiled, src_linear, W * itemsize);
    }
    if (total_physical_elements > W) {
      ZeroRowHighway(
          dst_tiled + W * itemsize,
          static_cast<size_t>((total_physical_elements - W) * itemsize));
    }
    return absl::OkStatus();
  }

  // Packed 1D tiling: {0:T(tile_W)(P,1)} or {0:T(outer_W)(tile_W)(P,1)}.
  const Packed1DTileGeometry geom =
      ComputePacked1DTileGeometry(W, itemsize, layout);
  const int64_t P = geom.P;
  const int64_t tile_W = geom.tile_W;
  const int64_t total_groups = geom.total_groups;
  const int64_t group_elems = geom.group_elems;
  const int64_t group_bytes = geom.group_bytes;

  DispatchByPackingFactor(P, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    for (int64_t g = 0; g < total_groups; ++g) {
      uint8_t* dst_group_ptr = dst_tiled + g * group_bytes;
      const int64_t group_start = g * group_elems;
      if (group_start + group_elems <= W) {
        CopyTilePackedNoPadding<kPackingFactor, 0>(
            src_linear, dst_group_ptr, /*tile_row=*/g, /*tile_col=*/0,
            /*tile_H=*/kPackingFactor, tile_W, /*W=*/tile_W, itemsize);
      } else if (group_start >= W) {
        ZeroRowHighway(dst_group_ptr, static_cast<size_t>(group_bytes));
      } else {
        ZeroRowHighway(dst_group_ptr, static_cast<size_t>(group_bytes));
        for (int64_t p = 0; p < kPackingFactor; ++p) {
          const int64_t row_start = (g * kPackingFactor + p) * tile_W;
          for (int64_t c = 0; c < tile_W; ++c) {
            const int64_t idx = row_start + c;
            if (idx < W) {
              std::memcpy(dst_group_ptr + (c * kPackingFactor + p) * itemsize,
                          src_linear + idx * itemsize, itemsize);
            }
          }
        }
      }
    }
  });

  return absl::OkStatus();
}

absl::Status DetileBuffer1DOptimized(const uint8_t* src_tiled,
                                     uint8_t* dst_linear,
                                     const xla::Shape& shape,
                                     const xla::Layout& layout) {
  const int64_t W = shape.dimensions(0);
  const int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const int num_tiles = layout.tiles().size();
  const auto& last_tile = layout.tiles(num_tiles - 1);

  // Unpacked 1D tiling: {0:T(T0)} or {0:T(T0)(T1)}.
  if (last_tile.dimensions().size() == 1) {
    if (W > 0) {
      std::memcpy(dst_linear, src_tiled, W * itemsize);
    }
    return absl::OkStatus();
  }

  // Packed 1D tiling: {0:T(tile_W)(P,1)} or {0:T(outer_W)(tile_W)(P,1)}.
  const Packed1DTileGeometry geom =
      ComputePacked1DTileGeometry(W, itemsize, layout);
  const int64_t P = geom.P;
  const int64_t tile_W = geom.tile_W;
  const int64_t total_groups = geom.total_groups;
  const int64_t group_elems = geom.group_elems;
  const int64_t group_bytes = geom.group_bytes;

  DispatchByPackingFactor(P, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    for (int64_t g = 0; g < total_groups; ++g) {
      const uint8_t* src_group_ptr = src_tiled + g * group_bytes;
      const int64_t group_start = g * group_elems;
      if (group_start + group_elems <= W) {
        DetileSingleTilePackedNoPadding<kPackingFactor, 0>(
            src_group_ptr, dst_linear, /*tile_row=*/g, /*tile_col=*/0,
            /*tile_H=*/kPackingFactor, tile_W, /*W=*/tile_W, itemsize);
      } else if (group_start >= W) {
        break;
      } else {
        for (int64_t p = 0; p < kPackingFactor; ++p) {
          const int64_t row_start = (g * kPackingFactor + p) * tile_W;
          for (int64_t c = 0; c < tile_W; ++c) {
            const int64_t idx = row_start + c;
            if (idx < W) {
              std::memcpy(dst_linear + idx * itemsize,
                          src_group_ptr + (c * kPackingFactor + p) * itemsize,
                          itemsize);
            }
          }
        }
      }
    }
  });

  return absl::OkStatus();
}

// Tiles a buffer using an optimized path for standard row-major layouts.
// It avoids global zero-initialization of the destination buffer to prevent
// CPU cache pollution, instead zeroing padding elements locally per tile.
absl::Status TileBufferNDOptimized(
    const uint8_t* src_linear, uint8_t* dst_tiled, const xla::Shape& shape,
    const xla::Layout& layout, std::optional<int> numa_node = std::nullopt) {
  const int R = shape.dimensions().size();
  if (R == 1 && layout.tiles(0).dimensions().size() == 1) {
    return TileBuffer1DOptimized(src_linear, dst_tiled, shape, layout);
  }

  const RowMajorTileGeometry geom = ComputeRowMajorTileGeometry(shape, layout);
  const int64_t H = geom.H;
  const int64_t W = geom.W;
  const int64_t itemsize = geom.itemsize;
  const int64_t tile_H = geom.tile_H;
  const int64_t tile_W = geom.tile_W;
  const int64_t packing_factor = geom.packing_factor;
  const int64_t num_tiles_0 = geom.num_tiles_0;
  const int64_t num_tiles_1 = geom.num_tiles_1;
  const int64_t tile_size_bytes = geom.tile_size_bytes;
  const int64_t batch_size = geom.batch_size;
  const int64_t matrix_size_bytes = geom.matrix_size_bytes;
  const int64_t tiled_matrix_size_bytes = geom.tiled_matrix_size_bytes;

  const bool has_padding = geom.has_padding;

  // Fast-path: When the tiled layout is byte-for-byte identical to the linear
  // layout (e.g. single column of tiles W == tile_W with no vertical padding
  // H % tile_H == 0, or 1D tensor with tile_H == 1 and no padding) AND
  // packing_factor == 1.
  if (packing_factor == 1 && !has_padding &&
      (W == tile_W || (H == 1 && tile_H == 1))) {
    std::memcpy(dst_tiled, src_linear, batch_size * matrix_size_bytes);
    return absl::OkStatus();
  }

  // Fast-path for 1D tensors (H == 1) with packing_factor == 1.
  if (packing_factor == 1 && H == 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
      uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t logical_col_start = tile_col * tile_W;
        int64_t valid_elements = std::min(tile_W, W - logical_col_start);
        uint8_t* dst_tile_ptr = dst_batch_ptr + tile_col * tile_size_bytes;
        if (valid_elements > 0) {
          std::memcpy(dst_tile_ptr,
                      src_batch_ptr + logical_col_start * itemsize,
                      valid_elements * itemsize);
          if (valid_elements < tile_W) {
            std::memset(dst_tile_ptr + valid_elements * itemsize, 0,
                        (tile_W - valid_elements) * itemsize);
          }
        } else {
          std::memset(dst_tile_ptr, 0, tile_W * itemsize);
        }
        if (tile_H > 1) {
          std::memset(dst_tile_ptr + tile_W * itemsize, 0,
                      (tile_H - 1) * tile_W * itemsize);
        }
      }
    }
    return absl::OkStatus();
  }

  int64_t total_tasks = batch_size * num_tiles_0;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t desired_chunks =
      std::max<int64_t>(1, total_bytes / kParallelizationThresholdBytes);
  int64_t row_bytes = (packing_factor == 1) ? (tile_W * itemsize) : 0;

  DispatchByPackingFactor(packing_factor, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    if (desired_chunks <= 1 || total_tasks <= 1) {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        if (!has_padding) {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                uint8_t* dst_tile_ptr =
                    dst_batch_ptr + tile_index * tile_size_bytes;
                CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              }
            }
          }
        } else {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              bool is_row_interior = (tile_row * tile_H + tile_H <= H);
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                uint8_t* dst_tile_ptr =
                    dst_batch_ptr + tile_index * tile_size_bytes;
                bool is_col_interior = (tile_col * tile_W + tile_W <= W);
                if (is_row_interior && is_col_interior) {
                  CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                      src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                      tile_W, W, itemsize);
                } else {
                  CopyTilePackedWithPadding<kPackingFactor>(
                      src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                      tile_W, H, W, itemsize, tile_size_bytes);
                }
              }
            }
          }
        }
      });
    } else {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        auto run_task = [&](int64_t b, int64_t tile_row) {
          const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;

          if (!has_padding) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              uint8_t* dst_tile_ptr =
                  dst_batch_ptr + tile_index * tile_size_bytes;
              CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                  src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                  tile_W, W, itemsize);
            }
          } else {
            bool is_row_interior = (tile_row * tile_H + tile_H <= H);
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              uint8_t* dst_tile_ptr =
                  dst_batch_ptr + tile_index * tile_size_bytes;
              bool is_col_interior = (tile_col * tile_W + tile_W <= W);
              if (is_row_interior && is_col_interior) {
                CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              } else {
                CopyTilePackedWithPadding<kPackingFactor>(
                    src_batch_ptr, dst_tile_ptr, tile_row, tile_col, tile_H,
                    tile_W, H, W, itemsize, tile_size_bytes);
              }
            }
          }
        };

        ExecuteParallelTasks(total_tasks, num_tiles_0, desired_chunks,
                             numa_node, run_task);
      });
    }
  });

  return absl::OkStatus();
}

absl::Status DetileBufferNDOptimized(
    const uint8_t* src_tiled, uint8_t* dst_linear, const xla::Shape& shape,
    const xla::Layout& layout, std::optional<int> numa_node = std::nullopt) {
  const int R = shape.dimensions().size();
  if (R == 1 && layout.tiles(0).dimensions().size() == 1) {
    return DetileBuffer1DOptimized(src_tiled, dst_linear, shape, layout);
  }

  const RowMajorTileGeometry geom = ComputeRowMajorTileGeometry(shape, layout);
  const int64_t H = geom.H;
  const int64_t W = geom.W;
  const int64_t itemsize = geom.itemsize;
  const int64_t tile_H = geom.tile_H;
  const int64_t tile_W = geom.tile_W;
  const int64_t packing_factor = geom.packing_factor;
  const int64_t num_tiles_0 = geom.num_tiles_0;
  const int64_t num_tiles_1 = geom.num_tiles_1;
  const int64_t tile_size_bytes = geom.tile_size_bytes;
  const int64_t batch_size = geom.batch_size;
  const int64_t matrix_size_bytes = geom.matrix_size_bytes;
  const int64_t tiled_matrix_size_bytes = geom.tiled_matrix_size_bytes;

  const bool has_padding = geom.has_padding;

  // Fast-path: When the tiled layout is byte-for-byte identical to the linear
  // layout (e.g. single column of tiles W == tile_W with no vertical padding
  // H % tile_H == 0, or 1D tensor with tile_H == 1 and no padding) AND
  // packing_factor == 1.
  if (packing_factor == 1 && !has_padding &&
      (W == tile_W || (H == 1 && tile_H == 1))) {
    std::memcpy(dst_linear, src_tiled, batch_size * matrix_size_bytes);
    return absl::OkStatus();
  }

  // Fast-path for 1D tensors (H == 1) with packing_factor == 1.
  if (packing_factor == 1 && H == 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      const uint8_t* src_batch_ptr = src_tiled + b * tiled_matrix_size_bytes;
      uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        int64_t logical_col_start = tile_col * tile_W;
        int64_t valid_elements = std::min(tile_W, W - logical_col_start);
        if (valid_elements <= 0) {
          break;
        }
        std::memcpy(dst_batch_ptr + logical_col_start * itemsize,
                    src_batch_ptr + tile_col * tile_size_bytes,
                    valid_elements * itemsize);
      }
    }
    return absl::OkStatus();
  }

  int64_t total_tasks = batch_size * num_tiles_0;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t desired_chunks =
      std::max<int64_t>(1, total_bytes / kParallelizationThresholdBytes);
  int64_t row_bytes = (packing_factor == 1) ? (tile_W * itemsize) : 0;

  DispatchByPackingFactor(packing_factor, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    if (desired_chunks <= 1 || total_tasks <= 1) {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        if (!has_padding) {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr =
                src_tiled + b * tiled_matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                const uint8_t* src_tile_ptr =
                    src_batch_ptr + tile_index * tile_size_bytes;
                DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              }
            }
          }
        } else {
          for (int64_t b = 0; b < batch_size; ++b) {
            const uint8_t* src_batch_ptr =
                src_tiled + b * tiled_matrix_size_bytes;
            uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
            for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
              bool is_row_interior = (tile_row * tile_H + tile_H <= H);
              for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
                int64_t tile_index = tile_row * num_tiles_1 + tile_col;
                const uint8_t* src_tile_ptr =
                    src_batch_ptr + tile_index * tile_size_bytes;
                bool is_col_interior = (tile_col * tile_W + tile_W <= W);
                if (is_row_interior && is_col_interior) {
                  DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                      src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                      tile_W, W, itemsize);
                } else {
                  DetileSingleTilePackedWithPadding<kPackingFactor>(
                      src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                      tile_W, H, W, itemsize);
                }
              }
            }
          }
        }
      });
    } else {
      DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
        constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
        auto run_detile_task = [&](int64_t b, int64_t tile_row) {
          const uint8_t* src_batch_ptr =
              src_tiled + b * tiled_matrix_size_bytes;
          uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;

          if (!has_padding) {
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              const uint8_t* src_tile_ptr =
                  src_batch_ptr + tile_index * tile_size_bytes;
              DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                  src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                  tile_W, W, itemsize);
            }
          } else {
            bool is_row_interior = (tile_row * tile_H + tile_H <= H);
            for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
              int64_t tile_index = tile_row * num_tiles_1 + tile_col;
              const uint8_t* src_tile_ptr =
                  src_batch_ptr + tile_index * tile_size_bytes;
              bool is_col_interior = (tile_col * tile_W + tile_W <= W);
              if (is_row_interior && is_col_interior) {
                DetileSingleTilePackedNoPadding<kPackingFactor, kRowBytes>(
                    src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                    tile_W, W, itemsize);
              } else {
                DetileSingleTilePackedWithPadding<kPackingFactor>(
                    src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H,
                    tile_W, H, W, itemsize);
              }
            }
          }
        };

        ExecuteParallelTasks(total_tasks, num_tiles_0, desired_chunks,
                             numa_node, run_detile_task);
      });
    }
  });

  return absl::OkStatus();
}

// Column-major tile copy helpers:
// In column-major layouts (minor_to_major = {R-2, R-1, ...}), H = D1 (the
// stride-1 dimension in src_linear/dst_linear) and W = D0 (the major dimension
// with stride H in src_linear/dst_linear). Because sub-tile packing (P, 1)
// groups P consecutive elements along H, each packed unit of (P * itemsize)
// bytes is already contiguous in both src_linear and dst_tiled.
template <typename Word>
inline Word LoadUnaligned(const uint8_t* ptr) {
  Word w;
  std::memcpy(&w, ptr, sizeof(Word));
  return w;
}

template <typename Word>
inline void StoreUnaligned(uint8_t* ptr, Word w) {
  std::memcpy(ptr, &w, sizeof(Word));
}

#if HWY_TARGET != HWY_SCALAR
// Transposes a 4x4 matrix of uint32_t (4 rows of 16 bytes -> 4 cols of 16
// bytes) using unaligned uint8_t* loads and stores.
inline void Transpose4x4U32(const uint8_t* r0_ptr, const uint8_t* r1_ptr,
                            const uint8_t* r2_ptr, const uint8_t* r3_ptr,
                            uint8_t* c0_ptr, uint8_t* c1_ptr, uint8_t* c2_ptr,
                            uint8_t* c3_ptr) {
  const hn::FixedTag<uint8_t, 16> du8;
  const hn::FixedTag<uint32_t, 4> du32;
  const hn::FixedTag<uint64_t, 2> du64;

  const auto r0 = hn::BitCast(du32, hn::LoadU(du8, r0_ptr));
  const auto r1 = hn::BitCast(du32, hn::LoadU(du8, r1_ptr));
  const auto r2 = hn::BitCast(du32, hn::LoadU(du8, r2_ptr));
  const auto r3 = hn::BitCast(du32, hn::LoadU(du8, r3_ptr));

  const auto t0 = hn::BitCast(du64, hn::InterleaveLower(du32, r0, r1));
  const auto t1 = hn::BitCast(du64, hn::InterleaveUpper(du32, r0, r1));
  const auto t2 = hn::BitCast(du64, hn::InterleaveLower(du32, r2, r3));
  const auto t3 = hn::BitCast(du64, hn::InterleaveUpper(du32, r2, r3));

  hn::StoreU(hn::BitCast(du8, hn::InterleaveLower(du64, t0, t2)), du8, c0_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveUpper(du64, t0, t2)), du8, c1_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveLower(du64, t1, t3)), du8, c2_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveUpper(du64, t1, t3)), du8, c3_ptr);
}

// Transposes an 8x8 matrix of uint16_t (8 rows of 16 bytes -> 8 cols of 16
// bytes) using unaligned uint8_t* loads and stores.
inline void Transpose8x8U16(const uint8_t* r0_ptr, const uint8_t* r1_ptr,
                            const uint8_t* r2_ptr, const uint8_t* r3_ptr,
                            const uint8_t* r4_ptr, const uint8_t* r5_ptr,
                            const uint8_t* r6_ptr, const uint8_t* r7_ptr,
                            uint8_t* c0_ptr, uint8_t* c1_ptr, uint8_t* c2_ptr,
                            uint8_t* c3_ptr, uint8_t* c4_ptr, uint8_t* c5_ptr,
                            uint8_t* c6_ptr, uint8_t* c7_ptr) {
  const hn::FixedTag<uint8_t, 16> du8;
  const hn::FixedTag<uint16_t, 8> du16;
  const hn::FixedTag<uint32_t, 4> du32;
  const hn::FixedTag<uint64_t, 2> du64;

  const auto r0 = hn::BitCast(du16, hn::LoadU(du8, r0_ptr));
  const auto r1 = hn::BitCast(du16, hn::LoadU(du8, r1_ptr));
  const auto r2 = hn::BitCast(du16, hn::LoadU(du8, r2_ptr));
  const auto r3 = hn::BitCast(du16, hn::LoadU(du8, r3_ptr));
  const auto r4 = hn::BitCast(du16, hn::LoadU(du8, r4_ptr));
  const auto r5 = hn::BitCast(du16, hn::LoadU(du8, r5_ptr));
  const auto r6 = hn::BitCast(du16, hn::LoadU(du8, r6_ptr));
  const auto r7 = hn::BitCast(du16, hn::LoadU(du8, r7_ptr));

  const auto s0 = hn::BitCast(du32, hn::InterleaveLower(du16, r0, r1));
  const auto s1 = hn::BitCast(du32, hn::InterleaveUpper(du16, r0, r1));
  const auto s2 = hn::BitCast(du32, hn::InterleaveLower(du16, r2, r3));
  const auto s3 = hn::BitCast(du32, hn::InterleaveUpper(du16, r2, r3));
  const auto s4 = hn::BitCast(du32, hn::InterleaveLower(du16, r4, r5));
  const auto s5 = hn::BitCast(du32, hn::InterleaveUpper(du16, r4, r5));
  const auto s6 = hn::BitCast(du32, hn::InterleaveLower(du16, r6, r7));
  const auto s7 = hn::BitCast(du32, hn::InterleaveUpper(du16, r6, r7));

  const auto t0 = hn::BitCast(du64, hn::InterleaveLower(du32, s0, s2));
  const auto t1 = hn::BitCast(du64, hn::InterleaveUpper(du32, s0, s2));
  const auto t2 = hn::BitCast(du64, hn::InterleaveLower(du32, s1, s3));
  const auto t3 = hn::BitCast(du64, hn::InterleaveUpper(du32, s1, s3));
  const auto t4 = hn::BitCast(du64, hn::InterleaveLower(du32, s4, s6));
  const auto t5 = hn::BitCast(du64, hn::InterleaveUpper(du32, s4, s6));
  const auto t6 = hn::BitCast(du64, hn::InterleaveLower(du32, s5, s7));
  const auto t7 = hn::BitCast(du64, hn::InterleaveUpper(du32, s5, s7));

  hn::StoreU(hn::BitCast(du8, hn::InterleaveLower(du64, t0, t4)), du8, c0_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveUpper(du64, t0, t4)), du8, c1_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveLower(du64, t1, t5)), du8, c2_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveUpper(du64, t1, t5)), du8, c3_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveLower(du64, t2, t6)), du8, c4_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveUpper(du64, t2, t6)), du8, c5_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveLower(du64, t3, t7)), du8, c6_ptr);
  hn::StoreU(hn::BitCast(du8, hn::InterleaveUpper(du64, t3, t7)), du8, c7_ptr);
}
#endif  // HWY_TARGET != HWY_SCALAR

template <typename Word>
void CopyTileColMajorUnitsNoPadding(const uint8_t* src_batch_ptr,
                                    uint8_t* dst_tile_ptr, int64_t tile_row,
                                    int64_t tile_col, int64_t tile_H,
                                    int64_t tile_W, int64_t H, int64_t itemsize,
                                    int64_t num_groups) {
  constexpr size_t kWordBytes = sizeof(Word);
  const int64_t logical_row_start = tile_row * tile_H;
  const int64_t logical_col_start = tile_col * tile_W;
  const int64_t src_stride_bytes = H * itemsize;
  const uint8_t* src_base =
      src_batch_ptr + (logical_col_start * H + logical_row_start) * itemsize;
  const int64_t dst_group_stride = tile_W * kWordBytes;

  if (num_groups == 4) {
    uint8_t* dst0 = dst_tile_ptr + 0 * dst_group_stride;
    uint8_t* dst1 = dst_tile_ptr + 1 * dst_group_stride;
    uint8_t* dst2 = dst_tile_ptr + 2 * dst_group_stride;
    uint8_t* dst3 = dst_tile_ptr + 3 * dst_group_stride;
    int64_t c = 0;
#if HWY_TARGET != HWY_SCALAR
    if constexpr (kWordBytes == 4) {
      for (; c + 4 <= tile_W; c += 4) {
        const uint8_t* s = src_base + c * src_stride_bytes;
        const int64_t d_off = c * 4;
        Transpose4x4U32(s + 0 * src_stride_bytes, s + 1 * src_stride_bytes,
                        s + 2 * src_stride_bytes, s + 3 * src_stride_bytes,
                        dst0 + d_off, dst1 + d_off, dst2 + d_off, dst3 + d_off);
      }
    }
#endif
    for (; c < tile_W; ++c) {
      const uint8_t* src_col = src_base + c * src_stride_bytes;
      const int64_t d_off = c * kWordBytes;
      StoreUnaligned<Word>(dst0 + d_off,
                           LoadUnaligned<Word>(src_col + 0 * kWordBytes));
      StoreUnaligned<Word>(dst1 + d_off,
                           LoadUnaligned<Word>(src_col + 1 * kWordBytes));
      StoreUnaligned<Word>(dst2 + d_off,
                           LoadUnaligned<Word>(src_col + 2 * kWordBytes));
      StoreUnaligned<Word>(dst3 + d_off,
                           LoadUnaligned<Word>(src_col + 3 * kWordBytes));
    }
  } else if (num_groups == 8) {
    uint8_t* dst0 = dst_tile_ptr + 0 * dst_group_stride;
    uint8_t* dst1 = dst_tile_ptr + 1 * dst_group_stride;
    uint8_t* dst2 = dst_tile_ptr + 2 * dst_group_stride;
    uint8_t* dst3 = dst_tile_ptr + 3 * dst_group_stride;
    uint8_t* dst4 = dst_tile_ptr + 4 * dst_group_stride;
    uint8_t* dst5 = dst_tile_ptr + 5 * dst_group_stride;
    uint8_t* dst6 = dst_tile_ptr + 6 * dst_group_stride;
    uint8_t* dst7 = dst_tile_ptr + 7 * dst_group_stride;
    int64_t c = 0;
#if HWY_TARGET != HWY_SCALAR
    if constexpr (kWordBytes == 2) {
      for (; c + 8 <= tile_W; c += 8) {
        const uint8_t* s = src_base + c * src_stride_bytes;
        const int64_t d_off = c * 2;
        Transpose8x8U16(s + 0 * src_stride_bytes, s + 1 * src_stride_bytes,
                        s + 2 * src_stride_bytes, s + 3 * src_stride_bytes,
                        s + 4 * src_stride_bytes, s + 5 * src_stride_bytes,
                        s + 6 * src_stride_bytes, s + 7 * src_stride_bytes,
                        dst0 + d_off, dst1 + d_off, dst2 + d_off, dst3 + d_off,
                        dst4 + d_off, dst5 + d_off, dst6 + d_off, dst7 + d_off);
      }
    }
#endif
    for (; c < tile_W; ++c) {
      const uint8_t* src_col = src_base + c * src_stride_bytes;
      const int64_t d_off = c * kWordBytes;
      StoreUnaligned<Word>(dst0 + d_off,
                           LoadUnaligned<Word>(src_col + 0 * kWordBytes));
      StoreUnaligned<Word>(dst1 + d_off,
                           LoadUnaligned<Word>(src_col + 1 * kWordBytes));
      StoreUnaligned<Word>(dst2 + d_off,
                           LoadUnaligned<Word>(src_col + 2 * kWordBytes));
      StoreUnaligned<Word>(dst3 + d_off,
                           LoadUnaligned<Word>(src_col + 3 * kWordBytes));
      StoreUnaligned<Word>(dst4 + d_off,
                           LoadUnaligned<Word>(src_col + 4 * kWordBytes));
      StoreUnaligned<Word>(dst5 + d_off,
                           LoadUnaligned<Word>(src_col + 5 * kWordBytes));
      StoreUnaligned<Word>(dst6 + d_off,
                           LoadUnaligned<Word>(src_col + 6 * kWordBytes));
      StoreUnaligned<Word>(dst7 + d_off,
                           LoadUnaligned<Word>(src_col + 7 * kWordBytes));
    }
  } else if (num_groups == 2) {
    uint8_t* dst0 = dst_tile_ptr + 0 * dst_group_stride;
    uint8_t* dst1 = dst_tile_ptr + 1 * dst_group_stride;
    for (int64_t c = 0; c < tile_W; ++c) {
      const uint8_t* src_col = src_base + c * src_stride_bytes;
      const int64_t d_off = c * kWordBytes;
      StoreUnaligned<Word>(dst0 + d_off,
                           LoadUnaligned<Word>(src_col + 0 * kWordBytes));
      StoreUnaligned<Word>(dst1 + d_off,
                           LoadUnaligned<Word>(src_col + 1 * kWordBytes));
    }
  } else {
    for (int64_t c = 0; c < tile_W; ++c) {
      const uint8_t* src_col = src_base + c * src_stride_bytes;
      for (int64_t g = 0; g < num_groups; ++g) {
        StoreUnaligned<Word>(dst_tile_ptr + (g * tile_W + c) * kWordBytes,
                             LoadUnaligned<Word>(src_col + g * kWordBytes));
      }
    }
  }
}

void CopyTileColMajorNoPadding(const uint8_t* src_batch_ptr,
                               uint8_t* dst_tile_ptr, int64_t tile_row,
                               int64_t tile_col, int64_t tile_H, int64_t tile_W,
                               int64_t H, int64_t itemsize,
                               int64_t packing_factor) {
  const int64_t num_groups = tile_H / packing_factor;
  const int64_t unit_bytes = packing_factor * itemsize;
  switch (unit_bytes) {
    case 4:
      CopyTileColMajorUnitsNoPadding<uint32_t>(src_batch_ptr, dst_tile_ptr,
                                               tile_row, tile_col, tile_H,
                                               tile_W, H, itemsize, num_groups);
      break;
    case 2:
      CopyTileColMajorUnitsNoPadding<uint16_t>(src_batch_ptr, dst_tile_ptr,
                                               tile_row, tile_col, tile_H,
                                               tile_W, H, itemsize, num_groups);
      break;
    case 1:
      CopyTileColMajorUnitsNoPadding<uint8_t>(src_batch_ptr, dst_tile_ptr,
                                              tile_row, tile_col, tile_H,
                                              tile_W, H, itemsize, num_groups);
      break;
    case 8:
      CopyTileColMajorUnitsNoPadding<uint64_t>(src_batch_ptr, dst_tile_ptr,
                                               tile_row, tile_col, tile_H,
                                               tile_W, H, itemsize, num_groups);
      break;
    default: {
      const int64_t logical_row_start = tile_row * tile_H;
      const int64_t logical_col_start = tile_col * tile_W;
      const int64_t src_stride_bytes = H * itemsize;
      const uint8_t* src_base =
          src_batch_ptr +
          (logical_col_start * H + logical_row_start) * itemsize;
      for (int64_t c = 0; c < tile_W; ++c) {
        const uint8_t* src_col = src_base + c * src_stride_bytes;
        for (int64_t g = 0; g < num_groups; ++g) {
          std::memcpy(dst_tile_ptr + (g * tile_W + c) * unit_bytes,
                      src_col + g * unit_bytes, unit_bytes);
        }
      }
      break;
    }
  }
}

void CopyTileColMajorWithPadding(const uint8_t* src_batch_ptr,
                                 uint8_t* dst_tile_ptr, int64_t tile_row,
                                 int64_t tile_col, int64_t tile_H,
                                 int64_t tile_W, int64_t H, int64_t W,
                                 int64_t itemsize, int64_t packing_factor,
                                 int64_t tile_size_bytes) {
  ZeroRowHighway(dst_tile_ptr, static_cast<size_t>(tile_size_bytes));
  const int64_t logical_row_start = tile_row * tile_H;
  const int64_t logical_col_start = tile_col * tile_W;
  const int64_t valid_H = std::clamp<int64_t>(H - logical_row_start, 0, tile_H);
  const int64_t valid_W = std::clamp<int64_t>(W - logical_col_start, 0, tile_W);
  if (valid_H <= 0 || valid_W <= 0) {
    return;
  }

  for (int64_t c = 0; c < valid_W; ++c) {
    const uint8_t* src_col =
        src_batch_ptr +
        ((logical_col_start + c) * H + logical_row_start) * itemsize;
    for (int64_t r = 0; r < valid_H; ++r) {
      int64_t g = r / packing_factor;
      int64_t p = r % packing_factor;
      int64_t dst_elem_idx = (g * tile_W + c) * packing_factor + p;
      std::memcpy(dst_tile_ptr + dst_elem_idx * itemsize,
                  src_col + r * itemsize, itemsize);
    }
  }
}

template <typename Word>
void DetileSingleTileColMajorUnitsNoPadding(const uint8_t* src_tile_ptr,
                                            uint8_t* dst_batch_ptr,
                                            int64_t tile_row, int64_t tile_col,
                                            int64_t tile_H, int64_t tile_W,
                                            int64_t H, int64_t itemsize,
                                            int64_t num_groups) {
  constexpr size_t kWordBytes = sizeof(Word);
  const int64_t logical_row_start = tile_row * tile_H;
  const int64_t logical_col_start = tile_col * tile_W;
  const int64_t dst_stride_bytes = H * itemsize;
  uint8_t* dst_base =
      dst_batch_ptr + (logical_col_start * H + logical_row_start) * itemsize;
  const int64_t src_group_stride = tile_W * kWordBytes;

  if (num_groups == 4) {
    const uint8_t* src0 = src_tile_ptr + 0 * src_group_stride;
    const uint8_t* src1 = src_tile_ptr + 1 * src_group_stride;
    const uint8_t* src2 = src_tile_ptr + 2 * src_group_stride;
    const uint8_t* src3 = src_tile_ptr + 3 * src_group_stride;
    int64_t c = 0;
#if HWY_TARGET != HWY_SCALAR
    if constexpr (kWordBytes == 4) {
      for (; c + 4 <= tile_W; c += 4) {
        const int64_t s_off = c * 4;
        uint8_t* d = dst_base + c * dst_stride_bytes;
        Transpose4x4U32(src0 + s_off, src1 + s_off, src2 + s_off, src3 + s_off,
                        d + 0 * dst_stride_bytes, d + 1 * dst_stride_bytes,
                        d + 2 * dst_stride_bytes, d + 3 * dst_stride_bytes);
      }
    }
#endif
    for (; c < tile_W; ++c) {
      uint8_t* dst_col = dst_base + c * dst_stride_bytes;
      const int64_t s_off = c * kWordBytes;
      StoreUnaligned<Word>(dst_col + 0 * kWordBytes,
                           LoadUnaligned<Word>(src0 + s_off));
      StoreUnaligned<Word>(dst_col + 1 * kWordBytes,
                           LoadUnaligned<Word>(src1 + s_off));
      StoreUnaligned<Word>(dst_col + 2 * kWordBytes,
                           LoadUnaligned<Word>(src2 + s_off));
      StoreUnaligned<Word>(dst_col + 3 * kWordBytes,
                           LoadUnaligned<Word>(src3 + s_off));
    }
  } else if (num_groups == 8) {
    const uint8_t* src0 = src_tile_ptr + 0 * src_group_stride;
    const uint8_t* src1 = src_tile_ptr + 1 * src_group_stride;
    const uint8_t* src2 = src_tile_ptr + 2 * src_group_stride;
    const uint8_t* src3 = src_tile_ptr + 3 * src_group_stride;
    const uint8_t* src4 = src_tile_ptr + 4 * src_group_stride;
    const uint8_t* src5 = src_tile_ptr + 5 * src_group_stride;
    const uint8_t* src6 = src_tile_ptr + 6 * src_group_stride;
    const uint8_t* src7 = src_tile_ptr + 7 * src_group_stride;
    int64_t c = 0;
#if HWY_TARGET != HWY_SCALAR
    if constexpr (kWordBytes == 2) {
      for (; c + 8 <= tile_W; c += 8) {
        const int64_t s_off = c * 2;
        uint8_t* d = dst_base + c * dst_stride_bytes;
        Transpose8x8U16(src0 + s_off, src1 + s_off, src2 + s_off, src3 + s_off,
                        src4 + s_off, src5 + s_off, src6 + s_off, src7 + s_off,
                        d + 0 * dst_stride_bytes, d + 1 * dst_stride_bytes,
                        d + 2 * dst_stride_bytes, d + 3 * dst_stride_bytes,
                        d + 4 * dst_stride_bytes, d + 5 * dst_stride_bytes,
                        d + 6 * dst_stride_bytes, d + 7 * dst_stride_bytes);
      }
    }
#endif
    for (; c < tile_W; ++c) {
      uint8_t* dst_col = dst_base + c * dst_stride_bytes;
      const int64_t s_off = c * kWordBytes;
      StoreUnaligned<Word>(dst_col + 0 * kWordBytes,
                           LoadUnaligned<Word>(src0 + s_off));
      StoreUnaligned<Word>(dst_col + 1 * kWordBytes,
                           LoadUnaligned<Word>(src1 + s_off));
      StoreUnaligned<Word>(dst_col + 2 * kWordBytes,
                           LoadUnaligned<Word>(src2 + s_off));
      StoreUnaligned<Word>(dst_col + 3 * kWordBytes,
                           LoadUnaligned<Word>(src3 + s_off));
      StoreUnaligned<Word>(dst_col + 4 * kWordBytes,
                           LoadUnaligned<Word>(src4 + s_off));
      StoreUnaligned<Word>(dst_col + 5 * kWordBytes,
                           LoadUnaligned<Word>(src5 + s_off));
      StoreUnaligned<Word>(dst_col + 6 * kWordBytes,
                           LoadUnaligned<Word>(src6 + s_off));
      StoreUnaligned<Word>(dst_col + 7 * kWordBytes,
                           LoadUnaligned<Word>(src7 + s_off));
    }
  } else if (num_groups == 2) {
    const uint8_t* src0 = src_tile_ptr + 0 * src_group_stride;
    const uint8_t* src1 = src_tile_ptr + 1 * src_group_stride;
    for (int64_t c = 0; c < tile_W; ++c) {
      uint8_t* dst_col = dst_base + c * dst_stride_bytes;
      const int64_t s_off = c * kWordBytes;
      StoreUnaligned<Word>(dst_col + 0 * kWordBytes,
                           LoadUnaligned<Word>(src0 + s_off));
      StoreUnaligned<Word>(dst_col + 1 * kWordBytes,
                           LoadUnaligned<Word>(src1 + s_off));
    }
  } else {
    for (int64_t c = 0; c < tile_W; ++c) {
      uint8_t* dst_col = dst_base + c * dst_stride_bytes;
      for (int64_t g = 0; g < num_groups; ++g) {
        StoreUnaligned<Word>(
            dst_col + g * kWordBytes,
            LoadUnaligned<Word>(src_tile_ptr + (g * tile_W + c) * kWordBytes));
      }
    }
  }
}

void DetileSingleTileColMajorNoPadding(const uint8_t* src_tile_ptr,
                                       uint8_t* dst_batch_ptr, int64_t tile_row,
                                       int64_t tile_col, int64_t tile_H,
                                       int64_t tile_W, int64_t H,
                                       int64_t itemsize,
                                       int64_t packing_factor) {
  const int64_t num_groups = tile_H / packing_factor;
  const int64_t unit_bytes = packing_factor * itemsize;
  switch (unit_bytes) {
    case 4:
      DetileSingleTileColMajorUnitsNoPadding<uint32_t>(
          src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H, tile_W, H,
          itemsize, num_groups);
      break;
    case 2:
      DetileSingleTileColMajorUnitsNoPadding<uint16_t>(
          src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H, tile_W, H,
          itemsize, num_groups);
      break;
    case 1:
      DetileSingleTileColMajorUnitsNoPadding<uint8_t>(
          src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H, tile_W, H,
          itemsize, num_groups);
      break;
    case 8:
      DetileSingleTileColMajorUnitsNoPadding<uint64_t>(
          src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H, tile_W, H,
          itemsize, num_groups);
      break;
    default: {
      const int64_t logical_row_start = tile_row * tile_H;
      const int64_t logical_col_start = tile_col * tile_W;
      const int64_t dst_stride_bytes = H * itemsize;
      uint8_t* dst_base =
          dst_batch_ptr +
          (logical_col_start * H + logical_row_start) * itemsize;
      for (int64_t c = 0; c < tile_W; ++c) {
        uint8_t* dst_col = dst_base + c * dst_stride_bytes;
        for (int64_t g = 0; g < num_groups; ++g) {
          std::memcpy(dst_col + g * unit_bytes,
                      src_tile_ptr + (g * tile_W + c) * unit_bytes, unit_bytes);
        }
      }
      break;
    }
  }
}

void DetileSingleTileColMajorWithPadding(const uint8_t* src_tile_ptr,
                                         uint8_t* dst_batch_ptr,
                                         int64_t tile_row, int64_t tile_col,
                                         int64_t tile_H, int64_t tile_W,
                                         int64_t H, int64_t W, int64_t itemsize,
                                         int64_t packing_factor) {
  const int64_t logical_row_start = tile_row * tile_H;
  const int64_t logical_col_start = tile_col * tile_W;
  const int64_t valid_H = std::clamp<int64_t>(H - logical_row_start, 0, tile_H);
  const int64_t valid_W = std::clamp<int64_t>(W - logical_col_start, 0, tile_W);
  if (valid_H <= 0 || valid_W <= 0) {
    return;
  }

  for (int64_t c = 0; c < valid_W; ++c) {
    uint8_t* dst_col =
        dst_batch_ptr +
        ((logical_col_start + c) * H + logical_row_start) * itemsize;
    for (int64_t r = 0; r < valid_H; ++r) {
      int64_t g = r / packing_factor;
      int64_t p = r % packing_factor;
      int64_t src_elem_idx = (g * tile_W + c) * packing_factor + p;
      std::memcpy(dst_col + r * itemsize,
                  src_tile_ptr + src_elem_idx * itemsize, itemsize);
    }
  }
}

// Tiles a buffer using an optimized path for standard column-major layouts.
// Iterates tile_col in the outer loop and tile_row in the inner loop so that
// the tile_W rows of src_linear stay hot in L1d cache (128 * 64B = 8 KB) as
// tile_row advances sequentially along the stride-1 dimension H = D1.
absl::Status TileBufferColMajorOptimized(
    const uint8_t* src_linear, uint8_t* dst_tiled, const xla::Shape& shape,
    const xla::Layout& layout, std::optional<int> numa_node = std::nullopt) {
  const int R = shape.dimensions().size();
  int64_t H = shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);
  int64_t packing_factor = 1;
  if (layout.tiles().size() >= 2) {
    packing_factor = layout.tiles(1).dimension(0);
  }

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;
  bool has_padding = (H % tile_H != 0) || (W % tile_W != 0);

  int64_t total_tasks = batch_size * num_tiles_1;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t desired_chunks =
      std::max<int64_t>(1, total_bytes / kParallelizationThresholdBytes);

  auto process_tile_col = [&](int64_t b, int64_t tile_col) {
    const uint8_t* src_batch_ptr = src_linear + b * matrix_size_bytes;
    uint8_t* dst_batch_ptr = dst_tiled + b * tiled_matrix_size_bytes;
    bool is_col_interior = (tile_col * tile_W + tile_W <= W);

    if (!has_padding) {
      for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
        int64_t tile_index = tile_row * num_tiles_1 + tile_col;
        uint8_t* dst_tile_ptr = dst_batch_ptr + tile_index * tile_size_bytes;
        CopyTileColMajorNoPadding(src_batch_ptr, dst_tile_ptr, tile_row,
                                  tile_col, tile_H, tile_W, H, itemsize,
                                  packing_factor);
      }
    } else {
      for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
        int64_t tile_index = tile_row * num_tiles_1 + tile_col;
        uint8_t* dst_tile_ptr = dst_batch_ptr + tile_index * tile_size_bytes;
        bool is_row_interior = (tile_row * tile_H + tile_H <= H);
        if (is_row_interior && is_col_interior) {
          CopyTileColMajorNoPadding(src_batch_ptr, dst_tile_ptr, tile_row,
                                    tile_col, tile_H, tile_W, H, itemsize,
                                    packing_factor);
        } else {
          CopyTileColMajorWithPadding(src_batch_ptr, dst_tile_ptr, tile_row,
                                      tile_col, tile_H, tile_W, H, W, itemsize,
                                      packing_factor, tile_size_bytes);
        }
      }
    }
  };

  if (desired_chunks <= 1 || total_tasks <= 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        process_tile_col(b, tile_col);
      }
    }
  } else {
    ExecuteParallelTasks(total_tasks, num_tiles_1, desired_chunks, numa_node,
                         process_tile_col);
  }

  return absl::OkStatus();
}

absl::Status DetileBufferColMajorOptimized(
    const uint8_t* src_tiled, uint8_t* dst_linear, const xla::Shape& shape,
    const xla::Layout& layout, std::optional<int> numa_node = std::nullopt) {
  const int R = shape.dimensions().size();
  int64_t H = shape.dimensions(layout.minor_to_major(1));
  int64_t W = shape.dimensions(layout.minor_to_major(0));
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  const xla::Tile& tile = layout.tiles(0);
  int64_t tile_H = tile.dimension(0);
  int64_t tile_W = tile.dimension(1);
  int64_t packing_factor = 1;
  if (layout.tiles().size() >= 2) {
    packing_factor = layout.tiles(1).dimension(0);
  }

  int64_t num_tiles_0 = xla::CeilOfRatio(H, tile_H);
  int64_t num_tiles_1 = xla::CeilOfRatio(W, tile_W);
  int64_t tile_size_bytes = tile_H * tile_W * itemsize;

  int64_t batch_size = 1;
  for (int i = 2; i < R; ++i) {
    batch_size *= shape.dimensions(layout.minor_to_major(i));
  }

  int64_t matrix_size_bytes = H * W * itemsize;
  int64_t tiled_matrix_size_bytes = num_tiles_0 * num_tiles_1 * tile_size_bytes;
  bool has_padding = (H % tile_H != 0) || (W % tile_W != 0);

  int64_t total_tasks = batch_size * num_tiles_1;
  int64_t total_bytes = batch_size * H * W * itemsize;
  int64_t desired_chunks =
      std::max<int64_t>(1, total_bytes / kParallelizationThresholdBytes);

  auto process_tile_col = [&](int64_t b, int64_t tile_col) {
    const uint8_t* src_batch_ptr = src_tiled + b * tiled_matrix_size_bytes;
    uint8_t* dst_batch_ptr = dst_linear + b * matrix_size_bytes;
    bool is_col_interior = (tile_col * tile_W + tile_W <= W);

    if (!has_padding) {
      for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
        int64_t tile_index = tile_row * num_tiles_1 + tile_col;
        const uint8_t* src_tile_ptr =
            src_batch_ptr + tile_index * tile_size_bytes;
        DetileSingleTileColMajorNoPadding(src_tile_ptr, dst_batch_ptr, tile_row,
                                          tile_col, tile_H, tile_W, H, itemsize,
                                          packing_factor);
      }
    } else {
      for (int64_t tile_row = 0; tile_row < num_tiles_0; ++tile_row) {
        int64_t tile_index = tile_row * num_tiles_1 + tile_col;
        const uint8_t* src_tile_ptr =
            src_batch_ptr + tile_index * tile_size_bytes;
        bool is_row_interior = (tile_row * tile_H + tile_H <= H);
        if (is_row_interior && is_col_interior) {
          DetileSingleTileColMajorNoPadding(src_tile_ptr, dst_batch_ptr,
                                            tile_row, tile_col, tile_H, tile_W,
                                            H, itemsize, packing_factor);
        } else {
          DetileSingleTileColMajorWithPadding(
              src_tile_ptr, dst_batch_ptr, tile_row, tile_col, tile_H, tile_W,
              H, W, itemsize, packing_factor);
        }
      }
    }
  };

  if (desired_chunks <= 1 || total_tasks <= 1) {
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
        process_tile_col(b, tile_col);
      }
    }
  } else {
    ExecuteParallelTasks(total_tasks, num_tiles_1, desired_chunks, numa_node,
                         process_tile_col);
  }

  return absl::OkStatus();
}

// In-place tiling of a 1D tensor with an unpacked 1D tile ({0:T(T0)} or
// {0:T(T0)(T1)}). The tiled bytes are the linear bytes followed by zero
// padding up to a multiple of the outer tile, so only the tail is written.
absl::Status ZeroPadUnpacked1DInPlace(uint8_t* buffer, size_t buffer_capacity,
                                      int64_t W, int64_t itemsize,
                                      const xla::Layout& layout) {
  const int64_t outer_W = layout.tiles(0).dimension(0);
  const int64_t total_physical_elements =
      xla::CeilOfRatio(W, outer_W) * outer_W;
  if (buffer_capacity <
      static_cast<size_t>(total_physical_elements * itemsize)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Buffer capacity ", buffer_capacity,
                     " is smaller than required 1D tiled buffer size ",
                     total_physical_elements * itemsize));
  }
  if (total_physical_elements > W) {
    ZeroRowHighway(
        buffer + W * itemsize,
        static_cast<size_t>((total_physical_elements - W) * itemsize));
  }
  return absl::OkStatus();
}

// In-place tiling of a 1D tensor with a packed tile ({0:T(tile_W)(P,1)} or
// {0:T(outer_W)(tile_W)(P,1)}). Each packing group holds P * tile_W
// consecutive logical elements and is rewritten as tile_W lanes of P
// interleaved elements. A group's tiled bytes occupy exactly the group's own
// linear bytes, so each group is staged through a group-sized bounce buffer
// and rewritten independently; the trailing partial group is zero padded.
absl::Status TilePacked1DInPlace(uint8_t* buffer, size_t buffer_capacity,
                                 int64_t W, int64_t itemsize,
                                 const xla::Layout& layout) {
  const Packed1DTileGeometry geom =
      ComputePacked1DTileGeometry(W, itemsize, layout);
  const int64_t P = geom.P;
  const int64_t tile_W = geom.tile_W;
  const int64_t total_groups = geom.total_groups;
  const int64_t group_elems = geom.group_elems;
  const int64_t group_bytes = geom.group_bytes;
  if (buffer_capacity < static_cast<size_t>(total_groups * group_bytes)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Buffer capacity ", buffer_capacity,
                     " is smaller than required 1D packed tiled buffer size ",
                     total_groups * group_bytes));
  }

  std::vector<uint8_t> group_buf(group_bytes);
  DispatchByPackingFactor(P, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    for (int64_t g = total_groups - 1; g >= 0; --g) {
      uint8_t* dst_group_ptr = buffer + g * group_bytes;
      const int64_t group_start = g * group_elems;
      if (group_start >= W) {
        ZeroRowHighway(dst_group_ptr, static_cast<size_t>(group_bytes));
        continue;
      }
      int64_t valid_elements = std::min(group_elems, W - group_start);
      std::memcpy(group_buf.data(), dst_group_ptr, valid_elements * itemsize);
      if (valid_elements < group_elems) {
        std::memset(group_buf.data() + valid_elements * itemsize, 0,
                    (group_elems - valid_elements) * itemsize);
      }
      if (group_start + group_elems <= W) {
        CopyTilePackedNoPadding<kPackingFactor, 0>(
            group_buf.data(), dst_group_ptr, /*tile_row=*/0, /*tile_col=*/0,
            /*tile_H=*/kPackingFactor, tile_W, /*W=*/tile_W, itemsize);
      } else {
        ZeroRowHighway(dst_group_ptr, static_cast<size_t>(group_bytes));
        for (int64_t p = 0; p < kPackingFactor; ++p) {
          const int64_t row_start = p * tile_W;
          for (int64_t c = 0; c < tile_W; ++c) {
            const int64_t idx = row_start + c;
            if (idx < valid_elements) {
              std::memcpy(dst_group_ptr + (c * kPackingFactor + p) * itemsize,
                          group_buf.data() + idx * itemsize, itemsize);
            }
          }
        }
      }
    }
  });
  return absl::OkStatus();
}

// In-place tiling of a single logical row (H == 1) under a 2D tile with no
// packing. Each tile_W-element column slice becomes the first row of its tile,
// followed by (tile_H - 1) zero rows. Tiles are moved from last to first:
// tile_col's destination offset (tile_col * tile_size_bytes) is never less than
// its source offset (tile_col * tile_W * itemsize), so a reverse sweep never
// overwrites a slice that has not been moved yet.
void TileSingleRowInPlace(uint8_t* buffer, const RowMajorTileGeometry& geom) {
  const int64_t W = geom.W;
  const int64_t itemsize = geom.itemsize;
  const int64_t tile_H = geom.tile_H;
  const int64_t tile_W = geom.tile_W;
  for (int64_t b = geom.batch_size - 1; b >= 0; --b) {
    uint8_t* src_batch_ptr = buffer + b * geom.matrix_size_bytes;
    uint8_t* dst_batch_ptr = buffer + b * geom.tiled_matrix_size_bytes;
    for (int64_t tile_col = geom.num_tiles_1 - 1; tile_col >= 0; --tile_col) {
      int64_t logical_col_start = tile_col * tile_W;
      int64_t valid_elements = std::min(tile_W, W - logical_col_start);
      uint8_t* dst_tile_ptr = dst_batch_ptr + tile_col * geom.tile_size_bytes;
      const uint8_t* src_tile_ptr =
          src_batch_ptr + logical_col_start * itemsize;
      if (valid_elements > 0) {
        std::memmove(dst_tile_ptr, src_tile_ptr, valid_elements * itemsize);
        if (valid_elements < tile_W) {
          std::memset(dst_tile_ptr + valid_elements * itemsize, 0,
                      (tile_W - valid_elements) * itemsize);
        }
      } else {
        std::memset(dst_tile_ptr, 0, tile_W * itemsize);
      }
      if (tile_H > 1) {
        std::memset(dst_tile_ptr + tile_W * itemsize, 0,
                    (tile_H - 1) * tile_W * itemsize);
      }
    }
  }
}

// General in-place tiling of a row-major 2D/ND tensor. A band is tile_H
// consecutive logical rows; its tiles are exactly the tiles in one tile row.
// Each band is copied to an L2-resident bounce buffer and scattered back as
// tiles. Bands run from last to first (and batches likewise): the tiled offset
// of a band is >= its linear offset, so writing band k only touches bytes of
// bands >= k, which have already been consumed.
void TileRowMajorBandsInPlace(uint8_t* buffer,
                              const RowMajorTileGeometry& geom) {
  const int64_t H = geom.H;
  const int64_t W = geom.W;
  const int64_t itemsize = geom.itemsize;
  const int64_t tile_H = geom.tile_H;
  const int64_t tile_W = geom.tile_W;
  const int64_t num_tiles_1 = geom.num_tiles_1;
  const int64_t tile_size_bytes = geom.tile_size_bytes;
  const int64_t row_bytes =
      (geom.packing_factor == 1) ? (tile_W * itemsize) : 0;
  std::vector<uint8_t> band_buf(tile_H * W * itemsize);

  DispatchByPackingFactor(geom.packing_factor, [&](auto kPackingFactorTag) {
    constexpr int64_t kPackingFactor = decltype(kPackingFactorTag)::value;
    DispatchByRowBytes(row_bytes, [&](auto kRowBytesTag) {
      constexpr size_t kRowBytes = decltype(kRowBytesTag)::value;
      for (int64_t b = geom.batch_size - 1; b >= 0; --b) {
        uint8_t* src_batch_ptr = buffer + b * geom.matrix_size_bytes;
        uint8_t* dst_batch_ptr = buffer + b * geom.tiled_matrix_size_bytes;

        for (int64_t tile_row = geom.num_tiles_0 - 1; tile_row >= 0;
             --tile_row) {
          int64_t logical_row_start = tile_row * tile_H;
          int64_t valid_band_H = std::min(H - logical_row_start, tile_H);
          if (valid_band_H <= 0) continue;

          size_t linear_bytes_to_copy =
              static_cast<size_t>(valid_band_H * W * itemsize);
          const uint8_t* linear_band_src =
              src_batch_ptr + logical_row_start * W * itemsize;
          std::memcpy(band_buf.data(), linear_band_src, linear_bytes_to_copy);
          if (valid_band_H < tile_H) {
            std::memset(band_buf.data() + linear_bytes_to_copy, 0,
                        (tile_H - valid_band_H) * W * itemsize);
          }

          bool is_row_interior = (valid_band_H == tile_H);
          for (int64_t tile_col = 0; tile_col < num_tiles_1; ++tile_col) {
            int64_t tile_index = tile_row * num_tiles_1 + tile_col;
            uint8_t* dst_tile_ptr =
                dst_batch_ptr + tile_index * tile_size_bytes;
            bool is_col_interior = (tile_col * tile_W + tile_W <= W);
            if (is_row_interior && is_col_interior) {
              CopyTilePackedNoPadding<kPackingFactor, kRowBytes>(
                  band_buf.data(), dst_tile_ptr, /*tile_row=*/0, tile_col,
                  tile_H, tile_W, W, itemsize);
            } else {
              CopyTilePackedWithPadding<kPackingFactor>(
                  band_buf.data(), dst_tile_ptr, /*tile_row=*/0, tile_col,
                  tile_H, tile_W, valid_band_H, W, itemsize, tile_size_bytes);
            }
          }
        }
      }
    });
  });
}

// Dispatches a standard row-major tiled tensor to the matching in-place tiler.
absl::Status TileBufferInPlaceRowMajor(uint8_t* buffer, size_t buffer_capacity,
                                       const xla::Shape& shape,
                                       const xla::Layout& layout) {
  const int R = shape.dimensions().size();
  if (R == 1 && layout.tiles(0).dimensions().size() == 1) {
    const int64_t W = shape.dimensions(0);
    const int64_t itemsize =
        xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    const xla::Tile& last_tile = layout.tiles(layout.tiles().size() - 1);
    if (last_tile.dimensions().size() == 1) {
      return ZeroPadUnpacked1DInPlace(buffer, buffer_capacity, W, itemsize,
                                      layout);
    }
    return TilePacked1DInPlace(buffer, buffer_capacity, W, itemsize, layout);
  }

  const RowMajorTileGeometry geom = ComputeRowMajorTileGeometry(shape, layout);
  const int64_t total_tiled_bytes = geom.total_tiled_bytes();
  if (buffer_capacity < static_cast<size_t>(total_tiled_bytes)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Buffer capacity ", buffer_capacity,
        " is smaller than required tiled buffer size ", total_tiled_bytes));
  }

  // The tiled layout is byte-for-byte identical to the linear layout: a single
  // column of tiles with no padding, or a 1D tensor with tile_H == 1.
  if (geom.packing_factor == 1 && !geom.has_padding &&
      (geom.W == geom.tile_W || (geom.H == 1 && geom.tile_H == 1))) {
    return absl::OkStatus();
  }

  if (geom.packing_factor == 1 && geom.H == 1) {
    TileSingleRowInPlace(buffer, geom);
    return absl::OkStatus();
  }

  TileRowMajorBandsInPlace(buffer, geom);
  return absl::OkStatus();
}

}  // namespace

int64_t GetTiledBufferElements(const xla::Shape& shape) {
  const int num_dims = shape.dimensions().size();
  if (num_dims == 0) {
    return 1;
  }
  if (!shape.has_layout() ||
      shape.layout().minor_to_major().size() != num_dims) {
    return xla::ShapeUtil::ElementsIn(shape);
  }

  std::vector<int64_t> current_shape;
  current_shape.reserve(std::max(num_dims, 2));
  for (int64_t i = num_dims - 1; i >= 0; --i) {
    int64_t logical_dim = shape.layout().minor_to_major(i);
    current_shape.push_back(shape.dimensions(logical_dim));
  }

  for (const xla::Tile& tile : shape.layout().tiles()) {
    const int64_t tile_rank = tile.dimensions().size();
    if (tile_rank > current_shape.size()) {
      int64_t pad_size = tile_rank - current_shape.size();
      current_shape.insert(current_shape.begin(), pad_size, 1);
    }

    const int64_t suffix_start = current_shape.size() - tile_rank;
    std::vector<int64_t> next_shape;
    next_shape.reserve(current_shape.size() + tile_rank);

    for (int i = 0; i < suffix_start; ++i) {
      next_shape.push_back(current_shape[i]);
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t d = current_shape[suffix_start + i];
      int64_t t = tile.dimension(i);
      if (t <= 0) {
        return shape.dimensions().empty() ? 1
                                          : xla::ShapeUtil::ElementsIn(shape);
      }
      next_shape.push_back(xla::CeilOfRatio(d, t));
    }

    for (int i = 0; i < tile_rank; ++i) {
      int64_t t = tile.dimension(i);
      next_shape.push_back(t);
    }

    current_shape = std::move(next_shape);
  }

  int64_t total_elements = 1;
  for (int64_t dim_size : current_shape) {
    total_elements *= dim_size;
  }
  return total_elements;
}

absl::Status DetileBuffer(const uint8_t* src_tiled, uint8_t* dst_linear,
                          const xla::Shape& shape, const xla::Layout& layout,
                          std::optional<int> numa_node) {
  if (layout.tiles().empty()) {
    const int64_t bytes = xla::ShapeUtil::ByteSizeOf(shape);
    if (bytes > 0) {
      std::memcpy(dst_linear, src_tiled, bytes);
    }
    return absl::OkStatus();
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return DetileBufferNDOptimized(src_tiled, dst_linear, shape, layout,
                                   numa_node);
  }

  if (IsStandardColMajorTiled(shape, layout)) {
    return DetileBufferColMajorOptimized(src_tiled, dst_linear, shape, layout,
                                         numa_node);
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  xla::Shape tiled_shape = shape;
  *tiled_shape.mutable_layout() = layout;
  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      tiled_shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(tiled_shape, indices) *
            itemsize;
        std::memcpy(dst_linear + linear_offset, src_tiled + physical_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

absl::Status TileBuffer(const uint8_t* src_linear, uint8_t* dst_tiled,
                        const xla::Shape& shape, const xla::Layout& layout,
                        std::optional<int> numa_node) {
  if (src_linear == dst_tiled) {
    xla::Shape tiled_shape = shape;
    *tiled_shape.mutable_layout() = layout;
    int64_t itemsize =
        xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    int64_t total_physical_elements = GetTiledBufferElements(tiled_shape);
    return TileBufferInPlace(dst_tiled, total_physical_elements * itemsize,
                             shape, layout, numa_node);
  }

  if (layout.tiles().empty()) {
    const int64_t bytes = xla::ShapeUtil::ByteSizeOf(shape);
    if (bytes > 0) {
      std::memcpy(dst_tiled, src_linear, bytes);
    }
    return absl::OkStatus();
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return TileBufferNDOptimized(src_linear, dst_tiled, shape, layout,
                                 numa_node);
  }

  if (IsStandardColMajorTiled(shape, layout)) {
    return TileBufferColMajorOptimized(src_linear, dst_tiled, shape, layout,
                                       numa_node);
  }

  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());

  xla::Shape tiled_shape = shape;
  *tiled_shape.mutable_layout() = layout;
  int64_t total_physical_elements = GetTiledBufferElements(tiled_shape);
  int64_t logical_elements = 1;
  for (int64_t dim : shape.dimensions()) {
    logical_elements *= dim;
  }
  if (total_physical_elements > logical_elements) {
    std::memset(dst_tiled, 0, total_physical_elements * itemsize);
  }

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());

  xla::ShapeUtil::ForEachIndexNoStatus(
      tiled_shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_offset =
            xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                               indices) *
            itemsize;
        int64_t physical_offset =
            xla::LayoutUtil::LinearIndexForNestedTiling(tiled_shape, indices) *
            itemsize;
        std::memcpy(dst_tiled + physical_offset, src_linear + linear_offset,
                    itemsize);
        return true;
      });

  return absl::OkStatus();
}

absl::Status TileBufferInPlace(uint8_t* buffer, size_t buffer_capacity,
                               const xla::Shape& shape,
                               const xla::Layout& layout,
                               std::optional<int> numa_node) {
  if (layout.tiles().empty()) {
    return absl::OkStatus();
  }

  if (IsStandardRowMajorTiled(shape, layout)) {
    return TileBufferInPlaceRowMajor(buffer, buffer_capacity, shape, layout);
  }

  xla::Shape tiled_shape = shape;
  *tiled_shape.mutable_layout() = layout;
  int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  int64_t total_physical_elements = GetTiledBufferElements(tiled_shape);
  if (buffer_capacity <
      static_cast<size_t>(total_physical_elements * itemsize)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Buffer capacity ", buffer_capacity,
                     " is smaller than required tiled buffer size ",
                     total_physical_elements * itemsize));
  }

  // Fallback for non-row-major layouts: allocate scratchpad, tile, and copy.
  std::vector<uint8_t> tmp(total_physical_elements * itemsize);
  TF_RETURN_IF_ERROR(TileBuffer(buffer, tmp.data(), shape, layout, numa_node));
  std::memcpy(buffer, tmp.data(), tmp.size());
  return absl::OkStatus();
}

}  // namespace tpu_raiden::weight_sync
