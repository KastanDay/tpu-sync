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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "tpu_sync/core/numa_thread_pool.h"

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/index_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace tpu_raiden::weight_sync {
namespace {

enum class TilingMethod {
  kAllocatedMemory,
  kInPlace,
};

class TilingUtilsTest : public ::testing::TestWithParam<TilingMethod> {
 protected:
  absl::Status TileBuffer(const uint8_t* src_linear, uint8_t* dst_tiled,
                          const xla::Shape& shape, const xla::Layout& layout,
                          std::optional<int> numa_node = std::nullopt) const {
    if (GetParam() == TilingMethod::kAllocatedMemory) {
      return ::tpu_raiden::weight_sync::TileBuffer(src_linear, dst_tiled, shape,
                                                   layout, numa_node);
    }
    int64_t linear_bytes =
        xla::ShapeUtil::ElementsIn(shape) *
        xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    int64_t tiled_bytes =
        GetTiledBufferElements(shape) *
        xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    std::memcpy(dst_tiled, src_linear, linear_bytes);
    return ::tpu_raiden::weight_sync::TileBufferInPlace(
        dst_tiled, tiled_bytes, shape, layout, numa_node);
  }
};

INSTANTIATE_TEST_SUITE_P(
    Method, TilingUtilsTest,
    ::testing::Values(TilingMethod::kAllocatedMemory, TilingMethod::kInPlace),
    [](const ::testing::TestParamInfo<TilingMethod>& info) {
      switch (info.param) {
        case TilingMethod::kAllocatedMemory:
          return "AllocatedMemory";
        case TilingMethod::kInPlace:
          return "InPlace";
      }
    });

TEST_P(TilingUtilsTest, Standard2D) {
  // 2D matrix of shape 8x8, element type float (4 bytes).
  // Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 4x4.
  const int64_t H = 8;
  const int64_t W = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  // Tiled buffer size should be H * W * sizeof(float)
  const int64_t tiled_size_bytes = H * W * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify tiled structure manually for a few values.
  // Original 8x8 matrix is divided into 2x2 grid of 4x4 tiles:
  // Tile(0, 0) covers rows 0-3, cols 0-3.
  // Tile(0, 1) covers rows 0-3, cols 4-7.
  // Tile(1, 0) covers rows 4-7, cols 0-3.
  // Tile(1, 1) covers rows 4-7, cols 4-7.
  //
  // Let's check row 0, col 4: linear index is 0*8 + 4 = 4. Value is 4.0.
  // In tiling: tile_h = 0, tile_w = 1. Tile index = 0 * 2 + 1 = 1.
  // Within tile: row_offset = 0, col_offset = 0. Offset within tile = 0*4 + 0 =
  // 0. Physical offset in tiled buffer: (tile_index * (tH * tW) +
  // offset_within_tile) * sizeof(float) = (1 * 16 + 0) * 4 = 64 bytes. Float
  // value at 64 bytes should be 4.0f.
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[16], 4.0f);

  // Let's check row 4, col 1: linear index is 4*8 + 1 = 33. Value is 33.0.
  // In tiling: tile_h = 1, tile_w = 0. Tile index = 1 * 2 + 0 = 2.
  // Within tile: row_offset = 0, col_offset = 1. Offset within tile = 0*4 + 1
  // = 1. Physical offset: (2 * 16 + 1) * 4 = 132 bytes (index 33 in float
  // array). Float value at index 33 in tiled array should be 33.0f.
  EXPECT_EQ(dst_tiled_float[33], 33.0f);

  // Now detile.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Standard2DLarge) {
  // 2D matrix of shape 256x512, element type float (4 bytes).
  // Total size: 256 * 512 * 4 = 512KB, which is > 128KB (triggers parallel
  // path). Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 128x128.
  const int64_t H = 256;
  const int64_t W = 512;
  const int64_t tH = 128;
  const int64_t tW = 128;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = H * W * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Now detile.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    ASSERT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Padding2DLarge) {
  // 2D matrix of shape 250x500, element type float (4 bytes).
  // Total size: 250 * 500 * 4 = 500KB, which is > 128KB (triggers parallel
  // path). Dimensions do not divide the tile size (128x128). Number of tiles in
  // H: ceil(250 / 128) = 2. Number of tiles in W: ceil(500 / 128) = 4. Total
  // tiles = 8. Total size in tiled buffer = 2 * 4 * 128 * 128 * 4 = 524288
  // bytes (512KB).
  const int64_t H = 250;
  const int64_t W = 500;
  const int64_t tH = 128;
  const int64_t tW = 128;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = 2 * 4 * tH * tW * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Now detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    ASSERT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Padding2D) {
  // 2D matrix of shape 6x6, element type float (4 bytes).
  // Layout has minor_to_major={1, 0} (row-major), and tiling with tile
  // dimensions 4x4. The dimensions do not divide the tile size. Number of tiles
  // in W: ceil(6 / 4) = 2. Number of tiles in H: ceil(6 / 4) = 2. Total tiles
  // = 4. Total size in tiled buffer = 2 * 2 * 4 * 4 * 4 = 256 bytes (64
  // floats).
  const int64_t H = 6;
  const int64_t W = 6;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = 2 * 2 * tH * tW * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify that padding elements are 0.
  // E.g., Tile(1, 1) covers rows 4-7, cols 4-7. But matrix only goes up to
  // index 5. Row 5, col 5 is in matrix. Row 5, col 6 is padding. Tile(1, 1)
  // index is 1 * 2 + 1 = 3. Within Tile(1, 1), Row 5 (offset 1), Col 6 (offset
  // 2) -> offset = 1 * 4 + 2 = 6. Float index: 3 * 16 + 6 = 54. It should be
  // 0.0f.
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[54], 0.0f);

  // Now detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, PermutedOuterLayout) {
  // Shape: F32[2, 3, 8, 8]
  // Outer dimensions: 0 (size 2), 1 (size 3)
  // Tiled dimensions: 2 (size 8), 3 (size 8)
  // Tile dimensions: 4x4
  // Layout minor_to_major: {3, 2, 0, 1}
  const int64_t D0 = 2;
  const int64_t D1 = 3;
  const int64_t D2 = 8;
  const int64_t D3 = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {D0, D1, D2, D3}, {3, 2, 0, 1},
      {xla::Tile({tH, tW})});

  const int64_t num_elements = D0 * D1 * D2 * D3;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  const float* dst_tiled_float =
      reinterpret_cast<const float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[128], 64.0f);

  // Detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Bf16SubTiling) {
  // Shape: BF16[16, 256]
  // Tile 1: 8x128
  // Tile 2: 2x1
  const int64_t H = 16;
  const int64_t W = 256;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(uint16_t);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  const uint16_t* dst_tiled_uint16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_uint16[1], 256);

  // Detile back.
  std::vector<uint16_t> dst_linear(num_elements, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Standard3D) {
  // 3D matrix of shape 2x8x8, element type float (4 bytes).
  // Layout has minor_to_major={2, 1, 0} (standard row-major), and tiling with
  // tile dimensions 4x4.
  const int64_t D0 = 2;
  const int64_t H = 8;
  const int64_t W = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {D0, H, W}, {2, 1, 0}, {xla::Tile({tH, tW})});

  const int64_t num_elements = D0 * H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify tiled structure.
  // Block 0 (d0=0) covers indices 0-63.
  // Block 1 (d0=1) covers indices 64-127.
  // Within Block 1, the 2D matrix is 8x8 tiled with 4x4.
  // Let's check d0=1, row 0, col 4: linear index is 1*64 + 0*8 + 4 = 68. Value
  // is 68.0. Physically, Block 1 starts at 64 * 4 = 256 bytes. Within Block 1,
  // row 0, col 4 is in Tile(0,1), offset 0. Tile(0,1) index is 1. Offset within
  // Block 1: 1 * 16 * 4 = 64 bytes. Total physical offset: 256 + 64 = 320 bytes
  // (index 80 in float array).
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[80], 68.0f);

  // Detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Standard4D) {
  // 4D matrix of shape 2x3x8x8, element type float (4 bytes).
  // Layout has minor_to_major={3, 2, 1, 0} (standard row-major), and tiling
  // with tile dimensions 4x4.
  const int64_t D0 = 2;
  const int64_t D1 = 3;
  const int64_t H = 8;
  const int64_t W = 8;
  const int64_t tH = 4;
  const int64_t tW = 4;

  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {D0, D1, H, W}, {3, 2, 1, 0},
      {xla::Tile({tH, tW})});

  const int64_t num_elements = D0 * D1 * H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i);
  }

  const int64_t tiled_size_bytes = num_elements * sizeof(float);
  std::vector<uint8_t> dst_tiled(tiled_size_bytes);

  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify tiled structure.
  // We have D0 * D1 = 6 batches.
  // Each batch is 8x8 tiled with 4x4.
  // Let's check d0=1, d1=1, row 0, col 4:
  // Linear index: d0*(D1*H*W) + d1*(H*W) + row*W + col
  // = 1*(3*8*8) + 1*(8*8) + 0*8 + 4
  // = 192 + 64 + 4 = 260. Value is 260.0.
  // Physically, each batch has size H * W * sizeof(float) = 64 * 4 = 256 bytes.
  // Batch index is d0 * D1 + d1 = 1 * 3 + 1 = 4.
  // Batch offset: 4 * 256 = 1024 bytes.
  // Within batch 4, row 0, col 4 is in Tile(0,1), offset 0.
  // Tile(0,1) index is 1.
  // Offset within batch: 1 * 16 * 4 = 64 bytes.
  // Total physical offset: 1024 + 64 = 1088 bytes (index 272 in float array).
  float* dst_tiled_float = reinterpret_cast<float*>(dst_tiled.data());
  EXPECT_EQ(dst_tiled_float[272], 260.0f);

  // Detile back.
  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, UntiledLayout) {
  // Test that untiled buffers pass through directly via std::memcpy.
  const int64_t H = 16;
  const int64_t W = 32;
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {H, W});
  // Verify layout has no tiles
  ASSERT_TRUE(shape.layout().tiles().empty());

  const int64_t num_elements = H * W;
  std::vector<float> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<float>(i * 2 + 1);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(float));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  std::vector<float> dst_linear(num_elements, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Standard1DContiguous) {
  // 1D tensor with tile (1, 128) where W % 128 == 0 (identical contiguous
  // layout).
  const int64_t W = 4096;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0}, {xla::Tile({1, 128})});

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  std::vector<uint8_t> dst_tiled(W * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  std::vector<uint16_t> dst_linear(W, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < W; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsPredicateTest, IsStandardRowMajorTiled) {
  // 1D shape with 1D tile.
  xla::Shape shape_1d_1d = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {4096}, {0}, {xla::Tile({128})});
  EXPECT_TRUE(IsStandardRowMajorTiled(shape_1d_1d, shape_1d_1d.layout()));

  // 1D shape with 2D tile.
  xla::Shape shape_1d_2d = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {4096}, {0}, {xla::Tile({8, 128})});
  EXPECT_TRUE(IsStandardRowMajorTiled(shape_1d_2d, shape_1d_2d.layout()));

  // 1D shape with 2-level 2D tile.
  xla::Shape shape_1d_packed = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {4096}, {0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});
  EXPECT_TRUE(
      IsStandardRowMajorTiled(shape_1d_packed, shape_1d_packed.layout()));

  // 1D shape with degenerate tile dimensions (zero or negative).
  xla::Shape shape_1d_zero = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {4096}, {0});
  shape_1d_zero.mutable_layout()->clear_tiles();
  shape_1d_zero.mutable_layout()->add_tiles()->add_dimensions(0);
  EXPECT_FALSE(IsStandardRowMajorTiled(shape_1d_zero, shape_1d_zero.layout()));

  xla::Shape shape_1d_neg = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {4096}, {0});
  shape_1d_neg.mutable_layout()->clear_tiles();
  shape_1d_neg.mutable_layout()->add_tiles()->add_dimensions(-1);
  EXPECT_FALSE(IsStandardRowMajorTiled(shape_1d_neg, shape_1d_neg.layout()));

  // 2D shape with 1D tile should NOT be standard row-major tiled (requires 2D
  // tile).
  xla::Shape shape_2d_1d = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {256, 512}, {1, 0}, {xla::Tile({128})});
  EXPECT_FALSE(IsStandardRowMajorTiled(shape_2d_1d, shape_2d_1d.layout()));

  // 2D shape with 2D tile.
  xla::Shape shape_2d_2d = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {256, 512}, {1, 0}, {xla::Tile({8, 128})});
  EXPECT_TRUE(IsStandardRowMajorTiled(shape_2d_2d, shape_2d_2d.layout()));

  // Non-row-major 2D layout (column-major {0, 1}).
  xla::Shape shape_2d_col = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {256, 512}, {0, 1}, {xla::Tile({8, 128})});
  EXPECT_FALSE(IsStandardRowMajorTiled(shape_2d_col, shape_2d_col.layout()));
}

TEST_P(TilingUtilsTest, Standard1DSingleDimensionTile) {
  const int64_t W = 4096;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0}, {xla::Tile({128})});

  EXPECT_TRUE(IsStandardRowMajorTiled(shape, shape.layout()));
  EXPECT_EQ(GetTiledBufferElements(shape), W);

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  std::vector<uint8_t> dst_tiled(W * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify byte-for-byte equivalence against XLA nested tiling reference.
  for (int64_t i = 0; i < W; ++i) {
    int64_t ref_phys_idx =
        xla::LayoutUtil::LinearIndexForNestedTiling(shape, {i});
    uint16_t actual_val =
        reinterpret_cast<const uint16_t*>(dst_tiled.data())[ref_phys_idx];
    EXPECT_EQ(actual_val, src_linear[i])
        << "Physical mismatch at element " << i;
  }

  std::vector<uint16_t> dst_linear(W, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < W; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Standard1DSingleDimensionTileWithPadding) {
  // 1D tensor shape 250 with 1D tile 128: Ceil(250 / 128) = 2 tiles -> 256
  // physical elements.
  const int64_t W = 250;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0}, {xla::Tile({128})});

  EXPECT_TRUE(IsStandardRowMajorTiled(shape, shape.layout()));
  int64_t tiled_elements = GetTiledBufferElements(shape);
  EXPECT_EQ(tiled_elements, 256);

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i + 7);
  }

  std::vector<uint8_t> dst_tiled(tiled_elements * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify elements 0..249 match XLA nested tiling reference.
  const uint16_t* dst_tiled_bf16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  for (int64_t i = 0; i < W; ++i) {
    int64_t ref_phys_idx =
        xla::LayoutUtil::LinearIndexForNestedTiling(shape, {i});
    EXPECT_EQ(dst_tiled_bf16[ref_phys_idx], src_linear[i])
        << "Physical mismatch at element " << i;
  }

  // Verify padding elements 250..255 are zeroed.
  for (int64_t i = 250; i < 256; ++i) {
    EXPECT_EQ(dst_tiled_bf16[i], 0) << "Padding not zeroed at " << i;
  }

  std::vector<uint16_t> dst_linear(W, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < W; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Standard1DSingleDimensionTileSmall) {
  // Shape 17 with tile 128: 1 tile -> 128 elements.
  const int64_t W = 17;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {W}, {0}, {xla::Tile({128})});

  EXPECT_TRUE(IsStandardRowMajorTiled(shape, shape.layout()));
  int64_t tiled_elements = GetTiledBufferElements(shape);
  EXPECT_EQ(tiled_elements, 128);

  std::vector<float> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<float>(i * 1.5f);
  }

  std::vector<uint8_t> dst_tiled(tiled_elements * sizeof(float));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  const float* dst_tiled_f32 = reinterpret_cast<const float*>(dst_tiled.data());
  for (int64_t i = 0; i < W; ++i) {
    EXPECT_EQ(dst_tiled_f32[i], src_linear[i]);
  }
  for (int64_t i = W; i < tiled_elements; ++i) {
    EXPECT_EQ(dst_tiled_f32[i], 0.0f) << "Padding not zeroed at index " << i;
  }

  std::vector<float> dst_linear(W, 0.0f);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < W; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, Standard1DWithTilePadding) {
  // 1D tensor with tile (8, 128) where W = 300 (has horizontal and vertical
  // tile padding).
  const int64_t W = 300;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0}, {xla::Tile({8, 128})});

  EXPECT_TRUE(IsStandardRowMajorTiled(shape, shape.layout()));
  int64_t tiled_elements = GetTiledBufferElements(shape);
  // Ceil(1/8)=1, Ceil(300/128)=3 => 1 * 3 * 8 * 128 = 3072 elements.
  EXPECT_EQ(tiled_elements, 3072);

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i + 5);
  }

  std::vector<uint8_t> dst_tiled(tiled_elements * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  // Verify byte-for-byte equivalence against XLA nested tiling reference.
  const uint16_t* dst_tiled_bf16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  for (int64_t i = 0; i < W; ++i) {
    int64_t ref_phys_idx =
        xla::LayoutUtil::LinearIndexForNestedTiling(shape, {i});
    EXPECT_EQ(dst_tiled_bf16[ref_phys_idx], src_linear[i])
        << "Physical mismatch at element " << i;
  }

  std::vector<uint16_t> dst_linear(W, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < W; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST(TilingUtilsPredicateTest, DegenerateTileDimensionsSafe) {
  // Shape 128 with tile {0} (degenerate tile dimension)
  xla::Shape shape_zero = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {128}, {0});
  shape_zero.mutable_layout()->clear_tiles();
  shape_zero.mutable_layout()->add_tiles()->add_dimensions(0);

  EXPECT_FALSE(IsStandardRowMajorTiled(shape_zero, shape_zero.layout()));
  // GetTiledBufferElements must not divide by zero
  EXPECT_EQ(GetTiledBufferElements(shape_zero), 128);
}

TEST_P(TilingUtilsTest, SingleColumn2DContiguous) {
  // 2D tensor where W == tile_W (128) and H % tile_H == 0 (256 % 8 == 0).
  const int64_t H = 256;
  const int64_t W = 128;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0}, {xla::Tile({8, 128})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(uint16_t));
  absl::Status tile_status =
      TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                 dst_tiled.data(), shape, shape.layout());
  EXPECT_TRUE(tile_status.ok()) << tile_status.ToString();

  std::vector<uint16_t> dst_linear(num_elements, 0);
  absl::Status detile_status = DetileBuffer(
      dst_tiled.data(), reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
      shape.layout());
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, SpecializedRowBytes_Int8_StandardTile) {
  // S8 dtype with tile (8, 128) => row_bytes = 128.
  const int64_t H = 64;
  const int64_t W = 256;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0}, {xla::Tile({8, 128})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(int8_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, SpecializedRowBytes_Int8_TransposedTile) {
  // S8 dtype with tile (128, 8) => row_bytes = 8.
  const int64_t H = 256;
  const int64_t W = 64;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0}, {xla::Tile({128, 8})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(int8_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, SpecializedRowBytes_BF16_StandardAndTransposed) {
  // BF16 dtype with tile (8, 128) => row_bytes = 256.
  // BF16 dtype with tile (128, 8) => row_bytes = 16.
  const int64_t H = 64;
  const int64_t W = 256;

  // Standard (8, 128) -> row_bytes = 256
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {H, W}, {1, 0}, {xla::Tile({8, 128})});
    const int64_t num_elements = H * W;
    std::vector<uint16_t> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i * 3 + 1);
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(uint16_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<uint16_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }

  // Transposed (128, 8) -> row_bytes = 16
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {256, 64}, {1, 0}, {xla::Tile({128, 8})});
    const int64_t num_elements = 256 * 64;
    std::vector<uint16_t> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i * 3 + 1);
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(uint16_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<uint16_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }
}

TEST_P(TilingUtilsTest, SpecializedRowBytes_FP32_StandardAndTransposed) {
  // F32 dtype with tile (8, 128) => row_bytes = 512.
  // F32 dtype with tile (128, 8) => row_bytes = 32.
  const int64_t H = 64;
  const int64_t W = 256;

  // Standard (8, 128) -> row_bytes = 512
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {H, W}, {1, 0}, {xla::Tile({8, 128})});
    const int64_t num_elements = H * W;
    std::vector<float> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<float>(i) * 1.5f;
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(float));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<float> dst_linear(num_elements, 0.0f);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }

  // Transposed (128, 8) -> row_bytes = 32
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {256, 64}, {1, 0}, {xla::Tile({128, 8})});
    const int64_t num_elements = 256 * 64;
    std::vector<float> src_linear(num_elements);
    for (int i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<float>(i) * 1.5f;
    }
    std::vector<uint8_t> dst_tiled(num_elements * sizeof(float));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<float> dst_linear(num_elements, 0.0f);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    for (int i = 0; i < num_elements; ++i) {
      EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
    }
  }
}

TEST_P(TilingUtilsTest, SpecializedRowBytes_64ByteRow) {
  // S8 with tile (8, 64) -> row_bytes = 64
  const int64_t H = 32;
  const int64_t W = 128;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0}, {xla::Tile({8, 64})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  std::vector<uint8_t> dst_tiled(num_elements * sizeof(int8_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());

  for (int i = 0; i < num_elements; ++i) {
    EXPECT_EQ(dst_linear[i], src_linear[i]) << "Mismatch at index " << i;
  }
}

TEST_P(TilingUtilsTest, SpecializedRowBytes_MultiBatch_AllTypes) {
  // Multi-batch 3D tensor: [4, 64, 256] with tile (8, 128)
  const int64_t B = 4;
  const int64_t H = 64;
  const int64_t W = 256;
  const int64_t total_elements = B * H * W;

  // 1. S8 (row_bytes = 128)
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::S8, {B, H, W}, {2, 1, 0}, {xla::Tile({8, 128})});
    std::vector<int8_t> src_linear(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      src_linear[i] = static_cast<int8_t>(i % 127);
    }
    std::vector<uint8_t> dst_tiled(total_elements * sizeof(int8_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<int8_t> dst_linear(total_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    EXPECT_EQ(dst_linear, src_linear);
  }

  // 2. BF16 (row_bytes = 256)
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {B, H, W}, {2, 1, 0}, {xla::Tile({8, 128})});
    std::vector<uint16_t> src_linear(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i);
    }
    std::vector<uint8_t> dst_tiled(total_elements * sizeof(uint16_t));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<uint16_t> dst_linear(total_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    EXPECT_EQ(dst_linear, src_linear);
  }

  // 3. F32 (row_bytes = 512)
  {
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::F32, {B, H, W}, {2, 1, 0}, {xla::Tile({8, 128})});
    std::vector<float> src_linear(total_elements);
    for (int i = 0; i < total_elements; ++i) {
      src_linear[i] = static_cast<float>(i) * 0.25f;
    }
    std::vector<uint8_t> dst_tiled(total_elements * sizeof(float));
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<float> dst_linear(total_elements, 0.0f);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    EXPECT_EQ(dst_linear, src_linear);
  }
}

TEST_P(TilingUtilsTest, NestedPacked_BF16_ExactTiles2D) {
  // Shape [64, 256] with nested layout {1, 0 : T(8, 128)(2, 1)}
  const int64_t H = 64;
  const int64_t W = 256;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i * 5 + 1);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  EXPECT_EQ(total_physical_elements, num_elements);

  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t));
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  // Verify against XLA canonical LinearIndexForNestedTiling
  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx])
            << "Mismatch at (" << indices[0] << ", " << indices[1] << ")";
        return true;
      });

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, NestedPacked_BF16_WithPadding2D) {
  // Shape [53, 175] with nested layout {1, 0 : T(8, 128)(2, 1)} (has padding in
  // H and W)
  const int64_t H = 53;
  const int64_t W = 175;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i * 3 + 7);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  // CeilOfRatio(53, 8) = 7, CeilOfRatio(175, 128) = 2 => 7 * 2 * 1024 = 14336
  EXPECT_EQ(total_physical_elements, 7 * 2 * 1024);

  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  // Verify against XLA canonical LinearIndexForNestedTiling for all valid
  // elements
  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx])
            << "Mismatch at (" << indices[0] << ", " << indices[1] << ")";
        return true;
      });

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, NestedPacked_BF16_1DTensor) {
  // Shape [2304] with nested layout {0 : T(8, 128)(2, 1)}
  const int64_t W = 2304;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {W}, {0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  std::vector<uint16_t> src_linear(W);
  for (int i = 0; i < W; ++i) {
    src_linear[i] = static_cast<uint16_t>(i + 42);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx])
            << "Mismatch at index " << indices[0];
        return true;
      });

  std::vector<uint16_t> dst_linear(W, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, NestedPacked_BF16_MultiBatch3D) {
  // Shape [3, 40, 300] with nested layout {2, 1, 0 : T(8, 128)(2, 1)}
  const int64_t B = 3;
  const int64_t H = 40;
  const int64_t W = 300;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {B, H, W}, {2, 1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = B * H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 65535);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const uint16_t* tiled_u16 =
      reinterpret_cast<const uint16_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx]);
        return true;
      });

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, NestedPacked_S8_PackingFactor4) {
  // Shape [32, 256] with nested layout {1, 0 : T(8, 128)(4, 1)} (FP8/INT8 4-row
  // packing)
  const int64_t H = 32;
  const int64_t W = 256;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({4, 1})});

  const int64_t num_elements = H * W;
  std::vector<int8_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<int8_t>(i % 127);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(int8_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  xla::Shape standard_shape =
      xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
  const int8_t* tiled_s8 = reinterpret_cast<const int8_t*>(dst_tiled.data());
  xla::ShapeUtil::ForEachIndexNoStatus(
      shape, [&](absl::Span<const int64_t> indices) -> bool {
        int64_t linear_idx = xla::IndexUtil::MultidimensionalIndexToLinearIndex(
            standard_shape, indices);
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
        EXPECT_EQ(tiled_s8[physical_idx], src_linear[linear_idx]);
        return true;
      });

  std::vector<int8_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST(TilingUtilsThreadPoolTest, ThreadPool_ExecuteOneTask) {
  tpu_raiden::NumaThreadPool pool(2);
  // On empty pool, ExecuteOneTask returns false
  EXPECT_FALSE(pool.ExecuteOneTask());

  std::atomic<bool> executed{false};
  // Schedule a task
  pool.Schedule(
      [&executed]() { executed.store(true, std::memory_order_release); });

  // Call ExecuteOneTask from current thread to help drain
  for (int i = 0; i < 100 && !executed.load(std::memory_order_acquire); ++i) {
    pool.ExecuteOneTask();
    std::this_thread::yield();
  }
  EXPECT_TRUE(executed.load(std::memory_order_acquire));
}

TEST_P(TilingUtilsTest, ThresholdGatedTiling_SmallTensorInline) {
  // Tensor of size 1 MB (512 x 1024 bfloat16 = 1,048,576 bytes)
  // This is < 4 MB threshold, so it executes inline on calling thread.
  const int64_t H = 512;
  const int64_t W = 1024;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i & 0x7FFF);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);

  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  std::vector<uint16_t> dst_linear(num_elements, 0);
  EXPECT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());

  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, ThresholdGatedTiling_LargeTensorParallel) {
  // Tensor of size 8 MB (2048 x 2048 bfloat16 = 8,388,608 bytes)
  // This is >= 4 MB threshold, so it exercises the parallel sub-tasking path.
  const int64_t H = 2048;
  const int64_t W = 2048;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 32749);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled_seq(total_physical_elements * sizeof(uint16_t),
                                     0);
  std::vector<uint8_t> dst_tiled_par(total_physical_elements * sizeof(uint16_t),
                                     0);

  // Two independent parallel runs must agree bit for bit: the dynamic
  // work-stealing split is non-deterministic across runs.
  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled_seq.data(), shape, shape.layout())
                  .ok());
  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled_par.data(), shape, shape.layout())
                  .ok());
  EXPECT_EQ(dst_tiled_par, dst_tiled_seq);

  // Detile in parallel and verify full round-trip accuracy
  std::vector<uint16_t> dst_linear(num_elements, 0);
  EXPECT_TRUE(DetileBuffer(dst_tiled_par.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, ThresholdGatedTiling_ParallelPathPinnedToNumaNode) {
  // Same parallel path with helper tasks pinned to NUMA node 0, which exists
  // on every host. Output must match the sequential path bit for bit.
  const int64_t H = 2048;
  const int64_t W = 2048;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 32749);
  }
  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled_seq(total_physical_elements * sizeof(uint16_t),
                                     0);
  std::vector<uint8_t> dst_tiled_par(total_physical_elements * sizeof(uint16_t),
                                     0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled_seq.data(), shape, shape.layout())
                  .ok());

  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled_par.data(), shape, shape.layout(),
                         /*numa_node=*/0)
                  .ok());
  EXPECT_EQ(dst_tiled_par, dst_tiled_seq);

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled_par.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout(), /*numa_node=*/0)
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest,
       ThresholdGatedTiling_InvokedFromWorkerThread_NoDeadlock) {
  // Test invoking TileBuffer on a large 8 MB tensor FROM WITHIN a worker thread
  // of another NumaThreadPool, as the per-shard tiling tasks do in production.
  // The caller must not be mistaken for a shared-pool worker and must not
  // starve while draining helper tasks.
  const int64_t H = 2048;
  const int64_t W = 2048;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 32749);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);

  tpu_raiden::NumaThreadPool pool(4);

  // Dispatch from inside a NumaThreadPool worker thread
  auto future = pool.Schedule([&]() -> absl::Status {
    EXPECT_TRUE(tpu_raiden::NumaThreadPool::IsCurrentThreadWorker());
    return TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                      dst_tiled.data(), shape, shape.layout());
  });

  absl::Status status = future.get();
  EXPECT_TRUE(status.ok()) << status.ToString();

  // Verify round-trip correctness
  std::vector<uint16_t> dst_linear(num_elements, 0);
  auto detile_future = pool.Schedule([&]() -> absl::Status {
    EXPECT_TRUE(tpu_raiden::NumaThreadPool::IsCurrentThreadWorker());
    return DetileBuffer(dst_tiled.data(),
                        reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                        shape.layout());
  });

  absl::Status detile_status = detile_future.get();
  EXPECT_TRUE(detile_status.ok()) << detile_status.ToString();
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, ThresholdGatedTiling_LargeTensorWithPadding) {
  // Test large tensor with dimensions not aligned to tile size:
  // H = 2050, W = 2050 (tile is 8, 128)
  // Total size ~8.4 MB (triggers parallel path and exercises padding logic)
  const int64_t H = 2050;
  const int64_t W = 2050;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {H, W}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  const int64_t num_elements = H * W;
  std::vector<uint16_t> src_linear(num_elements);
  for (int i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>(i % 32749);
  }

  int64_t total_physical_elements = GetTiledBufferElements(shape);
  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);

  EXPECT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  std::vector<uint16_t> dst_linear(num_elements, 0);
  EXPECT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());

  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, ByteProportionalChunking_And_4ShardConcurrentParity) {
  // 1. 4.59 MB tensor ([1792, 1280] BF16 -> 4,587,520 bytes, desired_chunks=1)
  {
    const int64_t H = 1792;
    const int64_t W = 1280;
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {H, W}, {1, 0},
        {xla::Tile({8, 128}), xla::Tile({2, 1})});
    const int64_t num_elements = H * W;
    std::vector<uint16_t> src(num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
      src[i] = static_cast<uint16_t>(i % 32749);
    }
    std::vector<uint8_t> tiled(GetTiledBufferElements(shape) * sizeof(uint16_t),
                               0);
    std::vector<uint16_t> dst(num_elements, 0);
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src.data()),
                           tiled.data(), shape, shape.layout())
                    .ok());
    ASSERT_TRUE(DetileBuffer(tiled.data(),
                             reinterpret_cast<uint8_t*>(dst.data()), shape,
                             shape.layout())
                    .ok());
    EXPECT_EQ(dst, src);
  }

  // 2. 4 concurrent shards of 33.55 MB MoE tensor ([2, 4096, 2048] BF16 ->
  // 33,554,432 bytes = 8 * 4 MiB, desired_chunks=8) sharing GetThreadPool()
  const int64_t E = 2;
  const int64_t H = 4096;
  const int64_t W = 2048;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {E, H, W}, {2, 1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});
  const int64_t num_elements = E * H * W;
  const int64_t tiled_bytes = GetTiledBufferElements(shape) * sizeof(uint16_t);

  constexpr int kNumShards = 4;
  std::vector<std::vector<uint16_t>> shard_srcs(kNumShards);
  std::vector<std::vector<uint8_t>> shard_tiled(kNumShards);
  std::vector<std::vector<uint16_t>> shard_dsts(kNumShards);
  for (int s = 0; s < kNumShards; ++s) {
    shard_srcs[s].resize(num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
      shard_srcs[s][i] = static_cast<uint16_t>((i + s * 101) % 32749);
    }
    shard_tiled[s].assign(tiled_bytes, 0);
    shard_dsts[s].assign(num_elements, 0);
  }

  std::vector<std::thread> threads;
  threads.reserve(kNumShards);
  for (int s = 0; s < kNumShards; ++s) {
    threads.emplace_back([&, s]() {
      ASSERT_TRUE(
          TileBuffer(reinterpret_cast<const uint8_t*>(shard_srcs[s].data()),
                     shard_tiled[s].data(), shape, shape.layout())
              .ok());
      ASSERT_TRUE(DetileBuffer(shard_tiled[s].data(),
                               reinterpret_cast<uint8_t*>(shard_dsts[s].data()),
                               shape, shape.layout())
                      .ok());
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (int s = 0; s < kNumShards; ++s) {
    EXPECT_EQ(shard_dsts[s], shard_srcs[s]);
  }
}

TEST(TilingUtilsPredicateTest, IsStandardColMajorTiledAnd1DPacked) {
  // 2D column-major unpacked {0,1:T(8,128)}
  xla::Shape col_unpacked = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {2560, 1216}, {0, 1}, {xla::Tile({8, 128})});
  EXPECT_TRUE(IsStandardColMajorTiled(col_unpacked, col_unpacked.layout()));
  EXPECT_FALSE(IsStandardRowMajorTiled(col_unpacked, col_unpacked.layout()));

  // 2D column-major packed {0,1:T(8,128)(2,1)}
  xla::Shape col_packed_p2 = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {2560, 1216}, {0, 1},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});
  EXPECT_TRUE(IsStandardColMajorTiled(col_packed_p2, col_packed_p2.layout()));
  EXPECT_FALSE(IsStandardRowMajorTiled(col_packed_p2, col_packed_p2.layout()));

  // 2D column-major packed {0,1:T(8,128)(4,1)}
  xla::Shape col_packed_p4 = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {2560, 1216}, {0, 1},
      {xla::Tile({8, 128}), xla::Tile({4, 1})});
  EXPECT_TRUE(IsStandardColMajorTiled(col_packed_p4, col_packed_p4.layout()));

  // 3D batched column-major {1,2,0:T(8,128)(2,1)}
  xla::Shape col_3d = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {4, 256, 128}, {1, 2, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});
  EXPECT_TRUE(IsStandardColMajorTiled(col_3d, col_3d.layout()));
  EXPECT_FALSE(IsStandardRowMajorTiled(col_3d, col_3d.layout()));

  // 3D with permuted batch dimension {0,1,2} is NOT standard batched col-major
  xla::Shape col_3d_permuted = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {4, 256, 128}, {0, 1, 2},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});
  EXPECT_FALSE(
      IsStandardColMajorTiled(col_3d_permuted, col_3d_permuted.layout()));

  // 1D tensor is NOT column-major tiled
  xla::Shape shape_1d = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {2560}, {0}, {xla::Tile({128})});
  EXPECT_FALSE(IsStandardColMajorTiled(shape_1d, shape_1d.layout()));

  // 1D 3-level packed {0:T(1024)(128)(2,1)}
  xla::Shape shape_1d_3level = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {2560}, {0},
      {xla::Tile({1024}), xla::Tile({128}), xla::Tile({2, 1})});
  EXPECT_TRUE(
      IsStandardRowMajorTiled(shape_1d_3level, shape_1d_3level.layout()));

  // 1D 2-level packed {0:T(128)(2,1)}
  xla::Shape shape_1d_2level = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {2560}, {0},
      {xla::Tile({128}), xla::Tile({2, 1})});
  EXPECT_TRUE(
      IsStandardRowMajorTiled(shape_1d_2level, shape_1d_2level.layout()));

  // 1D 2-level unpacked {0:T(1024)(128)}
  xla::Shape shape_1d_2level_unpacked =
      xla::ShapeUtil::MakeShapeWithDenseLayout(
          xla::PrimitiveType::BF16, {2560}, {0},
          {xla::Tile({1024}), xla::Tile({128})});
  EXPECT_TRUE(IsStandardRowMajorTiled(shape_1d_2level_unpacked,
                                      shape_1d_2level_unpacked.layout()));
}

TEST_P(TilingUtilsTest, ColMajor_Packed_BF16_ExactAndQwen3DownProj) {
  // Test Qwen3-4B (TP=8 on TPU v6e) mlp.down_proj.weight shape:
  // BF16[2560, 1216] with {0, 1: T(8, 128)(2, 1)}
  const int64_t D0 = 2560;
  const int64_t D1 = 1216;
  xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {D0, D1}, {0, 1},
      {xla::Tile({8, 128}), xla::Tile({2, 1})});

  ASSERT_TRUE(IsStandardColMajorTiled(shape, shape.layout()));
  const int64_t num_elements = D0 * D1;
  const int64_t total_physical_elements = GetTiledBufferElements(shape);
  EXPECT_EQ(total_physical_elements, num_elements);

  std::vector<uint16_t> src_linear(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    src_linear[i] = static_cast<uint16_t>((i * 7 + 13) % 65521);
  }

  std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t), 0);
  ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                         dst_tiled.data(), shape, shape.layout())
                  .ok());

  // Verify against XLA canonical LinearIndexForNestedTiling on a smaller exact
  // shape [128, 64] so full element-by-element verification runs quickly.
  {
    xla::Shape small_shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {128, 64}, {0, 1},
        {xla::Tile({8, 128}), xla::Tile({2, 1})});
    const int64_t small_elems = 128 * 64;
    std::vector<uint16_t> small_src(small_elems);
    for (int64_t i = 0; i < small_elems; ++i) {
      small_src[i] = static_cast<uint16_t>(i * 5 + 1);
    }
    std::vector<uint8_t> small_tiled(small_elems * sizeof(uint16_t), 0);
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(small_src.data()),
                           small_tiled.data(), small_shape,
                           small_shape.layout())
                    .ok());

    xla::Shape standard_shape = xla::ShapeUtil::MakeShape(
        small_shape.element_type(), small_shape.dimensions());
    const uint16_t* tiled_u16 =
        reinterpret_cast<const uint16_t*>(small_tiled.data());
    xla::ShapeUtil::ForEachIndexNoStatus(
        small_shape, [&](absl::Span<const int64_t> indices) -> bool {
          int64_t linear_idx =
              xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                                 indices);
          int64_t physical_idx =
              xla::LayoutUtil::LinearIndexForNestedTiling(small_shape, indices);
          EXPECT_EQ(tiled_u16[physical_idx], small_src[linear_idx])
              << "Mismatch at (" << indices[0] << ", " << indices[1] << ")";
          return true;
        });
  }

  std::vector<uint16_t> dst_linear(num_elements, 0);
  ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                           reinterpret_cast<uint8_t*>(dst_linear.data()), shape,
                           shape.layout())
                  .ok());
  EXPECT_EQ(dst_linear, src_linear);
}

TEST_P(TilingUtilsTest, ColMajor_Packed_And_Unpacked_WithPadding2D) {
  // Shape [175, 53] with {0, 1: T(8, 128)(2, 1)} (padding in both D0 and D1)
  const int64_t D0 = 175;
  const int64_t D1 = 53;
  for (bool packed : {false, true}) {
    xla::Shape shape = packed ? xla::ShapeUtil::MakeShapeWithDenseLayout(
                                    xla::PrimitiveType::BF16, {D0, D1}, {0, 1},
                                    {xla::Tile({8, 128}), xla::Tile({2, 1})})
                              : xla::ShapeUtil::MakeShapeWithDenseLayout(
                                    xla::PrimitiveType::BF16, {D0, D1}, {0, 1},
                                    {xla::Tile({8, 128})});
    ASSERT_TRUE(IsStandardColMajorTiled(shape, shape.layout()));

    const int64_t num_elements = D0 * D1;
    std::vector<uint16_t> src_linear(num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i * 3 + 7);
    }

    int64_t total_physical_elements = GetTiledBufferElements(shape);
    // H = D1 = 53 (Ceil(53/8)=7), W = D0 = 175 (Ceil(175/128)=2) -> 14 * 1024
    EXPECT_EQ(total_physical_elements, 7 * 2 * 1024);

    // Pre-fill dst_tiled with non-zero garbage to verify padding is zeroed.
    std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t),
                                   0xFF);
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());

    std::vector<bool> is_valid_phys(total_physical_elements, false);
    xla::Shape standard_shape =
        xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
    const uint16_t* tiled_u16 =
        reinterpret_cast<const uint16_t*>(dst_tiled.data());
    xla::ShapeUtil::ForEachIndexNoStatus(
        shape, [&](absl::Span<const int64_t> indices) -> bool {
          int64_t linear_idx =
              xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                                 indices);
          int64_t physical_idx =
              xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
          is_valid_phys[physical_idx] = true;
          EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx])
              << "Mismatch at (" << indices[0] << ", " << indices[1] << ")";
          return true;
        });

    for (int64_t p_idx = 0; p_idx < total_physical_elements; ++p_idx) {
      if (!is_valid_phys[p_idx]) {
        EXPECT_EQ(tiled_u16[p_idx], 0)
            << "Padding not zeroed at physical index " << p_idx;
      }
    }

    std::vector<uint16_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    EXPECT_EQ(dst_linear, src_linear);
  }
}

TEST_P(TilingUtilsTest, ColMajor_Packed_S8_And_MultiBatch3D_And_Parallel) {
  // 1. S8 with {0, 1: T(8, 128)(4, 1)} and padding
  {
    const int64_t D0 = 175;
    const int64_t D1 = 29;
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::S8, {D0, D1}, {0, 1},
        {xla::Tile({8, 128}), xla::Tile({4, 1})});
    ASSERT_TRUE(IsStandardColMajorTiled(shape, shape.layout()));

    const int64_t num_elements = D0 * D1;
    std::vector<int8_t> src_linear(num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<int8_t>((i % 125) + 1);
    }

    int64_t total_physical_elements = GetTiledBufferElements(shape);
    std::vector<uint8_t> dst_tiled(total_physical_elements, 0xFF);
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());

    xla::Shape standard_shape =
        xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
    const int8_t* tiled_s8 = reinterpret_cast<const int8_t*>(dst_tiled.data());
    xla::ShapeUtil::ForEachIndexNoStatus(
        shape, [&](absl::Span<const int64_t> indices) -> bool {
          int64_t linear_idx =
              xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                                 indices);
          int64_t physical_idx =
              xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
          EXPECT_EQ(tiled_s8[physical_idx], src_linear[linear_idx]);
          return true;
        });

    std::vector<int8_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    EXPECT_EQ(dst_linear, src_linear);
  }

  // 2. 3D batched column-major [3, 300, 40] with {1, 2, 0: T(8, 128)(2, 1)}
  {
    const int64_t B = 3;
    const int64_t D0 = 300;
    const int64_t D1 = 40;
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {B, D0, D1}, {1, 2, 0},
        {xla::Tile({8, 128}), xla::Tile({2, 1})});
    ASSERT_TRUE(IsStandardColMajorTiled(shape, shape.layout()));

    const int64_t num_elements = B * D0 * D1;
    std::vector<uint16_t> src_linear(num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>((i * 11 + 3) % 65535);
    }

    int64_t total_physical_elements = GetTiledBufferElements(shape);
    std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t),
                                   0);
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());

    xla::Shape standard_shape =
        xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
    const uint16_t* tiled_u16 =
        reinterpret_cast<const uint16_t*>(dst_tiled.data());
    xla::ShapeUtil::ForEachIndexNoStatus(
        shape, [&](absl::Span<const int64_t> indices) -> bool {
          int64_t linear_idx =
              xla::IndexUtil::MultidimensionalIndexToLinearIndex(standard_shape,
                                                                 indices);
          int64_t physical_idx =
              xla::LayoutUtil::LinearIndexForNestedTiling(shape, indices);
          EXPECT_EQ(tiled_u16[physical_idx], src_linear[linear_idx]);
          return true;
        });

    std::vector<uint16_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    EXPECT_EQ(dst_linear, src_linear);
  }

  // 3. Large 8.4 MB tensor with padding [2050, 2050] on 4-thread pool
  {
    const int64_t D0 = 2050;
    const int64_t D1 = 2050;
    xla::Shape shape = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {D0, D1}, {0, 1},
        {xla::Tile({8, 128}), xla::Tile({2, 1})});
    const int64_t num_elements = D0 * D1;
    std::vector<uint16_t> src_linear(num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
      src_linear[i] = static_cast<uint16_t>(i % 32749);
    }
    int64_t total_physical_elements = GetTiledBufferElements(shape);
    std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t),
                                   0);
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                           dst_tiled.data(), shape, shape.layout())
                    .ok());
    std::vector<uint16_t> dst_linear(num_elements, 0);
    ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                             reinterpret_cast<uint8_t*>(dst_linear.data()),
                             shape, shape.layout())
                    .ok());
    EXPECT_EQ(dst_linear, src_linear);
  }
}

TEST_P(TilingUtilsTest, Packed1D_2Level_And_3Level_AllCases) {
  // Test {0:T(1024)(128)(2,1)} (Qwen3-4B 1D norm layout) and {0:T(128)(2,1)}
  // across exact (2048), outer-tile padded (2560), and arbitrary (300, 17)
  // lengths.
  for (int64_t W : {17, 300, 2048, 2500, 2560}) {
    for (bool three_level : {false, true}) {
      xla::Shape shape =
          three_level
              ? xla::ShapeUtil::MakeShapeWithDenseLayout(
                    xla::PrimitiveType::BF16, {W}, {0},
                    {xla::Tile({1024}), xla::Tile({128}), xla::Tile({2, 1})})
              : xla::ShapeUtil::MakeShapeWithDenseLayout(
                    xla::PrimitiveType::BF16, {W}, {0},
                    {xla::Tile({128}), xla::Tile({2, 1})});
      ASSERT_TRUE(IsStandardRowMajorTiled(shape, shape.layout()));

      std::vector<uint16_t> src_linear(W);
      for (int64_t i = 0; i < W; ++i) {
        src_linear[i] = static_cast<uint16_t>(i * 7 + 19);
      }

      int64_t total_physical_elements = GetTiledBufferElements(shape);
      std::vector<uint8_t> dst_tiled(total_physical_elements * sizeof(uint16_t),
                                     0xFF);
      ASSERT_TRUE(
          TileBuffer(reinterpret_cast<const uint8_t*>(src_linear.data()),
                     dst_tiled.data(), shape, shape.layout())
              .ok());

      std::vector<bool> is_valid_phys(total_physical_elements, false);
      const uint16_t* tiled_u16 =
          reinterpret_cast<const uint16_t*>(dst_tiled.data());
      for (int64_t i = 0; i < W; ++i) {
        int64_t physical_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape, {i});
        ASSERT_GE(physical_idx, 0);
        ASSERT_LT(physical_idx, total_physical_elements);
        is_valid_phys[physical_idx] = true;
        EXPECT_EQ(tiled_u16[physical_idx], src_linear[i])
            << "Mismatch at W=" << W << ", three_level=" << three_level
            << ", i=" << i;
      }
      for (int64_t p_idx = 0; p_idx < total_physical_elements; ++p_idx) {
        if (!is_valid_phys[p_idx]) {
          EXPECT_EQ(tiled_u16[p_idx], 0)
              << "Padding not zeroed at W=" << W
              << ", three_level=" << three_level << ", p_idx=" << p_idx;
        }
      }

      std::vector<uint16_t> dst_linear(W, 0);
      ASSERT_TRUE(DetileBuffer(dst_tiled.data(),
                               reinterpret_cast<uint8_t*>(dst_linear.data()),
                               shape, shape.layout())
                      .ok());
      EXPECT_EQ(dst_linear, src_linear);
    }
  }

  // Also test S8 with P=4 ({0:T(1024)(128)(4,1)} and {0:T(128)(4,1)}) and
  // 2-level unpacked {0:T(1024)(128)}.
  for (int64_t W : {300, 2560}) {
    for (bool three_level_s8 : {false, true}) {
      xla::Shape shape_s8 =
          three_level_s8
              ? xla::ShapeUtil::MakeShapeWithDenseLayout(
                    xla::PrimitiveType::S8, {W}, {0},
                    {xla::Tile({1024}), xla::Tile({128}), xla::Tile({4, 1})})
              : xla::ShapeUtil::MakeShapeWithDenseLayout(
                    xla::PrimitiveType::S8, {W}, {0},
                    {xla::Tile({128}), xla::Tile({4, 1})});
      ASSERT_TRUE(IsStandardRowMajorTiled(shape_s8, shape_s8.layout()));
      std::vector<int8_t> src_s8(W);
      for (int64_t i = 0; i < W; ++i) {
        src_s8[i] = static_cast<int8_t>((i % 120) + 1);
      }
      int64_t phys_s8 = GetTiledBufferElements(shape_s8);
      std::vector<uint8_t> tiled_s8(phys_s8, 0xFF);
      ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_s8.data()),
                             tiled_s8.data(), shape_s8, shape_s8.layout())
                      .ok());
      for (int64_t i = 0; i < W; ++i) {
        int64_t p_idx =
            xla::LayoutUtil::LinearIndexForNestedTiling(shape_s8, {i});
        EXPECT_EQ(reinterpret_cast<const int8_t*>(tiled_s8.data())[p_idx],
                  src_s8[i]);
      }
      std::vector<int8_t> dst_s8(W, 0);
      ASSERT_TRUE(DetileBuffer(tiled_s8.data(),
                               reinterpret_cast<uint8_t*>(dst_s8.data()),
                               shape_s8, shape_s8.layout())
                      .ok());
      EXPECT_EQ(dst_s8, src_s8);
    }

    xla::Shape shape_unpacked = xla::ShapeUtil::MakeShapeWithDenseLayout(
        xla::PrimitiveType::BF16, {W}, {0},
        {xla::Tile({1024}), xla::Tile({128})});
    ASSERT_TRUE(
        IsStandardRowMajorTiled(shape_unpacked, shape_unpacked.layout()));
    std::vector<uint16_t> src_u16(W);
    for (int64_t i = 0; i < W; ++i) {
      src_u16[i] = static_cast<uint16_t>(i * 5 + 3);
    }
    int64_t phys_u16 = GetTiledBufferElements(shape_unpacked);
    std::vector<uint8_t> tiled_u16(phys_u16 * sizeof(uint16_t), 0xFF);
    ASSERT_TRUE(TileBuffer(reinterpret_cast<const uint8_t*>(src_u16.data()),
                           tiled_u16.data(), shape_unpacked,
                           shape_unpacked.layout())
                    .ok());
    const uint16_t* tiled_u16_ptr =
        reinterpret_cast<const uint16_t*>(tiled_u16.data());
    for (int64_t i = 0; i < W; ++i) {
      int64_t p_idx =
          xla::LayoutUtil::LinearIndexForNestedTiling(shape_unpacked, {i});
      EXPECT_EQ(tiled_u16_ptr[p_idx], src_u16[i]);
    }
    for (int64_t p_idx = W; p_idx < phys_u16; ++p_idx) {
      EXPECT_EQ(tiled_u16_ptr[p_idx], 0);
    }
    std::vector<uint16_t> dst_u16(W, 0);
    ASSERT_TRUE(DetileBuffer(tiled_u16.data(),
                             reinterpret_cast<uint8_t*>(dst_u16.data()),
                             shape_unpacked, shape_unpacked.layout())
                    .ok());
    EXPECT_EQ(dst_u16, src_u16);
  }
}

TEST(TilingUtilsStandaloneTest, InPlaceTilingMatchesTileBuffer) {
  auto test_shape = [](const xla::Shape& shape) {
    int64_t linear_elements = xla::ShapeUtil::ElementsIn(shape);
    int64_t itemsize =
        xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    int64_t physical_elements = GetTiledBufferElements(shape);
    int64_t linear_bytes = linear_elements * itemsize;
    int64_t physical_bytes = physical_elements * itemsize;

    std::vector<uint8_t> src_linear(linear_bytes);
    for (size_t i = 0; i < src_linear.size(); ++i) {
      src_linear[i] = static_cast<uint8_t>((i * 17 + 3) & 0xFF);
    }

    // 1. Run out-of-place TileBuffer
    std::vector<uint8_t> expected_tiled(physical_bytes, 0);
    ASSERT_TRUE(TileBuffer(src_linear.data(), expected_tiled.data(), shape,
                           shape.layout())
                    .ok());

    // 2. Run TileBufferInPlace
    std::vector<uint8_t> in_place_buf(physical_bytes, 0);
    std::memcpy(in_place_buf.data(), src_linear.data(), linear_bytes);
    ASSERT_TRUE(TileBufferInPlace(in_place_buf.data(), in_place_buf.size(),
                                  shape, shape.layout())
                    .ok());

    // Verify byte-for-byte identity with TileBuffer
    EXPECT_EQ(in_place_buf, expected_tiled);

    // 3. Verify DetileBuffer restores original linear bytes
    std::vector<uint8_t> detiled(linear_bytes, 0);
    ASSERT_TRUE(
        DetileBuffer(in_place_buf.data(), detiled.data(), shape, shape.layout())
            .ok());
    EXPECT_EQ(detiled, src_linear);

    // 4. Verify insufficient capacity returns error
    if (physical_bytes > 1) {
      std::vector<uint8_t> small_buf(physical_bytes - 1, 0);
      EXPECT_FALSE(TileBufferInPlace(small_buf.data(), small_buf.size(), shape,
                                     shape.layout())
                       .ok());
    }
  };

  // Case 1: Standard 2D no padding (F32, tile 8x8)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {32, 64}, {1, 0}, {xla::Tile({8, 8})}));

  // Case 2: Standard 2D with padding (BF16, tile 8x128, packing 2,1)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {65, 130}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})}));

  // Case 3: Standard 2D no padding (BF16, tile 8x128, packing 2,1)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {64, 256}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})}));

  // Case 4: Standard 3D batch with padding (BF16, tile 8x128, packing 2,1)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {2, 65, 130}, {2, 1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})}));

  // Case 5: Standard 3D batch no padding (BF16, tile 8x128, packing 2,1)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {3, 32, 128}, {2, 1, 0},
      {xla::Tile({8, 128}), xla::Tile({2, 1})}));

  // Case 6: FP8 / S8 packing factor 4 with padding (tile 8x128, packing 4,1)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::S8, {33, 135}, {1, 0},
      {xla::Tile({8, 128}), xla::Tile({4, 1})}));

  // Case 7: 1D unpacked tiling with padding (F32, tile 128)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::F32, {250}, {0}, {xla::Tile({128})}));

  // Case 8: 1D packed tiling with padding (BF16, tile 128, packing 2,1)
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {250}, {0},
      {xla::Tile({128}), xla::Tile({2, 1})}));

  // Case 9: Column-major fallback (BF16, minor_to_major {0, 1})
  test_shape(xla::ShapeUtil::MakeShapeWithDenseLayout(
      xla::PrimitiveType::BF16, {65, 130}, {0, 1},
      {xla::Tile({8, 128}), xla::Tile({2, 1})}));
}

// -----------------------------------------------------------------------------
// Property tests against an independent reference implementation of the XLA
// tiling definition (https://openxla.org/xla/tiled_layout).
// -----------------------------------------------------------------------------

// Upper bound on the rank of the tiled index space built by the reference
// (logical rank 4 plus two levels of rank-2 tiles, or three 1D/2D tiles).
constexpr int kMaxReferenceTiledRank = 16;

int64_t ReferenceCeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Computes, from first principles, the physical element offset of logical
// element |index| of |shape| in a buffer tiled per |shape|'s layout, and the
// total number of physical elements of that buffer. Deliberately independent
// of both tiling_utils and xla::LayoutUtil:
//   1. Order the logical dimensions major-to-minor using minor_to_major.
//   2. For each tile T = (t_0, ..., t_{k-1}), left-pad the shape with size-1
//      dimensions up to rank k if needed, then replace the k most-minor
//      dimensions (d_0, ..., d_{k-1}) by (ceil(d_0/t_0), ..., ceil(d_{k-1}/
//      t_{k-1}), t_0, ..., t_{k-1}) and an index (e_0, ..., e_{k-1}) by
//      (e_0/t_0, ..., e_{k-1}/t_{k-1}, e_0%t_0, ..., e_{k-1}%t_{k-1}).
//   3. Linearize the final index row-major over the final shape.
int64_t ReferenceTiledOffset(const xla::Shape& shape,
                             absl::Span<const int64_t> index,
                             int64_t* num_physical_elements) {
  std::array<int64_t, kMaxReferenceTiledRank> dims;
  std::array<int64_t, kMaxReferenceTiledRank> idx;
  int rank = shape.dimensions().size();
  for (int i = 0; i < rank; ++i) {
    const int64_t logical_dim = shape.layout().minor_to_major(rank - 1 - i);
    dims[i] = shape.dimensions(logical_dim);
    idx[i] = index[logical_dim];
  }
  for (const xla::Tile& tile : shape.layout().tiles()) {
    const int k = tile.dimensions().size();
    if (k > rank) {
      const int pad = k - rank;
      for (int i = rank - 1; i >= 0; --i) {
        dims[i + pad] = dims[i];
        idx[i + pad] = idx[i];
      }
      for (int i = 0; i < pad; ++i) {
        dims[i] = 1;
        idx[i] = 0;
      }
      rank = k;
    }
    CHECK_LE(rank + k, kMaxReferenceTiledRank);
    const int prefix = rank - k;
    for (int j = k - 1; j >= 0; --j) {
      const int64_t t = tile.dimension(j);
      const int64_t d = dims[prefix + j];
      const int64_t e = idx[prefix + j];
      dims[prefix + k + j] = t;
      idx[prefix + k + j] = e % t;
      dims[prefix + j] = ReferenceCeilDiv(d, t);
      idx[prefix + j] = e / t;
    }
    rank += k;
  }
  int64_t offset = 0;
  int64_t total = 1;
  for (int i = 0; i < rank; ++i) {
    offset = offset * dims[i] + idx[i];
    total *= dims[i];
  }
  *num_physical_elements = total;
  return offset;
}

// The reference physical element offset of every logical element, indexed by
// the element's row-major logical index.
struct ReferenceMapping {
  std::vector<int64_t> physical_offsets;
  int64_t num_physical_elements = 0;
};

// Calls |fn(linear_index, multi_index)| for every logical element of |dims| in
// row-major order.
template <typename Fn>
void ForEachRowMajorIndex(absl::Span<const int64_t> dims, Fn&& fn) {
  int64_t num_elements = 1;
  for (int64_t d : dims) num_elements *= d;
  std::vector<int64_t> index(dims.size(), 0);
  for (int64_t linear = 0; linear < num_elements; ++linear) {
    fn(linear, absl::Span<const int64_t>(index));
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      if (++index[i] < dims[i]) break;
      index[i] = 0;
    }
  }
}

ReferenceMapping ComputeReferenceMapping(const xla::Shape& shape) {
  ReferenceMapping mapping;
  mapping.physical_offsets.resize(xla::ShapeUtil::ElementsIn(shape));
  ForEachRowMajorIndex(
      shape.dimensions(), [&](int64_t linear, absl::Span<const int64_t> index) {
        mapping.physical_offsets[linear] =
            ReferenceTiledOffset(shape, index, &mapping.num_physical_elements);
      });
  if (mapping.physical_offsets.empty()) {
    std::vector<int64_t> zero_index(shape.dimensions().size(), 0);
    ReferenceTiledOffset(shape, zero_index, &mapping.num_physical_elements);
  }
  return mapping;
}

uint64_t SplitMix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Fills |num_elements| elements of |itemsize| bytes with the low bytes of a
// hash of the element index, so that (up to hash collisions for itemsize 1)
// every element is distinguishable from its neighbours.
std::vector<uint8_t> MakeHashedLinearBuffer(int64_t num_elements,
                                            int64_t itemsize) {
  std::vector<uint8_t> buffer(num_elements * itemsize);
  for (int64_t i = 0; i < num_elements; ++i) {
    const uint64_t h = SplitMix64(static_cast<uint64_t>(i));
    std::memcpy(buffer.data() + i * itemsize, &h, itemsize);
  }
  return buffer;
}

std::string BytesToHex(const uint8_t* bytes, int64_t n) {
  std::string out = "0x";
  for (int64_t i = n - 1; i >= 0; --i) {
    absl::StrAppend(&out, absl::Hex(bytes[i], absl::kZeroPad2));
  }
  return out;
}

// Returns a description of the first logical element whose bytes in |actual|
// differ from |expected| at its reference physical offset, or "" if all
// logical elements match. Padding positions are ignored.
std::string FirstTiledMismatch(const xla::Shape& shape,
                               const ReferenceMapping& mapping,
                               int64_t itemsize,
                               absl::Span<const uint8_t> actual,
                               absl::Span<const uint8_t> expected) {
  std::string first;
  int64_t num_mismatches = 0;
  ForEachRowMajorIndex(
      shape.dimensions(), [&](int64_t linear, absl::Span<const int64_t> index) {
        const int64_t byte_offset = mapping.physical_offsets[linear] * itemsize;
        if (std::memcmp(actual.data() + byte_offset,
                        expected.data() + byte_offset, itemsize) == 0) {
          return;
        }
        if (num_mismatches++ == 0) {
          first = absl::StrCat(
              "logical index (", absl::StrJoin(index, ","),
              ") -> physical byte offset ", byte_offset, ": expected ",
              BytesToHex(expected.data() + byte_offset, itemsize), ", got ",
              BytesToHex(actual.data() + byte_offset, itemsize));
        }
      });
  if (num_mismatches == 0) return "";
  return absl::StrCat(xla::ShapeUtil::HumanStringWithLayout(shape), ": ",
                      num_mismatches, " mismatching elements; first: ", first);
}

// Returns a description of the first differing byte of two linear buffers, or
// "" if they are identical.
std::string FirstLinearMismatch(const xla::Shape& shape,
                                absl::Span<const uint8_t> actual,
                                absl::Span<const uint8_t> expected) {
  if (actual.size() != expected.size()) {
    return absl::StrCat("size ", actual.size(), " vs ", expected.size());
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      return absl::StrCat(xla::ShapeUtil::HumanStringWithLayout(shape),
                          ": first differing linear byte offset ", i,
                          ": expected ", BytesToHex(&expected[i], 1), ", got ",
                          BytesToHex(&actual[i], 1));
    }
  }
  return "";
}

// Checks TileBuffer, TileBufferInPlace, DetileBuffer and GetTiledBufferElements
// for |shape| against the reference mapping:
//   (a) DetileBuffer(TileBuffer(x)) == x byte-for-byte.
//   (b) TileBuffer(x) and TileBufferInPlace(x) place every logical element at
//       its reference physical offset (padding ignored).
//   (c) DetileBuffer of the reference tiled buffer, with garbage in padding,
//       reproduces x.
// When |cross_check_xla| is set, the reference mapping itself is also checked
// against xla::LayoutUtil::LinearIndexForNestedTiling (slow for large shapes).
void CheckAgainstReference(const xla::Shape& shape,
                           bool cross_check_xla = true) {
  SCOPED_TRACE(xla::ShapeUtil::HumanStringWithLayout(shape));
  const xla::Layout& layout = shape.layout();
  const int64_t itemsize =
      xla::ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
  const int64_t num_elements = xla::ShapeUtil::ElementsIn(shape);

  const ReferenceMapping mapping = ComputeReferenceMapping(shape);
  const int64_t physical_bytes = mapping.num_physical_elements * itemsize;
  EXPECT_EQ(GetTiledBufferElements(shape), mapping.num_physical_elements);

  // The reference must be an injection into the physical buffer.
  std::vector<bool> seen(mapping.num_physical_elements, false);
  for (int64_t offset : mapping.physical_offsets) {
    ASSERT_GE(offset, 0);
    ASSERT_LT(offset, mapping.num_physical_elements);
    ASSERT_FALSE(seen[offset]) << "duplicate physical offset " << offset;
    seen[offset] = true;
  }

  // Cross-check the reference against XLA's own nested tiling index function.
  if (cross_check_xla) {
    int64_t xla_disagreements = 0;
    ForEachRowMajorIndex(
        shape.dimensions(),
        [&](int64_t linear, absl::Span<const int64_t> index) {
          if (xla::LayoutUtil::LinearIndexForNestedTiling(shape, index) !=
              mapping.physical_offsets[linear]) {
            ++xla_disagreements;
          }
        });
    EXPECT_EQ(xla_disagreements, 0);
  }

  const std::vector<uint8_t> linear =
      MakeHashedLinearBuffer(num_elements, itemsize);
  std::vector<uint8_t> expected_tiled(physical_bytes, 0x5A);
  for (int64_t i = 0; i < num_elements; ++i) {
    std::memcpy(expected_tiled.data() + mapping.physical_offsets[i] * itemsize,
                linear.data() + i * itemsize, itemsize);
  }

  // (b) Out-of-place tiling.
  std::vector<uint8_t> tiled(physical_bytes, 0xA5);
  ABSL_ASSERT_OK(TileBuffer(linear.data(), tiled.data(), shape, layout));
  EXPECT_EQ(FirstTiledMismatch(shape, mapping, itemsize, tiled, expected_tiled),
            "")
      << "TileBuffer";

  // (b) In-place tiling into an exactly-sized buffer.
  std::vector<uint8_t> in_place(physical_bytes, 0xA5);
  std::memcpy(in_place.data(), linear.data(), linear.size());
  ABSL_ASSERT_OK(
      TileBufferInPlace(in_place.data(), in_place.size(), shape, layout));
  EXPECT_EQ(
      FirstTiledMismatch(shape, mapping, itemsize, in_place, expected_tiled),
      "")
      << "TileBufferInPlace";

  // (a) Round trip.
  std::vector<uint8_t> round_trip(linear.size(), 0xC3);
  ABSL_ASSERT_OK(DetileBuffer(tiled.data(), round_trip.data(), shape, layout));
  EXPECT_EQ(FirstLinearMismatch(shape, round_trip, linear), "")
      << "DetileBuffer(TileBuffer(x))";

  // (c) Detiling the reference buffer.
  std::vector<uint8_t> detiled(linear.size(), 0xC3);
  ABSL_ASSERT_OK(
      DetileBuffer(expected_tiled.data(), detiled.data(), shape, layout));
  EXPECT_EQ(FirstLinearMismatch(shape, detiled, linear), "")
      << "DetileBuffer(reference)";
}

enum class DimOrder {
  // minor_to_major = {R-1, ..., 0}: the row-major fast path.
  kRowMajor,
  // minor_to_major = {R-2, R-1, R-3, ..., 0}: the column-major fast path.
  kColMajor,
  // minor_to_major = {0, 1, ..., R-1} (rank >= 3): the generic fallback path.
  kReversed,
};

std::vector<int64_t> MinorToMajor(DimOrder order, int rank) {
  std::vector<int64_t> minor_to_major(rank);
  for (int i = 0; i < rank; ++i) {
    minor_to_major[i] = (order == DimOrder::kReversed) ? i : rank - 1 - i;
  }
  if (order == DimOrder::kColMajor) {
    std::swap(minor_to_major[0], minor_to_major[1]);
  }
  return minor_to_major;
}

// A TPU-style tiled layout applied to every shape whose rank lies in
// [min_rank, max_rank].
struct TiledLayoutFamily {
  std::string name;
  xla::PrimitiveType type;
  DimOrder order;
  std::vector<xla::Tile> tiles;
  int min_rank;
  int max_rank;
};

std::vector<std::vector<int64_t>> ReferenceTestShapes() {
  // Shapes that are not multiples of any tile, of ranks 1-4.
  std::vector<std::vector<int64_t>> shapes = {
      {1000},   {1},         {4095},      {130, 257},      {1, 1},
      {9, 129}, {3, 100, 7}, {5, 1, 300}, {2, 3, 17, 130}, {1, 2, 33, 1},
  };
  // Deterministic pseudo-random shapes. Each dimension is either tiny, close
  // to a multiple of a common tile size, or uniform in [1, 300].
  std::mt19937_64 rng(0x7111E5EEDull);
  while (shapes.size() < 18) {
    const int rank = 1 + static_cast<int>(rng() % 4);
    std::vector<int64_t> dims(rank);
    int64_t num_elements = 1;
    for (int64_t& d : dims) {
      switch (rng() % 3) {
        case 0:
          d = 1 + static_cast<int64_t>(rng() % 4);
          break;
        case 1: {
          constexpr int64_t kTileSizes[] = {8, 16, 32, 128};
          const int64_t t = kTileSizes[rng() % 4];
          d = std::max<int64_t>(1, t * (1 + static_cast<int64_t>(rng() % 3)) -
                                       1 + static_cast<int64_t>(rng() % 3));
          break;
        }
        default:
          d = 1 + static_cast<int64_t>(rng() % 300);
      }
      num_elements *= d;
    }
    if (num_elements <= 60000) shapes.push_back(std::move(dims));
  }
  return shapes;
}

class TilingReferenceTest : public ::testing::TestWithParam<TiledLayoutFamily> {
};

TEST_P(TilingReferenceTest, MatchesReferenceAndRoundTrips) {
  const TiledLayoutFamily& family = GetParam();
  int num_checked = 0;
  for (const std::vector<int64_t>& dims : ReferenceTestShapes()) {
    const int rank = dims.size();
    if (rank < family.min_rank || rank > family.max_rank) continue;
    CheckAgainstReference(xla::ShapeUtil::MakeShapeWithDenseLayout(
        family.type, dims, MinorToMajor(family.order, rank), family.tiles));
    ++num_checked;
  }
  EXPECT_GT(num_checked, 0);
}

TiledLayoutFamily Family(std::string name, xla::PrimitiveType type,
                         DimOrder order, std::vector<xla::Tile> tiles,
                         int min_rank, int max_rank) {
  return {std::move(name), type, order, std::move(tiles), min_rank, max_rank};
}

INSTANTIATE_TEST_SUITE_P(
    TpuLayouts, TilingReferenceTest,
    ::testing::ValuesIn(std::vector<TiledLayoutFamily>{
        // itemsize 4.
        Family("F32_T8x128_RowMajor", xla::F32, DimOrder::kRowMajor,
               {xla::Tile({8, 128})}, /*min_rank=*/1, /*max_rank=*/4),
        Family("F32_T8x128_ColMajor", xla::F32, DimOrder::kColMajor,
               {xla::Tile({8, 128})}, /*min_rank=*/2, /*max_rank=*/4),
        Family("F32_T8x128_Reversed", xla::F32, DimOrder::kReversed,
               {xla::Tile({8, 128})}, /*min_rank=*/3, /*max_rank=*/4),
        Family("F32_T1024_1D", xla::F32, DimOrder::kRowMajor,
               {xla::Tile({1024})}, /*min_rank=*/1, /*max_rank=*/1),
        Family("F32_T256_1D", xla::F32, DimOrder::kRowMajor, {xla::Tile({256})},
               /*min_rank=*/1, /*max_rank=*/1),
        Family("F32_T1024_T128_1D", xla::F32, DimOrder::kRowMajor,
               {xla::Tile({1024}), xla::Tile({128})}, /*min_rank=*/1,
               /*max_rank=*/1),
        // itemsize 2.
        Family("BF16_T8x128_2x1_RowMajor", xla::BF16, DimOrder::kRowMajor,
               {xla::Tile({8, 128}), xla::Tile({2, 1})}, /*min_rank=*/1,
               /*max_rank=*/4),
        Family("BF16_T16x128_2x1_RowMajor", xla::BF16, DimOrder::kRowMajor,
               {xla::Tile({16, 128}), xla::Tile({2, 1})}, /*min_rank=*/1,
               /*max_rank=*/4),
        Family("BF16_T8x128_RowMajor", xla::BF16, DimOrder::kRowMajor,
               {xla::Tile({8, 128})}, /*min_rank=*/1, /*max_rank=*/4),
        Family("BF16_T8x128_2x1_ColMajor", xla::BF16, DimOrder::kColMajor,
               {xla::Tile({8, 128}), xla::Tile({2, 1})}, /*min_rank=*/2,
               /*max_rank=*/4),
        Family("BF16_T8x128_2x1_Reversed", xla::BF16, DimOrder::kReversed,
               {xla::Tile({8, 128}), xla::Tile({2, 1})}, /*min_rank=*/3,
               /*max_rank=*/4),
        Family("BF16_T1024_T128_2x1_1D", xla::BF16, DimOrder::kRowMajor,
               {xla::Tile({1024}), xla::Tile({128}), xla::Tile({2, 1})},
               /*min_rank=*/1, /*max_rank=*/1),
        Family("BF16_T128_2x1_1D", xla::BF16, DimOrder::kRowMajor,
               {xla::Tile({128}), xla::Tile({2, 1})}, /*min_rank=*/1,
               /*max_rank=*/1),
        // itemsize 1.
        Family("S8_T8x128_4x1_RowMajor", xla::S8, DimOrder::kRowMajor,
               {xla::Tile({8, 128}), xla::Tile({4, 1})}, /*min_rank=*/1,
               /*max_rank=*/4),
        Family("S8_T32x128_4x1_RowMajor", xla::S8, DimOrder::kRowMajor,
               {xla::Tile({32, 128}), xla::Tile({4, 1})}, /*min_rank=*/1,
               /*max_rank=*/4),
        Family("S8_T8x128_4x1_ColMajor", xla::S8, DimOrder::kColMajor,
               {xla::Tile({8, 128}), xla::Tile({4, 1})}, /*min_rank=*/2,
               /*max_rank=*/4),
        Family("S8_T32x128_4x1_Reversed", xla::S8, DimOrder::kReversed,
               {xla::Tile({32, 128}), xla::Tile({4, 1})}, /*min_rank=*/3,
               /*max_rank=*/4),
        Family("S8_T1024_T128_4x1_1D", xla::S8, DimOrder::kRowMajor,
               {xla::Tile({1024}), xla::Tile({128}), xla::Tile({4, 1})},
               /*min_rank=*/1, /*max_rank=*/1),
        Family("S8_T128_4x1_1D", xla::S8, DimOrder::kRowMajor,
               {xla::Tile({128}), xla::Tile({4, 1})}, /*min_rank=*/1,
               /*max_rank=*/1),
    }),
    [](const ::testing::TestParamInfo<TiledLayoutFamily>& info) {
      return info.param.name;
    });

// Tensors of at least 8 MB are tiled and detiled on the shared thread pool;
// check that path against the reference too, with padding in both tiled
// dimensions. The XLA cross-check is skipped here to keep the runtime small.
TEST(TilingReferenceLargeTest, ParallelPathMatchesReference) {
  CheckAgainstReference(xla::ShapeUtil::MakeShapeWithDenseLayout(
                            xla::BF16, {2049, 2100}, {1, 0},
                            {xla::Tile({8, 128}), xla::Tile({2, 1})}),
                        /*cross_check_xla=*/false);
  CheckAgainstReference(xla::ShapeUtil::MakeShapeWithDenseLayout(
                            xla::BF16, {2100, 2049}, {0, 1},
                            {xla::Tile({8, 128}), xla::Tile({2, 1})}),
                        /*cross_check_xla=*/false);
}

}  // namespace
}  // namespace tpu_raiden::weight_sync
