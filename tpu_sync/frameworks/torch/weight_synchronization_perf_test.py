# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-machine perf regression test for PyTorch weight synchronization."""

import asyncio
import time
from typing import Callable

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
import torch_tpu  # pylint: disable=unused-import

from tpu_sync.api.torch import weight_synchronizer
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

_NUM_LAYERS = flags.DEFINE_integer(
    "num_decoder_layers",
    4,
    "Number of Qwen 3.5 35B decoder layers to benchmark (default 4 = 1 full"
    " cycle of 3 GDN + 1 Full Attn).",
)
_BENCHMARK_ITERATIONS = flags.DEFINE_integer(
    "benchmark_iterations",
    3,
    "Number of timed benchmark iterations.",
)
_GROUP_SIZE = flags.DEFINE_integer(
    "group_size",
    70,
    "Number of weights to group per transfer request.",
)
_PARALLELISM = flags.DEFINE_integer(
    "parallelism",
    16,
    "Number of parallel TCP stream worker threads for H2H.",
)
_DTYPE = flags.DEFINE_string(
    "dtype",
    "bfloat16",
    "Data type for synthetic weight tensors (bfloat16 or float32).",
)


def _resolve_torch_dtype(dtype_str: str) -> tuple[torch.dtype, int]:
  """Maps flag string to (torch.dtype, element_byte_size)."""
  if dtype_str == "bfloat16":
    return torch.bfloat16, 2
  elif dtype_str == "float32":
    return torch.float32, 4
  raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_qwen3_5_35b_specs(
    num_layers: int,
    role: str = "source",
) -> list[tuple[tuple[int, ...], list[str], str]]:
  """Generates parameter specs for Qwen 3.5 35B in standard PyTree leaf order."""
  dim = 2048
  num_routed_experts = 256
  routed_mlp_dim = 512
  shared_mlp_dim = 512
  full_q_dim = 8192
  linear_qkvz_dim = 12288
  linear_ba_dim = 64
  linear_conv_dim = 8192
  linear_a_log_dim = 32
  linear_rms_dim = 128
  linear_out_dim = 4096

  specs = []
  is_dest = role == "destination"

  # 1. Final decoder layer norm
  specs.append((
      (dim,),
      [""] if is_dest else ["fsdp"],
      "decoder.decoder_norm.scale",
  ))

  # 2. Decoder layers (hybrid cycle: 3 GDN linear attn + 1 full GQA attn)
  for l in range(num_layers):
    is_full_attn = (l + 1) % 4 == 0

    if is_full_attn:
      specs.append((
          (dim, 2, 256),
          ["", "tp", ""] if is_dest else ["fsdp", "", ""],
          f"decoder.layers.{l}.attention.attention.key.kernel",
      ))
      specs.append((
          (256,),
          [""],
          f"decoder.layers.{l}.attention.attention.key_norm.scale",
      ))
      specs.append((
          (dim * 2, dim),
          ["tp", ""] if is_dest else ["tp", "fsdp"],
          f"decoder.layers.{l}.attention.attention.out.kernel",
      ))
      specs.append((
          (dim, full_q_dim // 512, 512),
          ["", "tp", ""] if is_dest else ["fsdp", "tp", ""],
          f"decoder.layers.{l}.attention.attention.query.kernel",
      ))
      specs.append((
          (256,),
          [""],
          f"decoder.layers.{l}.attention.attention.query_norm.scale",
      ))
      specs.append((
          (dim, 2, 256),
          ["", "tp", ""] if is_dest else ["fsdp", "", ""],
          f"decoder.layers.{l}.attention.attention.value.kernel",
      ))
    else:
      specs.append((
          (linear_a_log_dim,),
          [""],
          f"decoder.layers.{l}.attention.A_log",
      ))
      specs.append((
          (4, 1, linear_conv_dim),
          ["", "", "tp"] if is_dest else ["", "", ""],
          f"decoder.layers.{l}.attention.conv1d.kernel",
      ))
      specs.append((
          (linear_a_log_dim,),
          [""],
          f"decoder.layers.{l}.attention.dt_bias",
      ))
      specs.append((
          (dim, linear_ba_dim),
          ["", "tp"] if is_dest else ["fsdp", "tp"],
          f"decoder.layers.{l}.attention.in_proj_ba.kernel",
      ))
      specs.append((
          (dim, linear_qkvz_dim),
          ["", "tp"] if is_dest else ["fsdp", "tp"],
          f"decoder.layers.{l}.attention.in_proj_qkvz.kernel",
      ))
      specs.append((
          (linear_rms_dim,),
          [""],
          f"decoder.layers.{l}.attention.norm.rms_norm.scale",
      ))
      specs.append((
          (linear_out_dim, dim),
          ["tp", ""] if is_dest else ["tp", "fsdp"],
          f"decoder.layers.{l}.attention.out_proj.kernel",
      ))

    # Input layernorm
    specs.append((
        (dim,),
        [""] if is_dest else ["fsdp"],
        f"decoder.layers.{l}.input_layernorm.scale",
    ))

    # Sparse MoE block
    specs.append((
        (dim, num_routed_experts),
        ["", ""] if is_dest else ["fsdp", "tp"],
        f"decoder.layers.{l}.mlp.routed_experts.gate.kernel",
    ))
    specs.append((
        (num_routed_experts, dim, routed_mlp_dim),
        ["fsdp", "", "tp"],
        f"decoder.layers.{l}.mlp.routed_experts.wi_0",
    ))
    specs.append((
        (num_routed_experts, dim, routed_mlp_dim),
        ["fsdp", "", "tp"],
        f"decoder.layers.{l}.mlp.routed_experts.wi_1",
    ))
    specs.append((
        (num_routed_experts, routed_mlp_dim, dim),
        ["fsdp", "tp", ""],
        f"decoder.layers.{l}.mlp.routed_experts.wo",
    ))

    # Shared expert
    specs.append((
        (dim, shared_mlp_dim),
        ["", "tp"] if is_dest else ["fsdp", "tp"],
        f"decoder.layers.{l}.mlp.shared_expert.wi_0.kernel",
    ))
    specs.append((
        (dim, shared_mlp_dim),
        ["", "tp"] if is_dest else ["fsdp", "tp"],
        f"decoder.layers.{l}.mlp.shared_expert.wi_1.kernel",
    ))
    specs.append((
        (shared_mlp_dim, dim),
        ["tp", ""] if is_dest else ["tp", "fsdp"],
        f"decoder.layers.{l}.mlp.shared_expert.wo.kernel",
    ))
    specs.append((
        (dim, 1),
        ["", ""] if is_dest else ["fsdp", ""],
        f"decoder.layers.{l}.mlp.shared_expert_gate.kernel",
    ))

    # Post-attention layernorm
    specs.append((
        (dim,),
        [""] if is_dest else ["fsdp"],
        f"decoder.layers.{l}.post_attention_layernorm.scale",
    ))

  # 3. Output vocab projection and token embedder
  specs.append((
      (dim, 248320),
      ["", "fsdp"] if is_dest else ["fsdp", "tp"],
      "decoder.logits_dense.kernel",
  ))
  specs.append((
      (248320, dim),
      ["fsdp", ""] if is_dest else ["tp", "fsdp"],
      "token_embedder.embedding",
  ))
  return specs


def _allocate_model_tensors_and_metadata(
    specs: list[tuple[tuple[int, ...], list[str], str]],
    device: torch.device,
    dtype: torch.dtype,
    item_size: int,
    generator_fn: Callable[..., torch.Tensor],
) -> tuple[
    list[list[torch.Tensor]],
    list[raiden_service_pb2.VariableMetadataProto],
    int,
]:
  """Allocates single-shard TPU tensors and builds controller metadata protos."""
  tensors = []
  variable_protos = []
  total_bytes = 0

  for idx, (global_shape, spec_axes, name) in enumerate(specs):
    t = generator_fn(global_shape, dtype=dtype, device=device)
    tensors.append([t])

    num_elements = int(np.prod(global_shape))
    total_bytes += num_elements * item_size

    sharding_shape = [1] * len(spec_axes)
    layout = list(range(len(global_shape) - 1, -1, -1))
    variable_protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=name,
            shape=global_shape,
            mesh_shape=sharding_shape,
            layout=layout,
            item_size=item_size,
            layer_idx=idx,
            sharding_spec=spec_axes,
        )
    )

  return tensors, variable_protos, total_bytes


def _compute_skip_tiling_map(
    specs: list[tuple[tuple[int, ...], list[str], str]],
) -> dict[int, bool]:
  """Returns per-layer skip_tiling flags based on TPU (8, 128) tile alignment.

  When a tensor's minor dimensions are multiples of the TPU hardware tile shape
  (8, 128), physical tiled HBM byte size equals logical byte size, allowing
  zero-copy raw DMA without CPU tiling/detiling. Unaligned shapes must use
  standard tiling/detiling so logical byte transfers match HBM layout.

  Args:
    specs: List of (shape, sharding_spec, name) parameter descriptors.
  """
  skip_tiling = {}
  for idx, (shape, _, _) in enumerate(specs):
    if len(shape) >= 2 and shape[-2] % 8 == 0 and shape[-1] % 128 == 0:
      skip_tiling[idx] = True
    else:
      skip_tiling[idx] = False
  return skip_tiling


def _run_controller_transfer(
    controller: raiden_controller.RaidenController,
    src_unit: raiden_controller.RaidenId,
    dst_unit: raiden_controller.RaidenId,
    uuid: int,
    req_id: str,
    group_size: int,
    parallelism: int,
    skip_tiling: dict[int, bool],
    skip_d2h: bool = True,
) -> None:
  """Executes a synchronous controller-orchestrated H2H transfer."""
  future = controller.start_transfer(
      src_units=[src_unit],
      dst_units=[dst_unit],
      dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
      use_block_chunks=True,
      is_sender=True,
      expected_block_count=0,
      uuid=uuid,
      req_id=req_id,
      group_size=group_size,
      parallelism=parallelism,
      skip_tiling=skip_tiling,
      skip_d2h=skip_d2h,
  )
  loop = asyncio.new_event_loop()
  try:
    loop.run_until_complete(future.wait())
  finally:
    loop.close()


class WeightSynchronizationPerfTest(parameterized.TestCase):

  def _verify_tensor_parity(
      self,
      src_tensors: list[list[torch.Tensor]],
      dst_tensors: list[list[torch.Tensor]],
  ):
    """Verifies bit-level numerical equality between source and destination."""
    for l in range(len(src_tensors)):
      for sh in range(len(src_tensors[l])):
        self.assertTrue(
            torch.equal(dst_tensors[l][sh].cpu(), src_tensors[l][sh].cpu()),
            f"Data mismatch at layer {l}, shard {sh}",
        )

  def test_model_specs(self):
    specs = get_qwen3_5_35b_specs(num_layers=4)
    # 3 GDN layers * 17 tensors + 1 Full Attn layer * 16 tensors + 3 global = 70
    self.assertLen(specs, 70)

  def test_weight_synchronization_perf(self):
    src_device = torch.device("tpu:0")
    try:
      num_tpus = torch.tpu.device_count()
    except AttributeError:
      num_tpus = 1
    dst_device = torch.device("tpu:1") if num_tpus > 1 else src_device

    num_layers = _NUM_LAYERS.value
    num_iters = _BENCHMARK_ITERATIONS.value
    group_size = _GROUP_SIZE.value
    parallelism = _PARALLELISM.value
    torch_dtype, item_size = _resolve_torch_dtype(_DTYPE.value)

    logging.info(
        "Allocating Qwen 3.5 35B weight tensors (layers=%d, dtype=%s,"
        " src_device=%s, dst_device=%s)...",
        num_layers,
        _DTYPE.value,
        src_device,
        dst_device,
    )

    torch.manual_seed(42)
    src_specs = get_qwen3_5_35b_specs(num_layers=num_layers, role="source")
    dst_specs = get_qwen3_5_35b_specs(num_layers=num_layers, role="destination")

    src_tensors, src_protos, total_bytes = _allocate_model_tensors_and_metadata(
        src_specs,
        device=src_device,
        dtype=torch_dtype,
        item_size=item_size,
        generator_fn=torch.randn,
    )
    dst_tensors, dst_protos, _ = _allocate_model_tensors_and_metadata(
        dst_specs,
        device=dst_device,
        dtype=torch_dtype,
        item_size=item_size,
        generator_fn=torch.zeros,
    )
    torch.tpu.synchronize()

    total_gb = total_bytes / 1e9
    logging.info(
        "Allocated %d tensors: %.2f GB (%d bytes).",
        len(src_specs),
        total_gb,
        total_bytes,
    )

    # Start in-process RaidenController
    controller_network_client = raiden_controller.WeightSyncWorkerRpcClient(
        name_resolver=None
    )
    controller = raiden_controller.RaidenController(
        port=0,
        worker_rpc_client=controller_network_client,
    )
    controller_server = raiden_controller.RaidenControllerServer(controller)
    controller_server.start()
    controller_port = controller_server.port

    try:
      ws_source = weight_synchronizer.WeightSynchronizer(
          src_tensors,
          local_port=0,
          listener_port=0,
          parallelism=parallelism,
          bind_ip="127.0.0.1",
      )
      ws_dest = weight_synchronizer.WeightSynchronizer(
          dst_tensors,
          local_port=0,
          listener_port=0,
          parallelism=parallelism,
          bind_ip="127.0.0.1",
          auto_h2d=False,
      )

      skip_tiling_map = _compute_skip_tiling_map(src_specs)
      skip_tiling_list = [skip_tiling_map[i] for i in range(len(src_specs))]
      ws_source.test_only_set_skip_tiling(skip_tiling_list)
      ws_dest.test_only_set_skip_tiling(skip_tiling_list)

      src_unit = raiden_controller.RaidenId("trainer", "0", "weights")
      dst_unit = raiden_controller.RaidenId("sampler", "0", "weights")

      ctrl_client = raiden_controller.RaidenControllerClientFacade(
          f"127.0.0.1:{controller_port}",
          name_resolver=None,
      )
      ctrl_client.register_work_unit(
          src_unit,
          [f"127.0.0.1:{ws_source.local_port}"],
          f"127.0.0.1:{ws_source.listener_port}",
          mesh_shape=[1, 1],
          variables=src_protos,
          mesh_axes=["fsdp", "tp"],
      )
      ctrl_client.register_work_unit(
          dst_unit,
          [f"127.0.0.1:{ws_dest.local_port}"],
          f"127.0.0.1:{ws_dest.listener_port}",
          mesh_shape=[1, 1],
          variables=dst_protos,
          mesh_axes=["fsdp", "tp"],
      )

      # Warmup transfer (establishes TCP connections, plan caching, and
      # distributes skip_tiling plan to source and destination before D2H)
      logging.info(
          "Executing warmup transfer (group_size=%d, parallelism=%d)...",
          group_size,
          parallelism,
      )
      _run_controller_transfer(
          controller,
          src_unit,
          dst_unit,
          uuid=999,
          req_id="warmup",
          group_size=group_size,
          parallelism=parallelism,
          skip_tiling=skip_tiling_map,
          skip_d2h=False,
      )
      ws_dest.h2d()
      torch.tpu.synchronize()

      # Correctness gate before timed benchmark
      self._verify_tensor_parity(src_tensors, dst_tensors)
      logging.info("Warmup numerical parity check passed!")

      # Multi-iteration timed benchmark run
      d2h_latencies_ms = []
      h2h_latencies_ms = []
      h2d_latencies_ms = []
      e2e_latencies_ms = []

      logging.info("Executing %d benchmark iterations...", num_iters)
      for it in range(num_iters):
        # Stage 1: Standalone D2H copy + detiling
        t0 = time.perf_counter()
        ws_source.d2h()
        d2h_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Controller-orchestrated resharded H2H network push
        t1 = time.perf_counter()
        _run_controller_transfer(
            controller,
            src_unit,
            dst_unit,
            uuid=1000 + it,
            req_id=f"bench_{it}",
            group_size=group_size,
            parallelism=parallelism,
            skip_tiling=skip_tiling_map,
        )
        h2h_ms = (time.perf_counter() - t1) * 1000.0

        # Stage 3: Standalone H2D copy + tiling on destination
        t2 = time.perf_counter()
        ws_dest.h2d()
        torch.tpu.synchronize()
        h2d_ms = (time.perf_counter() - t2) * 1000.0

        e2e_ms = d2h_ms + h2h_ms + h2d_ms

        d2h_latencies_ms.append(d2h_ms)
        h2h_latencies_ms.append(h2h_ms)
        h2d_latencies_ms.append(h2d_ms)
        e2e_latencies_ms.append(e2e_ms)

        src_metrics = ws_source.get_metrics()
        logging.info(
            "Iteration %d/%d: D2H=%.2f ms (internal=%.2f ms), H2H=%.2f ms"
            " (internal=%.2f ms), H2D=%.2f ms, E2E=%.2f ms",
            it + 1,
            num_iters,
            d2h_ms,
            src_metrics.get("last_d2h_time_ms", 0.0),
            h2h_ms,
            src_metrics.get("last_h2h_time_ms", 0.0),
            h2d_ms,
            e2e_ms,
        )

      med_d2h_ms = float(np.median(d2h_latencies_ms))
      med_h2h_ms = float(np.median(h2h_latencies_ms))
      med_h2d_ms = float(np.median(h2d_latencies_ms))
      med_e2e_ms = float(np.median(e2e_latencies_ms))

      d2h_bw_gbs = total_gb / (med_d2h_ms / 1000.0) if med_d2h_ms > 0 else 0.0
      h2h_bw_gbs = total_gb / (med_h2h_ms / 1000.0) if med_h2h_ms > 0 else 0.0
      h2d_bw_gbs = total_gb / (med_h2d_ms / 1000.0) if med_h2d_ms > 0 else 0.0
      e2e_bw_gbs = total_gb / (med_e2e_ms / 1000.0) if med_e2e_ms > 0 else 0.0

      logging.info("=" * 80)
      logging.info(
          "QWEN 3.5 35B WEIGHT SYNCHRONIZATION BENCHMARK RESULTS (PyTorch)"
      )
      logging.info("=" * 80)
      logging.info(
          "Model Parameters Payload: %.2f GB (%d bytes, %d layers, dtype=%s)",
          total_gb,
          total_bytes,
          num_layers,
          _DTYPE.value,
      )
      logging.info(
          "Transfer Configuration  : group_size=%d, parallelism=%d, iters=%d",
          group_size,
          parallelism,
          num_iters,
      )
      logging.info("-" * 80)
      logging.info(
          "Device-to-Host (D2H)    : %8.2f ms | Throughput: %6.2f GB/s",
          med_d2h_ms,
          d2h_bw_gbs,
      )
      logging.info(
          "Host-to-Host (H2H)      : %8.2f ms | Throughput: %6.2f GB/s",
          med_h2h_ms,
          h2h_bw_gbs,
      )
      logging.info(
          "Host-to-Device (H2D)    : %8.2f ms | Throughput: %6.2f GB/s",
          med_h2d_ms,
          h2d_bw_gbs,
      )
      logging.info(
          "Total Pipeline E2E Time : %8.2f ms | Aggregate : %6.2f GB/s",
          med_e2e_ms,
          e2e_bw_gbs,
      )
      logging.info("=" * 80)

      # Post-benchmark parity verification across all tensors
      self._verify_tensor_parity(src_tensors, dst_tensors)
      logging.info(
          "Post-benchmark numerical parity verified across all %d tensors.",
          len(src_specs),
      )
    finally:
      controller_server.stop()


if __name__ == "__main__":
  absltest.main()
