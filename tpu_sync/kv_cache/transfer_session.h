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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SESSION_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SESSION_H_

#include <atomic>
#include <chrono>  // NOLINT(build/c++11)
#include <cstddef>
#include <deque>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"

namespace tpu_raiden {

// Manages the host layer staging slots of a transfer session when fewer host
// layer buffers are allocated than layers. A layer leases the slot at the head
// of the free queue when its host staging starts (first bytes arriving on a
// receiver, or D2H dispatch on a sender), keeps it until its host work
// completes (H2D copy on a receiver, or H2H push on a sender), and Release()
// returns the slot to the tail; Acquire() blocks while every slot is leased.
// Layers may acquire and complete in any order.
// Thread-safe.
class StagingLayerManager {
 public:
  // Initializes the manager with free slots [0, |num_slots|).
  explicit StagingLayerManager(size_t num_slots = 0);

  // Returns the slot leased to |unit|, leasing the head free slot if |unit|
  // has none. Waits until |deadline| for a release when every slot is leased;
  // fails kDeadlineExceeded on timeout and kCancelled after Shutdown() or once
  // |*cancelled| is set.
  absl::StatusOr<size_t> Acquire(size_t unit,
                                 std::chrono::steady_clock::time_point deadline,
                                 const std::atomic<bool>* cancelled = nullptr);

  // Returns the slot leased to |unit|, if any.
  std::optional<size_t> Lookup(size_t unit) const;

  // Ends |unit|'s lease, returning its slot to the tail of the free queue and
  // waking one waiter. Returns the released slot; nullopt if |unit| held none.
  std::optional<size_t> Release(size_t unit);

  // Ends every lease.
  void ReleaseAll();

  // Fails every current and future Acquire() with kCancelled.
  void Shutdown();

  size_t num_free() const;

 private:
  mutable absl::Mutex mu_;
  // Free slots in release order: Acquire() pops the head, Release() pushes
  // the tail.
  std::deque<size_t> free_ ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<size_t, size_t> leases_ ABSL_GUARDED_BY(mu_);
  bool shutdown_ ABSL_GUARDED_BY(mu_) = false;
};

// Base interface for producer and consumer KV-cache transfer sessions.
//
// Implementations are thread-safe and synchronize session lifecycle state
// internally.
class TransferSession {
 public:
  virtual ~TransferSession() = default;

  // Returns true once the session's outcome has been decided and all in-flight
  // operations have drained.
  virtual bool Done() const = 0;

  // Records the session's outcome with |status| and transitions the session
  // into draining (or marks it done immediately if no operations are in
  // flight).
  virtual void Finish(const absl::Status& status = absl::OkStatus()) = 0;

  // Returns the session's status (OkStatus() unless the session has failed).
  virtual absl::Status GetStatus() const = 0;

  // Blocks until Done() becomes true and returns GetStatus().
  virtual absl::Status AwaitForDone() = 0;

  // Returns true if the session's outcome has been decided and it is draining
  // in-flight operations (or has finished draining).
  virtual bool IsDraining() const = 0;

  // Returns the host staging buffer leased to |unit| (a layer index on the
  // layer transfer path, a pool index on the pool reshard path), leasing one
  // when it has none and waiting until the session deadline while every
  // buffer is leased. Fails with the session's terminal status (or
  // kCancelled) once the session is draining. Thread-safe.
  virtual absl::StatusOr<size_t> AcquireStagingSlot(size_t unit) = 0;

  // Returns the host staging buffer currently leased to |unit|, or
  // kFailedPrecondition when it holds none. Thread-safe.
  virtual absl::StatusOr<size_t> LookupStagingSlot(size_t unit) const = 0;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_KV_CACHE_TRANSFER_SESSION_H_
