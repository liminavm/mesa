/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_buffer.h"

#include <Metal/MTL4BufferRange.h>
#include <Metal/MTL4CommandAllocator.h>
#include <Metal/MTL4CommandBuffer.h>
#include <Metal/MTL4Counters.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

/* limina: RTLOG knob, cached — a getenv here sat on the per-draw path (round 24:
 * ~7% of the hot ring core in __findenv_locked). */
static inline bool
limina_kk_rtlog_cached(void)
{
   static int v = -1;
   if (v < 0)
      v = getenv("LIMINA_KK_RTLOG") != NULL;
   return v;
}

/* limina LIMINA_KK_ALLOC_STATS=<n> — the discriminating instrument for the IOAccelerator region
 * ratchet (spikes/vrend-region-leak/). Apple documents -[MTL4CommandAllocator allocatedSize] as
 * "the size of the internal memory heaps of this command allocator", i.e. a direct read of the
 * pool we have so far only inferred from ioclasscount.
 *
 * Two mechanisms fit every measurement taken to date and the destroy+recreate fix works under
 * both, which is exactly why that fix did not separate them:
 *
 *   (A) each allocator's pool grows without bound  -> per-allocator size climbs, count flat
 *   (B) each allocator plateaus at a high-water the reset never returns, and the *population*
 *       grows                                      -> per-allocator size plateaus, count climbs
 *
 * So report both, keyed by allocator identity: every <n> resets, dump the number of distinct
 * live allocators seen and the sum/max/mean of their last observed allocatedSize. Under (A) the
 * max climbs; under (B) the max flattens while the count climbs.
 *
 * Caveat, deliberately not glossed: allocatedSize reports the *encoding heaps*. The pooled
 * AGXResources may not be billed there, so a flat reading here does NOT retract the parking
 * proof — destroy-kills-them already settled that, and AGXResource remains the ground truth.
 * This narrows the mechanism; it does not adjudicate the fix. */
static pthread_mutex_t limina_alloc_stats_lock = PTHREAD_MUTEX_INITIALIZER;

#define LIMINA_ALLOC_SLOTS 8192
static struct {
   void *key;
   uint64_t size;
   int32_t pending;    /* committed-but-not-yet-complete command buffers on this allocator */
   uint64_t violations; /* resets taken while pending > 0 */
} limina_alloc_seen[LIMINA_ALLOC_SLOTS];
static uint32_t limina_alloc_distinct;
static uint64_t limina_alloc_resets;
static uint64_t limina_alloc_violations;  /* total resets taken while pending > 0 */
static uint64_t limina_alloc_commits;
static uint32_t limina_alloc_cb_overflow; /* cmd-buf map ran full; mappings were dropped */
static uint64_t limina_alloc_cb_charged;  /* committed cmd bufs successfully charged to an allocator */
static uint64_t limina_alloc_cb_missed;   /* committed cmd bufs with no recorded allocator */

static inline uint32_t
limina_alloc_stats_every(void)
{
   static int v = -1;
   if (v < 0) {
      const char *e = getenv("LIMINA_KK_ALLOC_STATS");
      v = (e && *e) ? atoi(e) : 0;
      if (v < 0)
         v = 0;
   }
   return (uint32_t)v;
}

int
limina_kk_alloc_stats_on(void)
{
   return limina_alloc_stats_every() != 0;
}

/* Slot for `key` in the allocator table. Caller holds the lock. -1 if absent and !create, or if
 * the table is full (open addressing; the live population is in the hundreds, so this stays far
 * from full — if it ever does fill, give up rather than spin). */
static int
limina_alloc_slot(void *key, bool create)
{
   uintptr_t h = ((uintptr_t)key >> 4) * 2654435761u;
   uint32_t i = (uint32_t)(h % LIMINA_ALLOC_SLOTS);
   for (uint32_t probe = 0; probe < LIMINA_ALLOC_SLOTS; ++probe) {
      uint32_t s = (i + probe) % LIMINA_ALLOC_SLOTS;
      if (limina_alloc_seen[s].key == key)
         return (int)s;
      if (limina_alloc_seen[s].key == NULL) {
         if (!create)
            return -1;
         limina_alloc_seen[s].key = key;
         limina_alloc_distinct++;
         return (int)s;
      }
   }
   return -1;
}

/* --- pending-at-reset: the discriminator between the two surviving mechanisms ---------------
 *
 * Apple: "You are responsible to ensure that all command buffers with memory originating from
 * this allocator instance are complete before calling resetting it." Two mechanisms explain the
 * per-workload-launch ratchet equally well and nothing measured so far separates them:
 *
 *   (1) resets landing while that allocator's command buffers are still in flight, so Metal
 *       cannot recycle the heaps and allocates fresh ones;
 *   (2) plain documented high-water retention, where each launch drives a *different subset* of
 *       the 450 allocators to a new high-water. No contract violation at all.
 *
 * So count the violations directly. mtl_begin_command_buffer records cmd_buf -> allocator;
 * commit increments that allocator's pending and arms a feedback handler to decrement it on
 * completion; reset checks pending and counts a violation if it is non-zero.
 *
 * Violations at launch magnitude => (1). Zero violations while the sum still steps per launch
 * => (2), proven, and "obey the contract" fixes nothing.
 *
 * Note on the feedback handler: KK appends one to a queue-lifetime MTL4CommitOptions on every
 * submit. Whether Metal snapshots that list per commit or reads it live at completion is
 * UNDOCUMENTED, so a block may or may not be invoked again for later commits.
 *
 * An earlier version captured a malloc'd batch and freed it on first fire. That is a
 * use-after-free if re-invocation happens at all: the second call reads `fired` from freed
 * memory. Batches therefore live in a static ring and are never freed — the block captures an
 * (index, generation) pair, and a stale invocation after wraparound fails the generation check
 * instead of touching recycled state. Correct under either accumulation semantics.
 *
 * Circumstantial evidence that handlers do NOT accumulate: if they did, every completion would
 * invoke O(N) blocks, and the 2.8M-reset continuous run would have collapsed under O(N^2) work.
 * It did not. But that is an inference, so the ring does not rely on it. */
#define LIMINA_CB_SLOTS 32768
static struct {
   void *cb;
   void *alloc;
} limina_cb_map[LIMINA_CB_SLOTS];

#define LIMINA_BATCH_RING 8192
#define LIMINA_BATCH_MAX_CBS 32
static struct {
   uint64_t gen;
   int fired;
   uint32_t count;
   void *allocs[LIMINA_BATCH_MAX_CBS];
} limina_batch_ring[LIMINA_BATCH_RING];
static uint64_t limina_batch_seq;
static uint64_t limina_batch_truncated; /* commits with more cbs than the ring slot holds */
static uint64_t limina_discharges;      /* completions observed — the discharge-side self-test */
/* In-flight command buffers, and the high-water of that. The peak is the direct answer to "is
 * ~450 allocators population or concurrency?" — it is how many command buffers are ever
 * simultaneously committed-but-not-complete. */
static uint32_t limina_alloc_outstanding;
static uint32_t limina_alloc_peak_pending;

/* Record which allocator this command buffer was begun on. */
static void
limina_alloc_track_begin(void *cb, void *alloc)
{
   pthread_mutex_lock(&limina_alloc_stats_lock);
   uintptr_t h = ((uintptr_t)cb >> 4) * 2654435761u;
   uint32_t i = (uint32_t)(h % LIMINA_CB_SLOTS);
   bool placed = false;
   for (uint32_t probe = 0; probe < LIMINA_CB_SLOTS; ++probe) {
      uint32_t s = (i + probe) % LIMINA_CB_SLOTS;
      if (limina_cb_map[s].cb == NULL || limina_cb_map[s].cb == cb) {
         limina_cb_map[s].cb = cb;
         limina_cb_map[s].alloc = alloc;
         placed = true;
         break;
      }
   }
   if (!placed)
      limina_alloc_cb_overflow++;
   pthread_mutex_unlock(&limina_alloc_stats_lock);
}

/* Look up and clear the mapping for `cb`. Caller holds the lock. */
static void *
limina_alloc_take_cb(void *cb)
{
   uintptr_t h = ((uintptr_t)cb >> 4) * 2654435761u;
   uint32_t i = (uint32_t)(h % LIMINA_CB_SLOTS);
   for (uint32_t probe = 0; probe < LIMINA_CB_SLOTS; ++probe) {
      uint32_t s = (i + probe) % LIMINA_CB_SLOTS;
      if (limina_cb_map[s].cb == NULL)
         return NULL;
      if (limina_cb_map[s].cb == cb) {
         void *a = limina_cb_map[s].alloc;
         /* Tombstone-free deletion is wrong under open addressing, so keep the key and null the
          * allocator: the slot stays probeable and is reused by the next begin that hashes here.
          * Command buffers are minted fresh per render pass, so slots do accumulate — the
          * overflow counter is reported so a full map is never silent. */
         limina_cb_map[s].alloc = NULL;
         return a;
      }
   }
   return NULL;
}

/* Returns an opaque (gen, slot) token, or 0 for "nothing to track". Token 0 is never valid
 * because generations start at 1. */
uint64_t
limina_kk_alloc_track_commit(void **cbs, uint32_t count)
{
   if (!limina_kk_alloc_stats_on() || count == 0)
      return 0;

   pthread_mutex_lock(&limina_alloc_stats_lock);
   limina_alloc_commits++;

   uint32_t slot = (uint32_t)(limina_batch_seq % LIMINA_BATCH_RING);
   uint64_t gen = ++limina_batch_seq; /* starts at 1 */
   limina_batch_ring[slot].gen = gen;
   limina_batch_ring[slot].fired = 0;
   limina_batch_ring[slot].count = 0;

   for (uint32_t i = 0; i < count; ++i) {
      void *alloc = limina_alloc_take_cb(cbs[i]);
      /* Self-test: a negative violation result is only meaningful if the tracker actually
       * charges the command buffers it sees. Count the misses so "no violations" can never be
       * "the map was empty" in disguise. */
      if (!alloc) {
         limina_alloc_cb_missed++;
         continue;
      }
      limina_alloc_cb_charged++;
      int s = limina_alloc_slot(alloc, true);
      if (s < 0)
         continue;
      limina_alloc_seen[s].pending++;
      if (limina_batch_ring[slot].count < LIMINA_BATCH_MAX_CBS) {
         limina_batch_ring[slot].allocs[limina_batch_ring[slot].count++] = alloc;
         if (++limina_alloc_outstanding > limina_alloc_peak_pending)
            limina_alloc_peak_pending = limina_alloc_outstanding;
      } else {
         /* Charged but not recorded for discharge — it would stay pending forever and
          * manufacture false violations. Undo the charge and count it. */
         limina_alloc_seen[s].pending--;
         limina_alloc_cb_charged--;
         limina_batch_truncated++;
      }
   }

   uint32_t n = limina_batch_ring[slot].count;
   pthread_mutex_unlock(&limina_alloc_stats_lock);

   if (n == 0)
      return 0;
   return (gen << 16) | slot;
}

void
limina_kk_alloc_track_complete(uint64_t token)
{
   if (!token)
      return;

   uint32_t slot = (uint32_t)(token & 0xffff);
   uint64_t gen = token >> 16;
   if (slot >= LIMINA_BATCH_RING)
      return;

   pthread_mutex_lock(&limina_alloc_stats_lock);
   /* Generation check: the slot may have been recycled by a later commit, in which case this is
    * a stale invocation and must not touch the current occupant. */
   if (limina_batch_ring[slot].gen != gen || limina_batch_ring[slot].fired) {
      pthread_mutex_unlock(&limina_alloc_stats_lock);
      return;
   }
   limina_batch_ring[slot].fired = 1;
   limina_discharges++;
   for (uint32_t i = 0; i < limina_batch_ring[slot].count; ++i) {
      int s = limina_alloc_slot(limina_batch_ring[slot].allocs[i], false);
      if (s >= 0 && limina_alloc_seen[s].pending > 0)
         limina_alloc_seen[s].pending--;
      if (limina_alloc_outstanding > 0)
         limina_alloc_outstanding--;
   }
   pthread_mutex_unlock(&limina_alloc_stats_lock);
}

/* Record this allocator's current pool size, then periodically dump the population. */
static void
limina_alloc_stats_note(id<MTL4CommandAllocator> alloc, uint32_t every)
{
   uint64_t sz = [alloc allocatedSize];

   pthread_mutex_lock(&limina_alloc_stats_lock);

   int slot = limina_alloc_slot((void *)alloc, true);
   if (slot >= 0) {
      limina_alloc_seen[slot].size = sz;
      /* THE measurement: is this reset landing on in-flight work? */
      if (limina_alloc_seen[slot].pending > 0) {
         limina_alloc_seen[slot].violations++;
         limina_alloc_violations++;
      }
   }

   if ((++limina_alloc_resets % every) == 0) {
      uint64_t sum = 0, max = 0;
      uint32_t pend_now = 0, ever_violated = 0, clean_n = 0;
      /* THE O2 JOIN: split the bytes by whether the allocator ever took a reset while its work
       * was in flight. If growth is violation-driven, it concentrates on violated_sum. If it
       * spreads across allocators that never violated, mechanism (2) is proven and the
       * anti-correlation argument no longer has to carry the claim alone. */
      uint64_t violated_sum = 0, clean_sum = 0;
      /* Size histogram: <1M, 1-8M, 8-32M, 32-128M, >=128M. Under mechanism (2) individual
       * allocators step up monotonically at workload launches and never move otherwise, so the
       * buckets should drift right in steps rather than spread smoothly. */
      uint32_t hist[5] = {0};
      for (uint32_t s = 0; s < LIMINA_ALLOC_SLOTS; ++s) {
         if (limina_alloc_seen[s].key == NULL)
            continue;
         uint64_t sz2 = limina_alloc_seen[s].size;
         sum += sz2;
         if (sz2 > max)
            max = sz2;
         if (limina_alloc_seen[s].pending > 0)
            pend_now++;
         if (limina_alloc_seen[s].violations) {
            ever_violated++;
            violated_sum += sz2;
         } else {
            clean_n++;
            clean_sum += sz2;
         }
         hist[sz2 < (1u << 20)     ? 0
              : sz2 < (8u << 20)   ? 1
              : sz2 < (32u << 20)  ? 2
              : sz2 < (128u << 20) ? 3
                                   : 4]++;
      }
      uint32_t n = limina_alloc_distinct;
      fprintf(stderr,
              "[LIMINA-ALLOC-STATS] resets=%llu distinct=%u sum=%.1fMiB max=%.3fMiB "
              "mean=%.3fMiB | commits=%llu VIOLATIONS=%llu allocs_ever_violated=%u "
              "pending_now=%u peak_pending=%u cb_overflow=%u charged=%llu missed=%llu "
              "discharges=%llu truncated=%llu | JOIN violated_n=%u violated_sum=%.1fMiB "
              "clean_n=%u clean_sum=%.1fMiB | hist<1M=%u 1-8M=%u 8-32M=%u 32-128M=%u "
              ">=128M=%u\n",
              (unsigned long long)limina_alloc_resets, n, sum / 1048576.0,
              max / 1048576.0, n ? (sum / (double)n) / 1048576.0 : 0.0,
              (unsigned long long)limina_alloc_commits,
              (unsigned long long)limina_alloc_violations, ever_violated,
              pend_now, limina_alloc_peak_pending, limina_alloc_cb_overflow,
              (unsigned long long)limina_alloc_cb_charged,
              (unsigned long long)limina_alloc_cb_missed,
              (unsigned long long)limina_discharges,
              (unsigned long long)limina_batch_truncated, ever_violated,
              violated_sum / 1048576.0, clean_n, clean_sum / 1048576.0, hist[0],
              hist[1], hist[2], hist[3], hist[4]);
   }

   pthread_mutex_unlock(&limina_alloc_stats_lock);
}

uint64_t
mtl_command_allocator_allocated_size(mtl_command_allocator *allocator)
{
   @autoreleasepool {
      return [(id<MTL4CommandAllocator>)allocator allocatedSize];
   }
}

void
mtl_command_allocator_reset(mtl_command_allocator *allocator)
{
   @autoreleasepool {
      id<MTL4CommandAllocator> alloc = (id<MTL4CommandAllocator>)allocator;
      uint32_t every = limina_alloc_stats_every();
      /* Sample BEFORE the reset: after it, the pool is marked reusable and the interesting
       * quantity (what this allocator was holding) may read differently. */
      if (every)
         limina_alloc_stats_note(alloc, every);
      [alloc reset];
   }
}

void
mtl_begin_command_buffer(mtl_command_buffer *command_buffer,
                         mtl_command_allocator *allocator)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)command_buffer;
      id<MTL4CommandAllocator> alloc = (id<MTL4CommandAllocator>)allocator;
      if (limina_kk_alloc_stats_on())
         limina_alloc_track_begin(command_buffer, allocator);
      [cmd_buf beginCommandBufferWithAllocator:alloc];
   }
}

void
mtl_end_command_buffer(mtl_command_buffer *command_buffer)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)command_buffer;
      [cmd_buf endCommandBuffer];
   }
}

void
mtl_command_resolve_counter_heap(mtl_command_buffer *command_buffer,
                                 mtl_counter_heap *heap, uint32_t first_index,
                                 uint32_t count, uint64_t dst_addr)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)command_buffer;
      id<MTL4CounterHeap> h = (id<MTL4CounterHeap>)heap;
      MTL4BufferRange range = {
         .bufferAddress = dst_addr,
         .length = count * sizeof(MTL4TimestampHeapEntry),
      };
      [cmd_buf resolveCounterHeap:h
                        withRange:NSMakeRange(first_index, count)
                       intoBuffer:range
                        waitFence:nil
                      updateFence:nil];
   }
}
