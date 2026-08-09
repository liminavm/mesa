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
} limina_alloc_seen[LIMINA_ALLOC_SLOTS];
static uint32_t limina_alloc_distinct;
static uint64_t limina_alloc_resets;

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

/* Record this allocator's current pool size, then periodically dump the population. */
static void
limina_alloc_stats_note(id<MTL4CommandAllocator> alloc, uint32_t every)
{
   uint64_t sz = [alloc allocatedSize];

   pthread_mutex_lock(&limina_alloc_stats_lock);

   /* Open addressing on the pointer; the live population is in the hundreds, so this stays
    * far from full. If it ever does fill, stop inserting rather than spin. */
   uintptr_t h = ((uintptr_t)alloc >> 4) * 2654435761u;
   uint32_t i = (uint32_t)(h % LIMINA_ALLOC_SLOTS);
   for (uint32_t probe = 0; probe < LIMINA_ALLOC_SLOTS; ++probe) {
      uint32_t s = (i + probe) % LIMINA_ALLOC_SLOTS;
      if (limina_alloc_seen[s].key == (void *)alloc) {
         limina_alloc_seen[s].size = sz;
         break;
      }
      if (limina_alloc_seen[s].key == NULL) {
         limina_alloc_seen[s].key = (void *)alloc;
         limina_alloc_seen[s].size = sz;
         limina_alloc_distinct++;
         break;
      }
   }

   if ((++limina_alloc_resets % every) == 0) {
      uint64_t sum = 0, max = 0;
      for (uint32_t s = 0; s < LIMINA_ALLOC_SLOTS; ++s) {
         if (limina_alloc_seen[s].key == NULL)
            continue;
         sum += limina_alloc_seen[s].size;
         if (limina_alloc_seen[s].size > max)
            max = limina_alloc_seen[s].size;
      }
      uint32_t n = limina_alloc_distinct;
      fprintf(stderr,
              "[LIMINA-ALLOC-STATS] resets=%llu distinct=%u sum=%.1fMiB max=%.3fMiB "
              "mean=%.3fMiB\n",
              (unsigned long long)limina_alloc_resets, n, sum / 1048576.0,
              max / 1048576.0, n ? (sum / (double)n) / 1048576.0 : 0.0);
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
