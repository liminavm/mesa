/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_device.h"

#include "kk_buffer.h"
#include "kk_buffer_view.h"

#include "util/u_atomic.h"

#include "kk_cmd_buffer.h"
#include "kk_entrypoints.h"
#include "kk_image.h"
#include "kk_image_view.h"
#include "kk_instance.h"
#include "kk_limina_work.h"
#include "kk_physical_device.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_device.h"
#include "kosmickrisp/bridge/ns_process_info.h"
#include "kosmickrisp/compiler/nir_to_msl.h"

#include "kk_dispatch_cmd.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"

#include "vulkan/wsi/wsi_common.h"
#include "vk_pipeline_cache.h"

#include "util/os_time.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* limina: the shared command-allocator pool. Rationale and the measurements behind it are on
 * struct kk_alloc_pool (kk_device.h). */

/* Total ~ F + active x B: the population term bounded by in-flight work, the per-allocator term
 * by B. B is also the churn knob, since every allocator that crosses it is released and replaced
 * by a mint, and it must sit above a busy client's working size. A venus client's allocator
 * settles near 13 MiB: at 4 MiB it was replaced every ~13 recordings and venus lost 27-55%
 * (vkmark, glmark2, vk-replay); at 16 MiB it is never replaced and venus matches no pool at all,
 * while the vrend device still holds ~200 MiB flat across launches
 * (limina perf/kk-alloc-pool-2026-10-01, spikes/kk-alloc-pool). */
#define KK_ALLOC_BUDGET_MIB_DEFAULT 16
/* Growth past this is reported. The pool is per VkDevice, and the busiest device is the vrend
 * tier's: host zink's one device serves every guest GL client, and each of its contexts keeps
 * three command buffers recording. A seated GNOME guest running Firefox WebGL, glmark2 and two
 * vkcubes peaked at 78 there, ~60 of them borrowed even once the apps had quit; every venus
 * device (one per guest Vulkan device) peaked at 1. */
#define KK_ALLOC_POOL_WATERMARK 128
/* The most allocators one device's pool will hold: over six times that measured peak, and far
 * below the thousands an unthrottled guest reached before Metal refused an allocation and the
 * worker aborted. It must stay above the most command buffers a device's clients record at once,
 * or a client that holds them all waits on itself until its Begin fails. */
#define KK_ALLOC_CEILING_DEFAULT 512
/* How long an acquire at the ceiling waits for room before failing. Short: zink retries a failed
 * vkBeginCommandBuffer itself with its own back-off, and a waiter here stalls a client thread. */
#define KK_ALLOC_WAIT_MS_DEFAULT 100
/* Idle-decay before an under-budget allocator counts as surplus. Long enough that a burst which
 * merely paused between frames keeps its working set; short enough that an app exit is reclaimed
 * while the compositor is still drawing (reclaim is acquire-driven, so a fully idle guest holds
 * its high-water until activity resumes, which is exactly when the memory is wanted again). */
#define KK_ALLOC_DECAY_MS_DEFAULT 2000
/* Never decay below this many. NOT peak_live: peak_live only ever grows, so a floor there would
 * pin the pool at its all-time high-water and reclaim nothing after a heavy app exits, which is
 * the entire point. A small constant protects steady-state concurrency. */
#define KK_ALLOC_FLOOR_DEFAULT 8
/* Releases done by one acquire, outside the mutex. Bounded so one vkBeginCommandBuffer never pays
 * for a whole post-workload drain; the next acquires pick up the rest. */
#define KK_ALLOC_RELEASE_BATCH 8

static uint64_t
kk_env_u64(const char *name, uint64_t dflt)
{
   const char *e = getenv(name);
   if (!e || !*e)
      return dflt;
   char *end = NULL;
   unsigned long long v = strtoull(e, &end, 10);
   return (end != e && *end == '\0') ? (uint64_t)v : dflt;
}

void
kk_alloc_pool_init(struct kk_device *dev)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;

   mtx_init(&pool->mtx, mtx_plain);
   cnd_init(&pool->room);
   util_dynarray_init(&pool->allocs, NULL);
   pool->budget_bytes =
      kk_env_u64("LIMINA_KK_ALLOC_BUDGET_MIB", KK_ALLOC_BUDGET_MIB_DEFAULT) * 1024u * 1024u;
   pool->decay_ns = kk_env_u64("LIMINA_KK_ALLOC_DECAY_MS", KK_ALLOC_DECAY_MS_DEFAULT) * 1000000ull;
   pool->floor = (uint32_t)kk_env_u64("LIMINA_KK_ALLOC_FLOOR", KK_ALLOC_FLOOR_DEFAULT);
   pool->ceiling = (uint32_t)kk_env_u64("LIMINA_KK_ALLOC_CEILING", KK_ALLOC_CEILING_DEFAULT);
   if (pool->ceiling && pool->ceiling <= pool->floor)
      pool->ceiling = pool->floor + 1;
   pool->wait_ns = kk_env_u64("LIMINA_KK_ALLOC_WAIT_MS", KK_ALLOC_WAIT_MS_DEFAULT) * 1000000ull;
   pool->watermark_warned = KK_ALLOC_POOL_WATERMARK;
   pool->destroy_enabled = kk_env_u64("LIMINA_KK_ALLOC_DESTROY", 1) != 0;
   pool->stats_every = kk_env_u64("LIMINA_KK_ALLOC_STATS", 0);
}

void
kk_alloc_pool_finish(struct kk_device *dev)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;

   /* The app has seen its work complete by now, but Metal runs the feedback handlers that
    * discharge it on its own thread, and they can still be on their way. Freeing a pooled
    * allocator under one is a use-after-free, so give them a bounded moment to land. */
   for (unsigned waited_ms = 0; waited_ms < 2000; ++waited_ms) {
      uint32_t pending = 0;
      mtx_lock(&pool->mtx);
      util_dynarray_foreach(&pool->allocs, kk_pooled_alloc_ptr, pap)
         pending += (*pap)->pending;
      mtx_unlock(&pool->mtx);
      if (pending == 0)
         break;
      os_time_sleep(1000);
   }

   /* One line per device, so a high-water mark is on record for every client that came and went
    * (the pool is per VkDevice: under venus, one per guest Vulkan device). */
   if (pool->acquires) {
      fprintf(stderr,
              "[LIMINA-ALLOC-POOL] teardown: pool=%p live=%u peak=%u released=%u ceiling-waits=%llu "
              "ceiling-fails=%llu\n",
              (void *)pool, pool->live, pool->peak_live, pool->destroyed,
              (unsigned long long)pool->ceiling_waits, (unsigned long long)pool->ceiling_fails);
   }

   util_dynarray_foreach(&pool->allocs, kk_pooled_alloc_ptr, pap) {
      mtl_release((*pap)->handle);
      free(*pap);
   }
   util_dynarray_fini(&pool->allocs);
   cnd_destroy(&pool->room);
   mtx_destroy(&pool->mtx);
}

/* Whether `pa` should leave the pool now. A drained over-budget allocator always goes; an idle
 * under-budget one goes once it has sat out the decay window, while the pool is above its floor.
 * Never while borrowed or while the GPU still has work begun on it. Caller holds pool->mtx. */
static bool
kk_alloc_pool_is_surplus(const struct kk_alloc_pool *pool, const struct kk_pooled_alloc *pa,
                         uint32_t live, uint64_t now)
{
   if (pa->in_use || pa->pending != 0)
      return false;
   if (pa->draining)
      return true;
   return pool->destroy_enabled && live > pool->floor && pa->idle_since != 0 &&
          now - pa->idle_since >= pool->decay_ns;
}

/* One stats line, in the shape spikes/kk-alloc-pool/ratchet.sh parses. Reads allocatedSize under
 * the mutex, which is fine at a cadence of thousands of acquires. Caller holds pool->mtx. */
static void
kk_alloc_pool_stats(struct kk_alloc_pool *pool)
{
   uint64_t sum = 0, max = 0;
   uint32_t draining = 0, borrowed = 0, pending = 0;
   util_dynarray_foreach(&pool->allocs, kk_pooled_alloc_ptr, pap) {
      struct kk_pooled_alloc *pa = *pap;
      uint64_t size = mtl_command_allocator_allocated_size(pa->handle);
      sum += size;
      if (size > max)
         max = size;
      draining += pa->draining;
      borrowed += pa->in_use;
      pending += pa->pending;
   }
   fprintf(stderr,
           "[LIMINA-ALLOC-STATS] pool=%p begins=%llu live=%u peak=%u draining=%u borrowed=%u "
           "pending=%u released=%u retired=%llu mean_uses=%.1f sum=%.1fMiB max=%.1fMiB\n",
           (void *)pool, (unsigned long long)pool->acquires, pool->live, pool->peak_live, draining,
           borrowed, pending, pool->destroyed, (unsigned long long)pool->retired,
           pool->retired ? (double)pool->retired_uses / (double)pool->retired : 0.0,
           sum / (1024.0 * 1024.0), max / (1024.0 * 1024.0));
}

/* Release what a pass of the acquire loop unlinked. Outside the lock on purpose: releasing an
 * allocator unmaps its heaps in the kernel, and vkBeginCommandBuffer must not serialise behind
 * that. Measured to return 100% of the heaps (spikes/vrend-region-leak/mtl4-repro/destroy-probe.m). */
static void
kk_alloc_pool_release_doomed(mtl_command_allocator **doomed, unsigned *n_doomed)
{
   for (unsigned i = 0; i < *n_doomed; i++)
      mtl_release(doomed[i]);
   *n_doomed = 0;
}

static void
kk_alloc_pool_wake(struct kk_alloc_pool *pool)
{
   if (pool->waiters)
      cnd_broadcast(&pool->room);
}

struct kk_pooled_alloc *
kk_alloc_pool_acquire(struct kk_device *dev)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;
   struct kk_pooled_alloc *found = NULL;
   mtl_command_allocator *doomed[KK_ALLOC_RELEASE_BATCH];
   unsigned n_doomed = 0;
   uint64_t deadline = 0;

   mtx_lock(&pool->mtx);

   for (;;) {
      const uint64_t now = os_time_get_nano();

      /* An allocator that is ready as-is: not borrowed, not over budget. Prefer the most recently
       * used one, so the rest age out under the decay clock instead of being kept warm in
       * rotation. */
      util_dynarray_foreach(&pool->allocs, kk_pooled_alloc_ptr, pap) {
         struct kk_pooled_alloc *pa = *pap;
         if (pa->in_use || pa->draining)
            continue;
         if (!found || pa->idle_since > found->idle_since)
            found = pa;
      }

      /* Unlink the surplus here and release it outside the mutex. Only alongside a pooled
       * hand-out for the decay case, so a decay release is never paired with a mint in the same
       * call: that pairing is exactly the thrash the floor and the decay window exist to avoid. A
       * drained over-budget allocator is unusable either way, so it goes regardless, and at the
       * ceiling that is what makes room. */
      uint32_t live_after = pool->live;
      for (unsigned i = 0; i < util_dynarray_num_elements(&pool->allocs, kk_pooled_alloc_ptr) &&
                           n_doomed < KK_ALLOC_RELEASE_BATCH;) {
         struct kk_pooled_alloc *pa =
            *util_dynarray_element(&pool->allocs, kk_pooled_alloc_ptr, i);
         if (pa == found || !kk_alloc_pool_is_surplus(pool, pa, live_after, now) ||
             (!pa->draining && !found)) {
            i++;
            continue;
         }
         doomed[n_doomed++] = pa->handle;
         util_dynarray_delete_unordered(&pool->allocs, kk_pooled_alloc_ptr, pa);
         free(pa);
         live_after--;
      }
      pool->destroyed += pool->live - live_after;
      pool->live = live_after;

      if (found || !pool->ceiling || pool->live < pool->ceiling)
         break;

      /* At the ceiling. A lost device completes nothing, so nothing will ever drain. */
      if (vk_device_is_lost_no_report(&dev->vk))
         break;

      if (!deadline) {
         deadline = now + pool->wait_ns;
         if (util_is_power_of_two_nonzero64(++pool->ceiling_waits)) {
            fprintf(stderr,
                    "[LIMINA-ALLOC-POOL] at its ceiling of %u allocators (LIMINA_KK_ALLOC_CEILING) — "
                    "waiting up to %llu ms for the GPU (%llu times so far)\n",
                    pool->ceiling, (unsigned long long)(pool->wait_ns / 1000000ull),
                    (unsigned long long)pool->ceiling_waits);
         }
      } else if (now >= deadline) {
         if (util_is_power_of_two_nonzero64(++pool->ceiling_fails)) {
            fprintf(stderr,
                    "[LIMINA-ALLOC-POOL] still at its ceiling of %u allocators (LIMINA_KK_ALLOC_CEILING) "
                    "after %llu ms — refusing a command allocator (%llu times so far)\n",
                    pool->ceiling, (unsigned long long)(pool->wait_ns / 1000000ull),
                    (unsigned long long)pool->ceiling_fails);
         }
         mtx_unlock(&pool->mtx);
         kk_alloc_pool_release_doomed(doomed, &n_doomed);
         return NULL;
      }

      /* Nothing unlinked can still be pending, so freeing it now is safe; do it before
       * sleeping rather than holding the heaps across the wait. */
      if (n_doomed) {
         mtx_unlock(&pool->mtx);
         kk_alloc_pool_release_doomed(doomed, &n_doomed);
         mtx_lock(&pool->mtx);
         continue;
      }

      struct timespec abs;
      timespec_get(&abs, TIME_UTC);
      uint64_t left = deadline - now;
      abs.tv_sec += left / 1000000000ull;
      abs.tv_nsec += left % 1000000000ull;
      if (abs.tv_nsec >= 1000000000l) {
         abs.tv_sec++;
         abs.tv_nsec -= 1000000000l;
      }
      pool->waiters++;
      cnd_timedwait(&pool->room, &pool->mtx, &abs);
      pool->waiters--;
   }

   if (!found) {
      /* A lost device completes nothing, so an allocator minted after the loss can never drain.
       * Measured on the WebGL {antialias:true} device loss: thousands of allocators with tens of
       * thousands of BO allocations behind them, ~100k compressor pages only a reboot returned.
       * Refuse instead; the caller handles NULL. */
      if (vk_device_is_lost_no_report(&dev->vk)) {
         static bool reported;
         if (!reported) {
            reported = true;
            fprintf(stderr, "[LIMINA-KK] device is lost — refusing to mint command allocators\n");
            fflush(stderr);
         }
         mtx_unlock(&pool->mtx);
         kk_alloc_pool_release_doomed(doomed, &n_doomed);
         return NULL;
      }

      /* Below the ceiling, never block waiting for a drain: stalling vkBeginCommandBuffer on GPU
       * progress invites jank and priority inversion. Mint instead, and make growth visible. */
      found = calloc(1, sizeof(*found));
      if (found)
         found->handle = mtl_new_command_allocator(dev->mtl_handle);
      kk_pooled_alloc_ptr *slot =
         found && found->handle ? util_dynarray_grow(&pool->allocs, kk_pooled_alloc_ptr, 1) : NULL;
      if (!slot) {
         if (found && found->handle)
            mtl_release(found->handle);
         free(found);
         mtx_unlock(&pool->mtx);
         kk_alloc_pool_release_doomed(doomed, &n_doomed);
         return NULL;
      }
      *slot = found;
      pool->live++;
      if (pool->live > pool->peak_live)
         pool->peak_live = pool->live;
      /* A high-water mark, reported at each doubling past the watermark rather than at every new
       * peak: a runaway used to print one line per allocator, which added load of its own. */
      if (pool->live > pool->watermark_warned) {
         pool->watermark_warned *= 2;
         fprintf(stderr,
                 "[LIMINA-ALLOC-POOL] grew to %u allocators (budget %llu MiB, LIMINA_KK_ALLOC_CEILING=%u) — "
                 "in-flight depth is outrunning completion\n",
                 pool->live, (unsigned long long)(pool->budget_bytes >> 20), pool->ceiling);
      }
   }

   found->in_use = true;
   found->idle_since = 0;
   found->uses++;
   pool->acquires++;
   if (pool->stats_every && pool->acquires % pool->stats_every == 0)
      kk_alloc_pool_stats(pool);
   mtx_unlock(&pool->mtx);

   kk_alloc_pool_release_doomed(doomed, &n_doomed);

   return found;
}

void
kk_alloc_pool_release(struct kk_device *dev, struct kk_pooled_alloc *pa)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;
   if (!pa)
      return;

   /* Read the size OUTSIDE the lock: this is a Metal call on the recording path. */
   uint64_t size = mtl_command_allocator_allocated_size(pa->handle);

   mtx_lock(&pool->mtx);
   pa->in_use = false;
   pa->idle_since = os_time_get_nano();
   /* Retire on budget when the borrow ends: allocatedSize never shrinks, so an allocator past the
    * budget stays past it, and the only way to get the memory back is to release it. */
   if (pool->budget_bytes && size >= pool->budget_bytes && !pa->draining) {
      pa->draining = true;
      pool->retired++;
      pool->retired_uses += pa->uses;
   }
   kk_alloc_pool_wake(pool);
   mtx_unlock(&pool->mtx);
}

void
kk_alloc_pool_charge(struct kk_device *dev, struct kk_pooled_alloc *pa)
{
   if (!pa)
      return;
   mtx_lock(&dev->alloc_pool.mtx);
   pa->pending++;
   mtx_unlock(&dev->alloc_pool.mtx);
}

void
kk_alloc_pool_discharge(struct kk_device *dev, struct kk_pooled_alloc *pa)
{
   if (!pa)
      return;
   mtx_lock(&dev->alloc_pool.mtx);
   assert(pa->pending > 0);
   if (pa->pending > 0)
      pa->pending--;
   /* A drained over-budget allocator is what frees room at the ceiling. */
   if (pa->pending == 0 && pa->draining)
      kk_alloc_pool_wake(&dev->alloc_pool);
   mtx_unlock(&dev->alloc_pool.mtx);
}

struct kk_mtl_compiler {
   mtl_compiler *handle;
   uint32_t refcount;
};

static struct hash_table compilers_ht;
static simple_mtx_t compilers_ht_lock;
static once_flag compilers_ht_once = ONCE_FLAG_INIT;

static void
kk_init_compiler_table()
{
   _mesa_pointer_hash_table_init(&compilers_ht, NULL);
   simple_mtx_init(&compilers_ht_lock, mtx_plain);
}

static mtl_compiler *
kk_acquire_compiler(struct kk_device *dev)
{
   /* KK_WORKAROUND_11 */
   struct kk_physical_device *pdev = kk_device_physical(dev);
   if (pdev->settings.disabled_workarounds & BITFIELD64_BIT(11)) {
      return mtl_new_compiler(dev->mtl_handle);
   }

   call_once(&compilers_ht_once, kk_init_compiler_table);
   simple_mtx_lock(&compilers_ht_lock);

   struct hash_entry *ent =
      _mesa_hash_table_search(&compilers_ht, dev->mtl_handle);

   struct kk_mtl_compiler *compiler;
   if (ent == NULL) {
      compiler = ralloc(NULL, struct kk_mtl_compiler);
      if (compiler == NULL) {
         simple_mtx_unlock(&compilers_ht_lock);
         return NULL;
      }

      compiler->handle = mtl_new_compiler(dev->mtl_handle);
      if (compiler->handle == NULL) {
         ralloc_free(compiler);
         simple_mtx_unlock(&compilers_ht_lock);
         return NULL;
      }

      compiler->refcount = 1;
      _mesa_hash_table_insert(&compilers_ht, dev->mtl_handle, compiler);
   } else {
      compiler = ent->data;
      compiler->refcount++;
   }

   simple_mtx_unlock(&compilers_ht_lock);
   return compiler->handle;
}

static void
kk_release_compiler(struct kk_device *dev)
{
   /* KK_WORKAROUND_11 */
   struct kk_physical_device *pdev = kk_device_physical(dev);
   if (pdev->settings.disabled_workarounds & BITFIELD64_BIT(11)) {
      mtl_release(dev->mtl_compiler_handle);
      return;
   }

   simple_mtx_lock(&compilers_ht_lock);

   struct hash_entry *ent =
      _mesa_hash_table_search(&compilers_ht, dev->mtl_handle);
   if (ent != NULL) {
      struct kk_mtl_compiler *compiler = ent->data;
      --compiler->refcount;

      if (compiler->refcount == 0) {
         _mesa_hash_table_remove(&compilers_ht, ent);
         mtl_release(compiler->handle);
         ralloc_free(compiler);
      }
   }

   simple_mtx_unlock(&compilers_ht_lock);
}

uint32_t kk_limina_sampler_retires = 0u;

/* limina: keep every sampler slot alive for the process lifetime. The hash table dedupes on
 * the packed descriptor, so the leak is bounded by the number of distinct sampler states. */
static bool
kk_limina_sampler_leak(void)
{
   static int cached = -1;
   if (cached < 0) {
      const char *e = getenv("LIMINA_KK_SAMPLER_LEAK");
      cached = (e && strcmp(e, "0") != 0) ? 1 : 0;
      if (cached)
         fprintf(stderr, "[LIMINA] KK sampler slots LEAKING (LIMINA_KK_SAMPLER_LEAK)\n");
   }
   return cached == 1;
}

DERIVE_HASH_TABLE(mtl_sampler_packed);

static VkResult
kk_init_sampler_heap(struct kk_device *dev, struct kk_sampler_heap *h)
{
   h->ht = mtl_sampler_packed_table_create(NULL);
   if (!h->ht)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   /* We optimistically size the table to fit the maximum number of samplers we
    * advertise. If this exceeds the hardware sampler limit, it is handled by
    * additional checks in `kk_sampler_heap_add_locked` */
   VkResult result = kk_query_table_init(dev, &h->table, MSL_MAX_SAMPLERS);

   if (result != VK_SUCCESS) {
      ralloc_free(h->ht);
      return result;
   }

   /* limina: the sampler table is dereferenced by raw GPU address out of argument-table slot 1,
    * and only ever by a shader that samples -- which is precisely the workload differential in
    * the WebGL MSAA device loss. kk_alloc_bo puts the heap in the residency set but never the
    * buffer placed on it, and kk_image.c already carries the finding that heap residency is not
    * enough for a texture on this driver. LIMINA_KK_SAMPTAB_RESIDENT makes the buffer itself
    * resident so that reading can be tested; the address is logged either way, so a kernel fault
    * VA can be checked against it. */
   kk_limina_addr_log("samplertab gpu=0x%llx..0x%llx",
                      (unsigned long long)h->table.bo->gpu,
                      (unsigned long long)(h->table.bo->gpu +
                                           (uint64_t)MSL_MAX_SAMPLERS * 8ull));
   {
      const char *e = getenv("LIMINA_KK_SAMPTAB_RESIDENT");
      if (e && strcmp(e, "0") != 0) {
         fprintf(stderr, "[LIMINA] KK sampler table pinned resident "
                         "(LIMINA_KK_SAMPTAB_RESIDENT)\n");
         kk_device_add_buffer_to_residency_set(dev, h->table.bo->map);
      }
   }

   simple_mtx_init(&h->lock, mtx_plain);
   return VK_SUCCESS;
}

static void
kk_destroy_sampler_heap(struct kk_device *dev, struct kk_sampler_heap *h)
{
   struct hash_entry *entry = _mesa_hash_table_next_entry(h->ht, NULL);
   while (entry) {
      struct kk_rc_sampler *sampler = (struct kk_rc_sampler *)entry->data;
      mtl_release(sampler->handle);
      entry = _mesa_hash_table_next_entry(h->ht, entry);
   }
   kk_query_table_finish(dev, &h->table);
   ralloc_free(h->ht);
   simple_mtx_destroy(&h->lock);
}

static VkResult
kk_sampler_heap_add_locked(struct kk_device *dev, struct kk_sampler_heap *h,
                           struct mtl_sampler_packed desc,
                           struct kk_rc_sampler **out)
{
   struct kk_physical_device *pdev = kk_device_physical(dev);

   struct hash_entry *ent = _mesa_hash_table_search(h->ht, &desc);
   if (ent != NULL) {
      *out = ent->data;

      assert((*out)->refcount != 0);
      (*out)->refcount++;

      return VK_SUCCESS;
   }

   /* Constrain to device max sampler count */
   if (h->ht->entries >= pdev->info.max_sampler_count)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct kk_rc_sampler *rc = ralloc(h->ht, struct kk_rc_sampler);
   if (!rc)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   mtl_sampler *handle = kk_sampler_create(dev, &desc);
   uint64_t gpu_id = mtl_sampler_get_gpu_resource_id(handle);

   uint32_t index;
   VkResult result = kk_query_table_add(dev, &h->table, gpu_id, &index);
   if (result != VK_SUCCESS) {
      mtl_release(handle);
      ralloc_free(rc);
      return result;
   }

   *rc = (struct kk_rc_sampler){
      .key = desc,
      .handle = handle,
      .refcount = 1,
      .index = index,
   };

   _mesa_hash_table_insert(h->ht, &rc->key, rc);
   *out = rc;

   return VK_SUCCESS;
}

VkResult
kk_sampler_heap_add(struct kk_device *dev, struct mtl_sampler_packed desc,
                    struct kk_rc_sampler **out)
{
   struct kk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   VkResult result = kk_sampler_heap_add_locked(dev, h, desc, out);
   simple_mtx_unlock(&h->lock);

   return result;
}

static void
kk_sampler_heap_remove_locked(struct kk_device *dev, struct kk_sampler_heap *h,
                              struct kk_rc_sampler *rc)
{
   assert(rc->refcount != 0);
   rc->refcount--;

   if (rc->refcount == 0) {
      /* limina: the descriptor a shader samples through holds a 16-bit INDEX into this
       * device-wide table, not the sampler's resource ID. Retiring the slot zeroes the
       * GPU-visible entry and hands the index straight back to the allocator, with nothing
       * waiting on the command buffers still executing against it. Counted (and, with
       * LIMINA_KK_SAMPLER_LEAK, suppressed) so an arm can say whether that ever happens. */
      if (kk_limina_sampler_leak())
         return;
      uint32_t n = p_atomic_inc_return(&kk_limina_sampler_retires);
      if ((n & 0xffu) == 1u)
         fprintf(stderr, "[LIMINA] KK sampler slot retired (#%u, index %u)\n", n,
                 rc->index);
      mtl_release(rc->handle);
      kk_query_table_remove(dev, &h->table, rc->index);
      _mesa_hash_table_remove_key(h->ht, &rc->key);
      ralloc_free(rc);
   }
}

void
kk_sampler_heap_remove(struct kk_device *dev, struct kk_rc_sampler *rc)
{
   struct kk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   kk_sampler_heap_remove_locked(dev, h, rc);
   simple_mtx_unlock(&h->lock);
}

static VkResult
kk_get_timestamp(struct vk_device *device, uint64_t *timestamp)
{
   struct kk_device *dev = container_of(device, struct kk_device, vk);

   uint64_t gpu_ns = mtl_device_get_gpu_timestamp(dev->mtl_handle);
   uint64_t frequency = mtl_device_timestamp_frequency(dev->mtl_handle);

   *timestamp =
      (uint64_t)(((unsigned __int128)gpu_ns * frequency) / 1000000000ull);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateDevice(VkPhysicalDevice physicalDevice,
                const VkDeviceCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);
   VkResult result = VK_ERROR_OUT_OF_HOST_MEMORY;
   struct kk_device *dev;

   dev = vk_zalloc2(&pdev->vk.instance->alloc, pAllocator, sizeof(*dev), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Fill the dispatch table we will expose to the users */
   dev->exposed_dispatch_table = kk_device_cmd_trampolines;
   vk_device_dispatch_table_from_entrypoints(&dev->exposed_dispatch_table,
                                             &kk_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dev->exposed_dispatch_table,
                                             &wsi_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dev->exposed_dispatch_table, &vk_common_device_entrypoints, false);

   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &kk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_common_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);

   result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table, pCreateInfo,
                           pAllocator);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   dev->vk.shader_ops = &kk_device_shader_ops;
   dev->mtl_handle = pdev->mtl_dev_handle;
   dev->vk.command_buffer_ops = &kk_cmd_buffer_ops;
   dev->vk.command_dispatch_table = &dev->vk.dispatch_table;
   dev->vk.get_timestamp = kk_get_timestamp;

   /* limina: the allocator pool must exist before any command buffer can record. It cannot
    * fail, so it goes first and the failure path below finishes it unconditionally. */
   kk_alloc_pool_init(dev);

   /* Create a new Metal pipeline compiler for the device */
   dev->mtl_compiler_handle = kk_acquire_compiler(dev);
   if (dev->mtl_compiler_handle == NULL)
      goto fail_init;

   /* We need to initialize the device residency set before any bo is created. */
   simple_mtx_init(&dev->residency_set.mutex, mtx_plain);
   dev->residency_set.handle = mtl_new_residency_set(dev->mtl_handle);
   if (dev->residency_set.handle == NULL)
      goto fail_compiler;

   if (pCreateInfo->queueCreateInfoCount > 0) {
      result =
         kk_queue_init(dev, &dev->queue, &pCreateInfo->pQueueCreateInfos[0], 0);
      if (result != VK_SUCCESS)
         goto fail_vab_memory;
      dev->has_queue = true;
   }

   result = kk_device_init_meta(dev);
   if (result != VK_SUCCESS)
      goto fail_mem_cache;

   result = kk_query_table_init(dev, &dev->occlusion_queries,
                                KK_MAX_OCCLUSION_QUERIES);
   if (result != VK_SUCCESS)
      goto fail_meta;

   result = kk_init_sampler_heap(dev, &dev->samplers);
   if (result != VK_SUCCESS)
      goto fail_query_table;

   result = kk_device_init_lib(dev);
   if (result != VK_SUCCESS)
      goto fail_sampler_heap;

   if (pdev->settings.gpu_capture_enabled) {
      const char *capture_directory =
         debug_get_option("MESA_KK_GPU_CAPTURE_DIRECTORY", NULL);
      mtl_start_gpu_capture(dev->mtl_handle, capture_directory);
   }

   *pDevice = kk_device_to_handle(dev);

   return VK_SUCCESS;

fail_sampler_heap:
   kk_destroy_sampler_heap(dev, &dev->samplers);
fail_query_table:
   kk_query_table_finish(dev, &dev->occlusion_queries);
fail_meta:
   kk_device_finish_meta(dev);
fail_mem_cache:
   if (dev->has_queue) {
      kk_queue_finish(dev, &dev->queue);
      dev->has_queue = false;
   }
fail_vab_memory:
   mtl_release(dev->residency_set.handle);
   simple_mtx_destroy(&dev->residency_set.mutex);
fail_compiler:
   kk_release_compiler(dev);
fail_init:
   kk_alloc_pool_finish(dev);
   vk_device_finish(&dev->vk);
fail_alloc:
   vk_free(&dev->vk.alloc, dev);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   if (!dev)
      return;

   /* End capture before we start releasing resources. Otherwise, Metal capture
    * may run into issues. */
   if (pdev->settings.gpu_capture_enabled) {
      mtl_stop_gpu_capture();
   }

   /* Meta first since it may destroy Vulkan objects */
   kk_device_finish_meta(dev);
   kk_device_finish_lib(dev);
   kk_query_table_finish(dev, &dev->occlusion_queries);
   kk_destroy_sampler_heap(dev, &dev->samplers);

   /* Geometry heap */
   if (dev->heap)
      kk_destroy_bo(dev, dev->heap);
   if (dev->fan_indices)
      kk_destroy_bo(dev, dev->fan_indices);
   if (dev->gs_sink)
      kk_destroy_bo(dev, dev->gs_sink);

   if (dev->has_queue) {
      kk_queue_finish(dev, &dev->queue);
      dev->has_queue = false;
   }

   /* limina: after the queue, whose internal command pool borrows from it. */
   kk_alloc_pool_finish(dev);

   /* Release the residency set last once all BOs are released. */
   mtl_release(dev->residency_set.handle);
   simple_mtx_destroy(&dev->residency_set.mutex);

   kk_release_compiler(dev);

   vk_device_finish(&dev->vk);

   vk_free(&dev->vk.alloc, dev);
}

/* We need to implement this ourselves so we give the fake ones for vk_common_*
 * to work when executing actual commands */
static PFN_vkVoidFunction
kk_device_get_proc_addr(const struct kk_device *device, const char *name)
{
   if (device == NULL || name == NULL)
      return NULL;

   struct vk_instance *instance = device->vk.physical->instance;
   return vk_device_dispatch_table_get_if_supported(
      &device->exposed_dispatch_table, name, instance->app_info.api_version,
      &instance->enabled_extensions, &device->vk.enabled_extensions);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
kk_GetDeviceProcAddr(VkDevice _device, const char *pName)
{
   VK_FROM_HANDLE(kk_device, device, _device);
   return kk_device_get_proc_addr(device, pName);
}

uint32_t kk_limina_resident_heaps, kk_limina_resident_buffers,
   kk_limina_resident_textures;

void
kk_device_add_heap_to_residency_set(struct kk_device *dev, mtl_heap *heap)
{
   if (unlikely(heap == NULL))
      return;

   p_atomic_inc(&kk_limina_resident_heaps);
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, heap);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_heap_from_residency_set(struct kk_device *dev, mtl_heap *heap)
{
   if (unlikely(heap == NULL))
      return;

   p_atomic_dec(&kk_limina_resident_heaps);
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_remove_allocation(dev->residency_set.handle, heap);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_add_buffer_to_residency_set(struct kk_device *dev, mtl_buffer *buffer)
{
   p_atomic_inc(&kk_limina_resident_buffers);
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, buffer);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_buffer_from_residency_set(struct kk_device *dev,
                                           mtl_buffer *buffer)
{
   p_atomic_dec(&kk_limina_resident_buffers);
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_remove_allocation(dev->residency_set.handle, buffer);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

/* An imported MTLTexture is its own allocation — it belongs to no heap of ours,
 * so it must be made resident in its own right or the first submit that samples
 * or renders to it faults. */
void
kk_device_add_texture_to_residency_set(struct kk_device *dev,
                                       mtl_texture *texture)
{
   /* limina: a NULL allocation is accepted here and then dereferenced by
    * -[AGXG13XFamilyResidencySet _commitAddedAllocations:count:], which segfaults at 0x18 --
    * inside Apple's driver, on a queue submit, far from whoever added it. Refuse it here and
    * name the caller, so the same mistake is a log line instead of a crash. */
   if (unlikely(texture == NULL)) {
      static uint32_t warned = 0u;
      if (warned < 4u) {
         ++warned;
         fprintf(stderr, "[LIMINA-RESIDENCY] refused a NULL texture, called from %p\n",
                 __builtin_return_address(0));
         fflush(stderr);
      }
      return;
   }

   p_atomic_inc(&kk_limina_resident_textures);
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, texture);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_texture_from_residency_set(struct kk_device *dev,
                                            mtl_texture *texture)
{
   if (unlikely(texture == NULL))
      return;

   p_atomic_dec(&kk_limina_resident_textures);
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_remove_allocation(dev->residency_set.handle, texture);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_make_resources_resident(struct kk_device *dev)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_commit(dev->residency_set.handle);
   mtl_residency_set_request_residency(dev->residency_set.handle);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

/* VK_EXT_debug_utils */
VKAPI_ATTR VkResult VKAPI_CALL
kk_SetDebugUtilsObjectNameEXT(VkDevice device,
                              const VkDebugUtilsObjectNameInfoEXT *pNameInfo)
{
   VkResult result = vk_common_SetDebugUtilsObjectNameEXT(device, pNameInfo);
   if (result != VK_SUCCESS)
      return result;

   if (pNameInfo->pObjectName == NULL)
      return result;

#define CASE(obj_type, kk_type, vk_type)                                       \
   case VK_OBJECT_TYPE_##obj_type: {                                           \
      VK_FROM_HANDLE(kk_type, obj, (vk_type)pNameInfo->objectHandle);          \
      kk_type##_set_label(obj, pNameInfo->pObjectName);                        \
      break;                                                                   \
   }

   switch (pNameInfo->objectType) {
      CASE(COMMAND_BUFFER, kk_cmd_buffer, VkCommandBuffer);
      CASE(COMMAND_POOL, kk_cmd_pool, VkCommandPool);
      CASE(DEVICE_MEMORY, kk_device_memory, VkDeviceMemory);
      CASE(BUFFER, kk_buffer, VkBuffer);
      CASE(BUFFER_VIEW, kk_buffer_view, VkBufferView);
      CASE(IMAGE, kk_image, VkImage);
      CASE(IMAGE_VIEW, kk_image_view, VkImageView);
      CASE(DESCRIPTOR_POOL, kk_descriptor_pool, VkDescriptorPool);
      /* TODO: it would be nice to label pipelines and shaders, but we don't
       * have easy access to the relevant vulkan runtime objects inside
       * kk_compile_shaders(), where the Metal objects are set up.
       */
   default:
      break;
   }
#undef CASE

   return VK_SUCCESS;
}
