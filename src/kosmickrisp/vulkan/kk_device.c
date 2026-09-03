/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_device.h"

#include <signal.h>
#include <unistd.h>
#include <string.h>

#include "kk_cmd_buffer.h"
#include "kk_entrypoints.h"
#include "kk_instance.h"
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

#include <time.h>

/* limina: the shared command-allocator pool. Rationale and the measurements behind it are in
 * the block comment on struct kk_alloc_pool (kk_device.h) and spikes/vrend-region-leak/. */

#define KK_ALLOC_BUDGET_MIB_DEFAULT 4
#define KK_ALLOC_POOL_WATERMARK 64
/* Idle-decay before a drained allocator counts as surplus. Long enough that a burst which merely
 * paused between frames keeps its working set; short enough that an app exit is reclaimed while
 * the compositor is still drawing (reclaim is acquire-driven, so a fully idle guest holds its
 * high-water until activity resumes — which is exactly when the memory is wanted again). */
#define KK_ALLOC_DECAY_MS_DEFAULT 2000
/* Never destroy below this many per class. NOT peak_live: peak_live only ever grows, so a floor
 * there would pin the pool at its all-time high-water and reclaim nothing after a heavy app
 * exits, which is the entire point. A small constant protects steady-state concurrency and lets
 * the decay clock do the real work. */
#define KK_ALLOC_FLOOR_DEFAULT 8

/* Seconds between pool reports. A count of encoder closes reads as conservative and is not: a
 * busy desktop closes thousands a second, so the cadence of a line meant for human eyes ends up
 * set by how hard the GPU is working. Pace it by the clock instead. */
#define KK_ALLOC_REPORT_SECS 10u

/* Encoder closes between on-disk snapshot attempts. The snapshot has its own one-second clock;
 * this only keeps the time read off the very hottest path. */
#define KK_ALLOC_SNAPSHOT_EVERY 25u

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

/* limina: the use-after-destroy detector. A destroyed allocator keeps its struct, stamped DEAD,
 * so every pool entry point can tell a stale pointer from a live one. Returns false when the
 * pointer is stale, having already said so; the caller must then do nothing with it.
 *
 * Loud by design: this is hunting a fault that otherwise lands as a nil store deep inside AGX,
 * where the stack names Apple's code and not ours. LIMINA_KK_ALLOC_GUARD=abort promotes the
 * report to an abort() so a core dump is taken at the moment of misuse rather than later. */
static bool
kk_pa_live(struct kk_alloc_pool *pool, struct kk_pooled_alloc *pa, const char *where)
{
   if (likely(pa->magic == KK_PA_LIVE))
      return true;

   pool->use_after_destroy++;
   fprintf(stderr,
           "[LIMINA-ALLOC-POOL] USE AFTER DESTROY: %s() reached allocator %p with magic %08x "
           "(class %u, %u begins) — this allocator was destroyed while something still held it\n",
           where, (void *)pa, pa->magic, pa->klass, pa->begins);
   fflush(stderr);

   static int mode = -1;
   if (mode < 0) {
      const char *e = getenv("LIMINA_KK_ALLOC_GUARD");
      mode = (e && !strcmp(e, "abort")) ? 1 : 0;
   }
   if (mode)
      abort();
   return false;
}

void
kk_alloc_pool_report(struct kk_device *dev, const char *why)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;
   simple_mtx_lock(&pool->mtx);
   for (unsigned k = 0; k < KK_ALLOC_CLASS_COUNT; ++k) {
      fprintf(stderr,
              "[LIMINA-ALLOC-POOL] %s %s: live=%u peak=%u retired=%u | size hiwater=%llu KiB "
              "(budget %llu KiB, %u retirements) | peak ops/cmdbuf=%u\n",
              why, k == KK_ALLOC_CLASS_RENDER ? "render" : "compute", pool->live[k],
              pool->peak_live[k], pool->destroyed[k],
              (unsigned long long)(pool->hiwater_bytes[k] >> 10),
              (unsigned long long)(pool->budget_bytes >> 10), pool->over_budget[k],
              pool->peak_ops[k]);
      fprintf(stderr,
              "[LIMINA-ALLOC-POOL] %s %s growth: %u MiB crossings | peak ops between resets=%u\n",
              why, k == KK_ALLOC_CLASS_RENDER ? "render" : "compute", pool->growths[k],
              pool->peak_ops_since_reset[k]);
   }
   fprintf(stderr,
           "[LIMINA-ALLOC-POOL] %s tombstones=%u use-after-destroy=%u | resets=%u/%u "
           "(render/compute) unmatched-discharge=%u\n",
           why, (unsigned)util_dynarray_num_elements(&pool->tombstones, kk_pooled_alloc_ptr),
           pool->use_after_destroy, pool->resets[KK_ALLOC_CLASS_RENDER],
           pool->resets[KK_ALLOC_CLASS_COMPUTE], pool->unmatched_discharge);
   fflush(stderr);
   simple_mtx_unlock(&pool->mtx);
}

__thread struct kk_pooled_alloc *kk_tls_open_alloc;

/* limina: keep a current pool snapshot on disk, for reading AFTER a crash.
 *
 * The periodic stderr report can be thousands of encoder closes stale when a rare fault lands,
 * and the obvious fix — a SIGSEGV handler — is wrong here: libkrun installs its own SIGSEGV and
 * SIGBUS handler for the released-RAM sweep and legitimately RESUMES from those faults, so a hook
 * ahead of it would fire on a hot VMM path rather than only on a real crash.
 *
 * So write the state to a file instead, truncated each time. Nothing to unwind, no signal-safety
 * question, and the last snapshot survives whatever killed the process.
 * LIMINA_KK_POOL_SNAPSHOT=<path> enables it; unset means no file is written. */
static void
kk_pool_snapshot(struct kk_alloc_pool *pool)
{
   static const char *base;
   static bool looked_up;
   if (!looked_up) {
      base = getenv("LIMINA_KK_POOL_SNAPSHOT");
      if (base && !base[0])
         base = NULL;
      looked_up = true;
   }
   if (!base)
      return;

   /* A worker holds MANY VkDevices — one per venus context — and each has its own pool. Writing
    * them all to one path made the file whichever device wrote last, so its counters appeared to
    * move BACKWARDS between samples. One file per pool instead. */
   if (!pool->snapshot_path[0]) {
      snprintf(pool->snapshot_path, sizeof(pool->snapshot_path), "%s.%p", base, (void *)pool);
   }
   /* And rate-limit by TIME, not by encoder closes: at 2000 closes a second the every-25 cadence
    * was ~80 opens a second per device, on a hot path, for a file nothing reads until a crash. */
   const uint64_t now = os_time_get_nano();
   if (pool->snapshot_last_ns && now - pool->snapshot_last_ns < 1000000000ull)
      return;
   pool->snapshot_last_ns = now;

   FILE *f = fopen(pool->snapshot_path, "w");
   if (!f)
      return;
   for (unsigned k = 0; k < KK_ALLOC_CLASS_COUNT; ++k) {
      fprintf(f,
              "%s live=%u peak=%u destroyed=%u hiwater_kib=%llu retirements=%u resets=%u "
              "peak_ops=%u growths=%u peak_ops_since_reset=%u\n",
              k == KK_ALLOC_CLASS_RENDER ? "render" : "compute", pool->live[k], pool->peak_live[k],
              pool->destroyed[k], (unsigned long long)(pool->hiwater_bytes[k] >> 10),
              pool->over_budget[k], pool->resets[k], pool->peak_ops[k], pool->growths[k],
              pool->peak_ops_since_reset[k]);
   }
   fprintf(f, "tombstones=%u use_after_destroy=%u unmatched_discharge=%u releases=%u\n",
           (unsigned)util_dynarray_num_elements(&pool->tombstones, kk_pooled_alloc_ptr),
           pool->use_after_destroy, pool->unmatched_discharge, pool->releases);

   /* One line per LIVE allocator. The class aggregates above are lifetime maxima, which is the
    * wrong tense for a post-mortem: the four snapshots left by the 2026-09-02 crash said nothing
    * the periodic stderr report had not, because "the largest any allocator ever got" cannot
    * identify the one that was in trouble. These lines can — above all `ops_since_reset` and
    * `mib_seen` for the allocator an encoder was holding when the process died.
    *
    * Bounded by the pool's floor and peak (low hundreds of lines), and written at most once a
    * second by the rate limit above. */
   for (unsigned k = 0; k < KK_ALLOC_CLASS_COUNT; ++k) {
      util_dynarray_foreach(&pool->allocs[k], kk_pooled_alloc_ptr, pap) {
         const struct kk_pooled_alloc *pa = *pap;
         fprintf(f,
                 "alloc %p class=%u magic=%08x in_use=%d draining=%d pending=%u begins=%u "
                 "resets=%u ops_since_reset=%u mib=%u peak_kib=%llu\n",
                 (const void *)pa, (unsigned)pa->klass, pa->magic, (int)pa->in_use,
                 (int)pa->draining, pa->pending, pa->begins, pa->resets, pa->ops_since_reset,
                 pa->mib_seen, (unsigned long long)(pa->peak_bytes >> 10));
      }
   }
   fclose(f);
}

void
kk_alloc_pool_init(struct kk_device *dev)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;

   simple_mtx_init(&pool->mtx, mtx_plain);
   for (unsigned i = 0; i < KK_ALLOC_CLASS_COUNT; ++i)
      util_dynarray_init(&pool->allocs[i], NULL);
   util_dynarray_init(&pool->tombstones, NULL);

   /* The budget is the whole knob: an allocator leaves service once it holds this much, so the
    * steady state is roughly (pool size) x (budget). Overridable for the A/B ladder. */
   pool->budget_bytes =
      kk_env_u64("LIMINA_KK_ALLOC_BUDGET_MIB", KK_ALLOC_BUDGET_MIB_DEFAULT) * 1024u * 1024u;

   pool->decay_ns =
      kk_env_u64("LIMINA_KK_ALLOC_DECAY_MS", KK_ALLOC_DECAY_MS_DEFAULT) * 1000000ull;
   pool->floor = (uint32_t)kk_env_u64("LIMINA_KK_ALLOC_FLOOR", KK_ALLOC_FLOOR_DEFAULT);
   pool->destroy_enabled = kk_env_u64("LIMINA_KK_ALLOC_DESTROY", 1) != 0;
}

/* Pick at most one surplus allocator, unlink it, and hand its Metal handle back to be released
 * OUTSIDE the pool mutex — the release does kernel unmap work of potentially multi-millisecond
 * cost, and every encoder thread contends on this lock from cs_start_render.
 *
 * `keep` is the allocator this acquire is about to hand out; never consider it. Called only when
 * the acquire already has a usable allocator, so a destroy can never be followed by a mint in the
 * same call — that pairing is exactly the thrash the pool exists to avoid, and it would bite
 * hardest at the concurrency edge (peak 184 command buffers in flight).
 *
 * Caller must hold pool->mtx. */
static mtl_command_allocator *
kk_alloc_pool_take_surplus(struct kk_alloc_pool *pool, enum kk_alloc_class klass,
                           const struct kk_pooled_alloc *keep, uint64_t now)
{
   if (!pool->destroy_enabled || pool->live[klass] <= pool->floor)
      return NULL;

   struct kk_pooled_alloc *victim = NULL;
   util_dynarray_foreach(&pool->allocs[klass], kk_pooled_alloc_ptr, pap) {
      struct kk_pooled_alloc *pa = *pap;
      if (pa == keep || pa->in_use || !pa->draining || pa->pending != 0)
         continue;
      if (pa->idle_since == 0)
         continue;
      if (now - pa->idle_since < pool->decay_ns)
         continue;
      victim = pa;
      break;
   }

   if (!victim)
      return NULL;

   mtl_command_allocator *handle = victim->handle;
   util_dynarray_delete_unordered(&pool->allocs[klass], kk_pooled_alloc_ptr, victim);
   pool->live[klass]--;
   pool->destroyed[klass]++;
   /* limina: stamp and keep rather than free. The struct is ~48 bytes and this is the whole
    * detector — freeing it would turn a stale pointer back into unreadable heap. If the
    * tombstone list cannot grow we fall back to freeing: losing detection beats leaking. */
   victim->magic = KK_PA_DEAD;
   victim->handle = NULL;
   kk_pooled_alloc_ptr *grave = util_dynarray_grow(&pool->tombstones, kk_pooled_alloc_ptr, 1);
   if (grave)
      *grave = victim;
   else
      free(victim);

   /* Reclaim must not be invisible: growth already warns, so shrink gets the same treatment.
    * Off by default — during a post-workload drain this fires once per acquire. */
   static int log_reclaim = -1;
   if (log_reclaim < 0)
      log_reclaim = getenv("LIMINA_KK_ALLOC_POOL_LOG") != NULL;
   if (log_reclaim)
      fprintf(stderr, "[LIMINA-ALLOC-POOL] retired one class-%u allocator; live=%u destroyed=%u\n",
              klass, pool->live[klass], pool->destroyed[klass]);

   return handle;
}

void
kk_alloc_pool_finish(struct kk_device *dev)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;

   kk_alloc_pool_report(dev, "teardown");

   /* Device teardown frees whatever the reclaim policy did not: every context is gone. */
   for (unsigned i = 0; i < KK_ALLOC_CLASS_COUNT; ++i) {
      util_dynarray_foreach(&pool->allocs[i], kk_pooled_alloc_ptr, pap) {
         struct kk_pooled_alloc *pa = *pap;
         mtl_release(pa->handle);
         free(pa);
      }
      util_dynarray_fini(&pool->allocs[i]);
   }
   util_dynarray_foreach(&pool->tombstones, kk_pooled_alloc_ptr, pap)
      free(*pap);
   util_dynarray_fini(&pool->tombstones);

   /* Drop this pool's snapshot. The file exists so a crash leaves the pool's last state on
    * disk; a pool that reached its own destructor did not crash and has nothing to say. Which
    * makes the leftovers the useful signal — after a crash, the files still present name the
    * pools that were live when it happened. Without this the directory only ever grows: one
    * file per VkDevice per run, and a desktop session creates dozens. */
   if (pool->snapshot_path[0])
      unlink(pool->snapshot_path);

   simple_mtx_destroy(&pool->mtx);
}

struct kk_pooled_alloc *
kk_alloc_pool_acquire(struct kk_device *dev, enum kk_alloc_class klass)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;
   struct kk_pooled_alloc *found = NULL;
   mtl_command_allocator *doomed = NULL;
   const uint64_t now = os_time_get_nano();

   simple_mtx_lock(&pool->mtx);

   /* Pass 1: an allocator that is ready as-is — not borrowed, not over budget. */
   util_dynarray_foreach(&pool->allocs[klass], kk_pooled_alloc_ptr, pap) {
      struct kk_pooled_alloc *pa = *pap;
      if (!pa->in_use && !pa->draining) {
         found = pa;
         break;
      }
   }

   if (!found) {
      /* Pass 2: a retired one that has drained. It becomes usable again only once every command
       * buffer begun on it has completed — Apple's stated precondition for reset, and the reason
       * the charge happens at begin rather than at commit. Reset lazily here, on an app thread,
       * rather than on the completion callback. */
      util_dynarray_foreach(&pool->allocs[klass], kk_pooled_alloc_ptr, pap) {
         struct kk_pooled_alloc *pa = *pap;
         if (pa->in_use || pa->pending != 0)
            continue;
         assert(pa->draining);
         pool->resets[klass]++;
         pa->resets++;
         /* A reset is where AGX's own pool goes back to one segment, so the ops counter that
          * matters for the next growth starts here. */
         pa->ops_since_reset = 0;
         mtl_command_allocator_reset(pa->handle);
         pa->draining = false;
         pa->idle_since = 0;
         found = pa;
         break;
      }
   }

   /* Retire surplus only once this call is already served from the pool. Gating this on "pass 1
    * succeeded" instead was wrong, and measured so: allocatedSize never shrinks, so a render
    * allocator is permanently over budget once it crosses it. In steady state every render
    * allocator is draining, pass 1 never succeeds, and destroy fired 66 times for compute and
    * ZERO times for render — the one class that holds the memory. What actually matters is only
    * that we never destroy in a call that then has to mint, and `found != NULL` here is exactly
    * that condition. */
   if (found)
      doomed = kk_alloc_pool_take_surplus(pool, klass, found, now);

   if (!found) {
      /* Never block waiting for a drain: stalling cs_start_render on GPU progress invites jank
       * and priority inversion, and the in-flight depth that drives this is already bounded by
       * the client's own fencing. Mint instead, and make growth loud. */
      found = calloc(1, sizeof(*found));
      if (!found) {
         simple_mtx_unlock(&pool->mtx);
         return NULL;
      }
      found->handle = mtl_new_command_allocator(dev->mtl_handle);
      if (!found->handle) {
         free(found);
         simple_mtx_unlock(&pool->mtx);
         return NULL;
      }
      found->klass = klass;
      found->magic = KK_PA_LIVE;
      kk_pooled_alloc_ptr *slot =
         util_dynarray_grow(&pool->allocs[klass], kk_pooled_alloc_ptr, 1);
      if (!slot) {
         mtl_release(found->handle);
         free(found);
         simple_mtx_unlock(&pool->mtx);
         return NULL;
      }
      *slot = found;
      pool->live[klass]++;
      if (pool->live[klass] > pool->peak_live[klass])
         pool->peak_live[klass] = pool->live[klass];
      if (pool->live[klass] > KK_ALLOC_POOL_WATERMARK &&
          pool->live[klass] > pool->watermark_warned) {
         pool->watermark_warned = pool->live[klass];
         fprintf(stderr,
                 "[LIMINA-ALLOC-POOL] class %u grew to %u allocators (budget %llu MiB) — "
                 "in-flight depth is outrunning completion\n",
                 klass, pool->live[klass],
                 (unsigned long long)(pool->budget_bytes >> 20));
      }
   }

   found->in_use = true;
   found->idle_since = 0;
   simple_mtx_unlock(&pool->mtx);

   /* Outside the lock on purpose: releasing an allocator unmaps its heaps in the kernel, and
    * cs_start_render must not serialise behind that. Measured to return 100% of the heaps
    * (spikes/vrend-region-leak/mtl4-repro/destroy-probe.m). */
   if (doomed)
      mtl_release(doomed);

   return found;
}

void
kk_alloc_pool_release(struct kk_device *dev, struct kk_pooled_alloc *pa, uint32_t ops)
{
   struct kk_alloc_pool *pool = &dev->alloc_pool;
   if (!pa)
      return;

   /* Check staleness BEFORE touching the handle: a destroyed allocator's handle is NULL, and
    * this is the one entry point that dereferences it. The unlocked read is sound — magic is
    * written once, under the lock, at destroy — and a false "live" here is caught a line later. */
   if (unlikely(pa->magic != KK_PA_LIVE)) {
      simple_mtx_lock(&pool->mtx);
      kk_pa_live(pool, pa, "kk_alloc_pool_release");
      simple_mtx_unlock(&pool->mtx);
      return;
   }

   /* Read the size OUTSIDE the lock: this is a Metal call on the encode hot path. */
   uint64_t size = mtl_command_allocator_allocated_size(pa->handle);

   simple_mtx_lock(&pool->mtx);
   if (!kk_pa_live(pool, pa, "kk_alloc_pool_release")) {
      simple_mtx_unlock(&pool->mtx);
      return;
   }
   pa->in_use = false;
   if (size > pa->peak_bytes)
      pa->peak_bytes = size;
   if (size > pool->hiwater_bytes[pa->klass])
      pool->hiwater_bytes[pa->klass] = size;
   if (ops > pool->peak_ops[pa->klass])
      pool->peak_ops[pa->klass] = ops;

   /* limina (growth trace): the work this allocator has taken on since its last reset, and the
    * whole-MiB boundaries it has crossed. The fault under investigation is AGX failing to chain
    * a second 1 MiB segment, so "how far past a boundary, after how much work, how many resets
    * in" is the shape of the evidence wanted at the next crash. */
   pa->ops_since_reset += ops;
   if (pa->ops_since_reset > pool->peak_ops_since_reset[pa->klass])
      pool->peak_ops_since_reset[pa->klass] = pa->ops_since_reset;

   const uint32_t mib = (uint32_t)(size >> 20);
   if (mib > pa->mib_seen) {
      const uint32_t from = pa->mib_seen;
      pa->mib_seen = mib;
      pool->growths[pa->klass]++;
      /* Counted always, logged rarely. Every allocator crosses several boundaries and the pool
       * retires them by the thousand, so a line per crossing is a flood — and a flood is how the
       * last two diagnostics here went wrong. One line a second is for the human; the snapshot
       * below carries the full per-allocator detail for the post-mortem. */
      /* A growth on an allocator that has ALREADY been reset is the event the hypothesis is
       * about — a reset leaving the segment chain short — and it is rare, because mib_seen is a
       * lifetime maximum, so an allocator only ever grows past its own high-water. The startup
       * burst is all resets=0, so exempting this case costs nothing and never loses the one
       * crossing worth having. */
      const bool after_reset = pa->resets > 0;
      const uint64_t now = os_time_get_nano();
      if (after_reset || now - pool->growth_log_last_ns >= 1000000000ull) {
         pool->growth_log_last_ns = now;
         fprintf(stderr,
                 "[LIMINA-ALLOC-POOL] GROWTH alloc=%p class=%u %u->%u MiB | ops=%u "
                 "ops_since_reset=%u begins=%u resets=%u pending=%u (%u crossings this class)\n",
                 (void *)pa, pa->klass, from, mib, ops, pa->ops_since_reset, pa->begins,
                 pa->resets, pa->pending, pool->growths[pa->klass]);
         fflush(stderr);
      }
   }
   /* Retire on budget. Note this is checked at encoder end, not at reset time: allocatedSize
    * never shrinks, so by the time a reset-time check fires the growth is already permanent. */
   if (pool->budget_bytes && size >= pool->budget_bytes) {
      if (!pa->draining)
         pool->over_budget[pa->klass]++;
      pa->draining = true;
   }
   /* Start the decay clock if it is already drained; otherwise the last discharge does it. */
   if (pa->draining && pa->pending == 0)
      pa->idle_since = os_time_get_nano();
   /* Riding the release path rather than a tick keeps the device pointer in hand; the clock,
    * not the release count, decides when a line is due. */
   pool->releases++;
   const uint64_t now_ns = os_time_get_nano();
   const bool due = now_ns - pool->report_last_ns >= (uint64_t)KK_ALLOC_REPORT_SECS * 1000000000ull;
   if (due)
      pool->report_last_ns = now_ns;
   /* The snapshot is a few hundred bytes over a truncating write, so it can run far more often
    * than the log line — the point is that it is never stale when a crash reads it. */
   if (pool->releases % KK_ALLOC_SNAPSHOT_EVERY == 0)
      kk_pool_snapshot(pool);
   simple_mtx_unlock(&pool->mtx);

   if (due)
      kk_alloc_pool_report(dev, "periodic");
}

void
kk_alloc_pool_charge(struct kk_device *dev, struct kk_pooled_alloc *pa)
{
   if (!pa)
      return;
   simple_mtx_lock(&dev->alloc_pool.mtx);
   if (kk_pa_live(&dev->alloc_pool, pa, "kk_alloc_pool_charge")) {
      pa->pending++;
      pa->begins++;
   }
   simple_mtx_unlock(&dev->alloc_pool.mtx);
}

void
kk_alloc_pool_discharge(struct kk_device *dev, struct kk_pooled_alloc *pa)
{
   if (!pa)
      return;
   simple_mtx_lock(&dev->alloc_pool.mtx);
   if (!kk_pa_live(&dev->alloc_pool, pa, "kk_alloc_pool_discharge")) {
      simple_mtx_unlock(&dev->alloc_pool.mtx);
      return;
   }
   assert(pa->pending > 0);
   if (pa->pending > 0) {
      pa->pending--;
   } else {
      /* An unmatched discharge means the charge ledger is broken, and the next acquire may reset
       * this allocator while the GPU still reads it. Say so where it happens rather than leaving
       * it to surface as a nil store inside AGX. */
      dev->alloc_pool.unmatched_discharge++;
      fprintf(stderr,
              "[LIMINA-ALLOC-POOL] UNMATCHED DISCHARGE on allocator %p (class %u, %u begins) — "
              "pending was already 0, so a reset can now land under live GPU work\n",
              (void *)pa, pa->klass, pa->begins);
      fflush(stderr);
   }
   /* The GPU is done with everything begun here, so a retired allocator is now surplus-eligible
    * and its decay clock starts. Not eligible while borrowed — release() stamps that case. */
   if (pa->pending == 0 && pa->draining && !pa->in_use)
      pa->idle_since = os_time_get_nano();
   simple_mtx_unlock(&dev->alloc_pool.mtx);
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
   if (dev->disabled_workarounds & BITFIELD64_BIT(11)) {
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
   if (dev->disabled_workarounds & BITFIELD64_BIT(11)) {
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

static void
kk_parse_device_environment_options(struct kk_device *dev)
{
   dev->gpu_capture_enabled =
      debug_get_bool_option("MESA_KK_GPU_CAPTURE", false);
   if (dev->gpu_capture_enabled) {
      const char *capture_directory =
         debug_get_option("MESA_KK_GPU_CAPTURE_DIRECTORY", NULL);
      mtl_start_gpu_capture(dev->mtl_handle, capture_directory);
   }

   const char *list = debug_get_option("MESA_KK_DISABLE_WORKAROUNDS", "");
   const char *all_workarounds = "all";
   const size_t all_len = strlen(all_workarounds);
   for (unsigned n; n = strcspn(list, ","), *list; list += MAX2(1, n)) {
      if (n == all_len && !strncmp(list, all_workarounds, n)) {
         dev->disabled_workarounds = UINT64_MAX;
         break;
      }

      int index = atoi(list);
      dev->disabled_workarounds |= BITFIELD64_BIT(index);
   }

   /* Workarounds resolved on macOS 27 */
   if (ns_is_os_version_at_least(27, 0, 0)) {
      dev->disabled_workarounds |= BITFIELD64_MASK(7);
      dev->disabled_workarounds |= BITFIELD64_BIT(12);
   }
   /* M5-only workarounds */
   if (kk_device_physical(dev)->info.gpu_apple_family < 10) {
      dev->disabled_workarounds |= BITFIELD64_BIT(16);
   }
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

   kk_parse_device_environment_options(dev);

   /* Create a new Metal pipeline compiler for the device */
   dev->mtl_compiler_handle = kk_acquire_compiler(dev);
   if (dev->mtl_compiler_handle == NULL)
      goto fail_init;

   /* limina: the allocator pool must exist before any command buffer can encode. */
   kk_alloc_pool_init(dev);

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
   vk_device_finish(&dev->vk);
fail_alloc:
   vk_free(&dev->vk.alloc, dev);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, _device);

   if (!dev)
      return;

   /* End capture before we start releasing resources. Otherwise, Metal capture
    * may run into issues. */
   if (dev->gpu_capture_enabled) {
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

   if (dev->has_queue) {
      kk_queue_finish(dev, &dev->queue);
      dev->has_queue = false;
   }

   /* limina: every context is gone by here, which is the only point at which command
    * allocators are destroyed. Must follow kk_queue_finish so nothing is still in flight. */
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

void
kk_device_add_heap_to_residency_set(struct kk_device *dev, mtl_heap *heap)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, heap);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_heap_from_residency_set(struct kk_device *dev, mtl_heap *heap)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_remove_allocation(dev->residency_set.handle, heap);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_add_buffer_to_residency_set(struct kk_device *dev, mtl_buffer *buffer)
{
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, buffer);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_buffer_from_residency_set(struct kk_device *dev,
                                           mtl_buffer *buffer)
{
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
   simple_mtx_lock(&dev->residency_set.mutex);
   mtl_residency_set_add_allocation(dev->residency_set.handle, texture);
   simple_mtx_unlock(&dev->residency_set.mutex);
}

void
kk_device_remove_texture_from_residency_set(struct kk_device *dev,
                                            mtl_texture *texture)
{
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
