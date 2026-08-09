/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_bo.h"

#include "kk_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/u_memory.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if DETECT_OS_APPLE
#include <mach/mach_init.h>
#include <mach/mach_vm.h>
#endif /* DETECT_OS_APPLE */

/* limina: BO census — LIMINA_KK_BOCENSUS=<secs>, 0/unset = off.
 *
 * Every kk_alloc_bo creates one MTLHeap plus one MTLBuffer, and each of those is an
 * `IOAccelerator (graphics)` VM region in the host worker. So the live count here IS the
 * quantity that ratchets in the vrend region leak (perf/2026-08-08-remeasure.md,
 * spikes/vrend-region-leak/): 3 833 regions at a fresh idle boot, 28 000 after ONE WebGL
 * open/close cycle, never returning.
 *
 * WHY THIS COUNTER HAD TO EXIST. virglrenderer already has a GPU-memory ledger and a Metal
 * refcount census (vkr_budget.h), and both are BLIND to this: measured across a leak cycle
 * they sat at "11.7 MiB live, 4 charges" and "iosurface +3, texture +0" while the process
 * grew by 3.2 GB. Nothing leaking passes a charged call site, because these allocations are
 * made by KosmicKrisp underneath the GL stack rather than by virglrenderer itself. This is
 * the missing half of the accounting, at the one place that actually mints the objects.
 *
 * Cost: two atomics on a path that already creates a Metal heap. The env read is cached —
 * a getenv on a hot path takes the environ lock and has twice cost us real throughput.
 */
static atomic_ullong kk_bo_census_alloc, kk_bo_census_free;
static atomic_ullong kk_bo_census_live_B, kk_bo_census_peak_B;
/* MTLTextures are minted on a different path from BOs (kk_image_plane_create_texture, plus
 * several mtl_retain sites where a plane adopts an existing texture), and they are their own
 * IOAccelerator allocations. Counting only BOs was not enough: one WebGL open/close cycle grew
 * the process by ~24 000 regions while live BOs moved by well under 200. */
atomic_ullong kk_tex_census_acquire, kk_tex_census_release;
static atomic_llong kk_bo_census_secs = -1;
static atomic_ullong kk_bo_census_last_ns;

static uint64_t
kk_bo_census_now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void
kk_bo_census_tick(void)
{
   long long secs = atomic_load(&kk_bo_census_secs);
   if (secs < 0) {
      const char *e = getenv("LIMINA_KK_BOCENSUS");
      secs = (e && *e) ? strtoll(e, NULL, 10) : 0;
      atomic_store(&kk_bo_census_secs, secs);
      if (secs > 0)
         atomic_store(&kk_bo_census_last_ns, kk_bo_census_now_ns());
   }
   if (secs <= 0)
      return;

   uint64_t now = kk_bo_census_now_ns();
   uint64_t last = atomic_load(&kk_bo_census_last_ns);
   if (now - last < (uint64_t)secs * 1000000000ull)
      return;
   /* One reporter per window; a loser of this race just skips the tick. */
   if (!atomic_compare_exchange_strong(&kk_bo_census_last_ns, &last, now))
      return;

   unsigned long long a = atomic_load(&kk_bo_census_alloc);
   unsigned long long f = atomic_load(&kk_bo_census_free);
   unsigned long long live = atomic_load(&kk_bo_census_live_B);
   unsigned long long peak = atomic_load(&kk_bo_census_peak_B);
   unsigned long long ta = atomic_load(&kk_tex_census_acquire);
   unsigned long long tr = atomic_load(&kk_tex_census_release);
   fprintf(stderr,
           "[LIMINA-KK-BOCENSUS] bo alloc=%llu free=%llu live=%lld (bytes live=%.1f MiB "
           "peak=%.1f MiB) | tex acquire=%llu release=%llu live=%lld | 1 bo = MTLHeap + "
           "MTLBuffer; each texture is its own allocation too\n",
           a, f, (long long)(a - f), (double)live / (1024.0 * 1024.0),
           (double)peak / (1024.0 * 1024.0), ta, tr, (long long)(ta - tr));

   /* The per-class histogram is the one that names a leak outright, so it rides the same
    * timer rather than needing its own knob. */
   limina_mtl_class_census_dump();
}

static void
kk_bo_census_charge(uint64_t size_B)
{
   atomic_fetch_add(&kk_bo_census_alloc, 1ull);
   uint64_t live = atomic_fetch_add(&kk_bo_census_live_B, size_B) + size_B;
   uint64_t peak = atomic_load(&kk_bo_census_peak_B);
   while (live > peak &&
          !atomic_compare_exchange_weak(&kk_bo_census_peak_B, &peak, live))
      ;
   kk_bo_census_tick();
}

static void
kk_bo_census_credit(uint64_t size_B)
{
   atomic_fetch_add(&kk_bo_census_free, 1ull);
   atomic_fetch_sub(&kk_bo_census_live_B, size_B);
   /* Tick on the FREE path too. Reporting only from the alloc path means the last line ever
    * printed is the state at the last allocation — so after a workload closes, the census
    * goes silent holding a stale number, which is precisely the moment a leak hunt cares
    * about ("did the close give it back?"). Measured that way once and nearly read an
    * unchanged line as "nothing was released". */
   kk_bo_census_tick();
}

VkResult
kk_alloc_bo(struct kk_device *dev, struct vk_object_base *log_obj,
            uint64_t size_B, uint64_t align_B, struct kk_bo **bo_out)
{
   VkResult result = VK_SUCCESS;

   // TODO_KOSMICKRISP: Probably requires handling the buffer maximum 256MB
   uint64_t minimum_alignment = 0u;
   mtl_heap_buffer_size_and_align_with_length(dev->mtl_handle, &size_B,
                                              &minimum_alignment);
   minimum_alignment = MAX2(minimum_alignment, align_B);

#if DETECT_ARCH_X86_64
   /* KK_WORKAROUND_8 */
   if (!(dev->disabled_workarounds & BITFIELD64_BIT(8))) {
      minimum_alignment = MAX2(minimum_alignment, 16384);
   }
#endif

   size_B = align64(size_B, minimum_alignment);
   mtl_heap *handle =
      mtl_new_heap(dev->mtl_handle, size_B, KK_MTL_RESOURCE_OPTIONS);
   if (handle == NULL) {
      result = vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY, "%m");
      goto fail_heap;
   }

   mtl_buffer *map = mtl_new_buffer_with_length(handle, size_B, 0u);
   if (map == NULL) {
      result = vk_errorf(log_obj, VK_ERROR_OUT_OF_DEVICE_MEMORY, "%m");
      goto fail_map;
   }

   struct kk_bo *bo = CALLOC_STRUCT(kk_bo);

   if (bo == NULL) {
      result = vk_errorf(log_obj, VK_ERROR_OUT_OF_HOST_MEMORY, "%m");
      goto fail_alloc;
   }

   bo->mtl_handle = handle;
   bo->size_B = size_B;
   bo->map = map;
   bo->gpu = mtl_buffer_get_gpu_address(map);
   bo->cpu = mtl_get_contents(map);

   kk_device_add_heap_to_residency_set(dev, handle);

   kk_bo_census_charge(size_B);

   *bo_out = bo;
   return result;

fail_alloc:
   mtl_release(map);
fail_map:
   mtl_release(handle);
fail_heap:
   return result;
}

void
kk_destroy_bo(struct kk_device *dev, struct kk_bo *bo)
{
   /* An imported MTLTexture has neither heap nor buffer — the texture is the
    * whole allocation. Release it and stop; the paths below would deref NULL. */
   if (bo->texture) {
      /* limina: LIMINA_KK_TEX_TRACE=N — the retain count around the residency removal, on the
       * last lines where the texture is still legally ours to touch. KK holds exactly one ref
       * (the import) plus the residency set's; anything left after both come off is a holder
       * we did not put there. Budgeted: a storm destroys hundreds of these a second. The
       * running census after it is the verdict — a create/dealloc gap means the texture (and
       * with it the IOSurface it is backed by) never died. */
      static atomic_int traced;
      static atomic_int budget = -1;
      int b = atomic_load(&budget);
      if (b < 0) {
         const char *e = getenv("LIMINA_KK_TEX_TRACE");
         b = (e && *e) ? (int)strtol(e, NULL, 10) : 0;
         atomic_store(&budget, b);
      }
      bool trace = b > 0 && atomic_fetch_add(&traced, 1) < b;
      long before = trace ? mtl_limina_retain_count(bo->texture) : 0;

      kk_device_remove_texture_from_residency_set(dev, bo->texture);

      if (trace) {
         char census[128];
         mtl_limina_texture_census(census, sizeof(census));
         fprintf(stderr,
                 "[LIMINA-KK-TEXFREE] tex=%p retain before=%ld after-residency-remove=%ld "
                 "(ours: 1) | %s\n",
                 bo->texture, before, mtl_limina_retain_count(bo->texture), census);
      }

      /* And a heartbeat regardless of the trace budget: the storm destroys hundreds of
       * textures a second, so the gap must stay visible long after the first N lines. */
      static atomic_uint destroyed;
      unsigned n = atomic_fetch_add(&destroyed, 1) + 1;
      if ((n % 256u) == 0u) {
         char census[128];
         mtl_limina_texture_census(census, sizeof(census));
         fprintf(stderr, "[LIMINA-KK-TEXFREE] %s\n", census);
      }

      mtl_release(bo->texture);
      FREE(bo);
      return;
   }

   /* limina: credit the census only for heap-backed BOs, which is exactly the set
    * kk_alloc_bo charged. The early-returning texture path above and the imported
    * host-pointer path below (mtl_handle == NULL) are built elsewhere and were never
    * charged — crediting them would drive the live count negative and make the one
    * number we are trying to trust a lie. */
   if (bo->mtl_handle)
      kk_bo_census_credit(bo->size_B);

   /* We may only have a mapped buffer, for example if the memory was imported
    * from a host pointer */
   if (bo->mtl_handle)
      kk_device_remove_heap_from_residency_set(dev, bo->mtl_handle);
   else
      kk_device_remove_buffer_from_residency_set(dev, bo->map);

   mtl_release(bo->map);

   if (bo->mtl_handle)
      mtl_release(bo->mtl_handle);

   FREE(bo);
}

VkResult
kk_bo_map_placed(struct kk_device *dev, struct kk_bo *bo, void **addr)
{
#if DETECT_OS_APPLE
   mach_vm_address_t src_addr = (mach_vm_address_t)bo->cpu;
   mach_vm_address_t dst_addr = (mach_vm_address_t)*addr;

   int flags = VM_FLAGS_ANYWHERE;
   if (dst_addr)
      flags = VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE;

   vm_prot_t cur_prot;
   vm_prot_t max_prot;

   /* Metal provides us with its own CPU mapping of the memory. mach_vm_remap
    * allows us to create a second mapping pointing to that same memory, placed
    * wherever the caller requests. Note that this is used even without a fixed
    * address, since we need a unique mapping to support reserved unmap. */
   kern_return_t ret = mach_vm_remap(mach_task_self(), &dst_addr, bo->size_B, 0,
                                     flags, mach_task_self(), src_addr, false,
                                     &cur_prot, &max_prot, VM_INHERIT_DEFAULT);
   if (ret != KERN_SUCCESS)
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "mach_vm_remap failed: %d", ret);

   *addr = (void *)dst_addr;
#endif /* DETECT_OS_APPLE */

   return VK_SUCCESS;
}

VkResult
kk_bo_unmap(struct kk_device *dev, struct kk_bo *bo, void *addr, bool reserved)
{
#if DETECT_OS_APPLE
   mach_vm_address_t unmap_addr = (mach_vm_address_t)addr;
   if (reserved) {
      /* Replace with a reserved mapping using mach_vm_map */
      kern_return_t ret = mach_vm_map(
         mach_task_self(), &unmap_addr, bo->size_B, 0,
         VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, MEMORY_OBJECT_NULL, 0, false,
         VM_PROT_NONE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_DEFAULT);
      if (ret != KERN_SUCCESS)
         return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                          "mach_vm_map failed: %d", ret);
   } else {
      /* According to spec, we can only return a fail code here if the reserve
       * flag is set, so ignore the result */
      mach_vm_deallocate(mach_task_self(), unmap_addr, bo->size_B);
   }
#endif /* DETECT_OS_APPLE */

   return VK_SUCCESS;
}
