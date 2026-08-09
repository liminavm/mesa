/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_bridge.h"

// kk_image_layout.h should also includes "vulkan/vulkan.h", but just to be safe
#include "vulkan/vulkan.h"
#include "kk_image_layout.h"

#include "util/macros.h"

#include <objc/runtime.h>
#include <pthread.h>
#include <stdio.h>

#include <Metal/MTLCommandBuffer.h>
#include <Metal/MTLCommandQueue.h>
#include <Metal/MTLDevice.h>
#include <Metal/MTLHeap.h>
#include <Metal/MTLEvent.h>

#include <QuartzCore/CAMetalLayer.h>

static_assert(sizeof(MTLResourceID) == sizeof(uint64_t), "Must match, otherwise descriptors are broken");

mtl_texture *
mtl_drawable_get_texture(void *drawable_ptr)
{
   @autoreleasepool {
      id<CAMetalDrawable> drawable = (id<CAMetalDrawable>)drawable_ptr;
      return drawable.texture;
   }
}

/*
 * limina: per-class live-object census — LIMINA_KK_BOCENSUS=<secs> prints it.
 *
 * WHY. The vrend region leak (spikes/vrend-region-leak/) ratchets ~15 500 `IOAccelerator
 * (graphics)` VM regions per workload cycle and never returns them. Every accounting layer
 * that existed was blind or exonerated: virglrenderer's GPU-memory ledger sat flat at 11.7 MiB
 * while the worker grew to 10 GB, KK BO heaps plateaued, and live plane textures FELL. Chasing
 * one object class at a time was going to take a rebuild-and-reboot per guess.
 *
 * So count every Metal object we mint, keyed by its Objective-C class, and report live counts
 * sorted by size of the leak. This names the class outright instead of eliminating them singly.
 *
 * SHAPE. Creation is spread over 36 `mtl_new_*` functions, so each notes its result here.
 * Destruction has exactly ONE chokepoint, `mtl_release` — but that is called for every release,
 * not only the last one, so decrementing there unconditionally would undercount live objects by
 * however many `mtl_retain`s are outstanding. We therefore decrement only on the release that
 * actually deallocates, detected by a retain count of 1 immediately before it. (This codebase
 * already reasons about retain counts in exactly this way — see mtl_limina_retain_count.)
 *
 * Objects created behind Metal's back — anything we never pass through an mtl_new_* — are
 * invisible here, and a class whose live count is honestly flat is genuinely exonerated.
 *
 * ⚠ READ `made == live` CAREFULLY — it is not automatically a leak. It means only that the
 * class never reached its dealloc THROUGH mtl_release. Some objects are released by other
 * means and will therefore always look immortal here:
 *
 *   MTLTextureDescriptorInternal   `[mtl_new_texture_descriptor(...) autorelease]`
 *                                  (mtl_heap.m, mtl_buffer.m) and a bare `[desc release]`
 *                                  in mtl_device.m — never mtl_release, so its 2 000+ "live"
 *                                  is an artefact of this counter, not a finding.
 *
 * The signal is a class that DOES go through mtl_release and still never deallocates. That is
 * exactly how the leak was pinned: MTLHeap is released at kk_bo.c (`mtl_release(bo->mtl_handle)`
 * in kk_destroy_bo), yet AGXG13XFamilyHeap reported made == live == 1 191 with zero deallocs
 * while kk_destroy_bo had run 239 times. Something outside KK still holds every heap; the
 * residency set (`[set addAllocation:]` retains) is the first place to look.
 */
#define LIMINA_MTL_CENSUS_SLOTS 128
static pthread_mutex_t limina_mtl_census_lock = PTHREAD_MUTEX_INITIALIZER;
static struct {
   Class cls;
   unsigned long long made;
   unsigned long long freed;
} limina_mtl_census[LIMINA_MTL_CENSUS_SLOTS];

static int
limina_mtl_census_slot_locked(Class cls)
{
   for (int i = 0; i < LIMINA_MTL_CENSUS_SLOTS; i++) {
      if (limina_mtl_census[i].cls == cls)
         return i;
      if (limina_mtl_census[i].cls == NULL) {
         limina_mtl_census[i].cls = cls;
         return i;
      }
   }
   return -1; /* table full: later classes go uncounted rather than mis-attributed */
}

void *
limina_mtl_note_new(void *handle)
{
   if (handle == NULL)
      return NULL;
   Class cls = object_getClass((id)handle);
   pthread_mutex_lock(&limina_mtl_census_lock);
   int i = limina_mtl_census_slot_locked(cls);
   if (i >= 0)
      limina_mtl_census[i].made++;
   pthread_mutex_unlock(&limina_mtl_census_lock);
   return handle;
}

static void
limina_mtl_note_dealloc(void *handle)
{
   Class cls = object_getClass((id)handle);
   pthread_mutex_lock(&limina_mtl_census_lock);
   int i = limina_mtl_census_slot_locked(cls);
   if (i >= 0)
      limina_mtl_census[i].freed++;
   pthread_mutex_unlock(&limina_mtl_census_lock);
}

void
limina_mtl_class_census_dump(void)
{
   struct limina_mtl_census_row {
      const char *name;
      long long live;
      unsigned long long made;
   };
   struct limina_mtl_census_row rows[LIMINA_MTL_CENSUS_SLOTS];
   int n = 0;

   pthread_mutex_lock(&limina_mtl_census_lock);
   for (int i = 0; i < LIMINA_MTL_CENSUS_SLOTS && limina_mtl_census[i].cls; i++) {
      rows[n].name = class_getName(limina_mtl_census[i].cls);
      rows[n].live =
         (long long)(limina_mtl_census[i].made - limina_mtl_census[i].freed);
      rows[n].made = limina_mtl_census[i].made;
      n++;
   }
   pthread_mutex_unlock(&limina_mtl_census_lock);

   /* Sorted by live descending: the leaking class is meant to be the first line. */
   for (int a = 0; a < n; a++)
      for (int b = a + 1; b < n; b++)
         if (rows[b].live > rows[a].live) {
            struct limina_mtl_census_row t = rows[a];
            rows[a] = rows[b];
            rows[b] = t;
         }

   fprintf(stderr, "[LIMINA-KK-MTLCLASS] live objects by class (top %d of %d):\n",
           n < 12 ? n : 12, n);
   for (int i = 0; i < n && i < 12; i++)
      fprintf(stderr, "[LIMINA-KK-MTLCLASS]   %-44s live=%-8lld made=%llu\n",
              rows[i].name, rows[i].live, rows[i].made);
}

void *
mtl_retain(void *handle)
{
   @autoreleasepool {
      NSObject *obj = (NSObject *)handle;
      return [obj retain];
   }
}

void
mtl_release(void *handle)
{
   @autoreleasepool {
      NSObject *obj = (NSObject *)handle;
      /* limina: only the release that actually deallocates reduces the live count. */
      if (handle != NULL && [obj retainCount] == 1)
         limina_mtl_note_dealloc(handle);
      [obj release];
   }
}
