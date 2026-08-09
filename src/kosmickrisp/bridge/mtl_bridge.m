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
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

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
 * SHAPE. Creation is noted in each of the 36 `mtl_new_*` functions. Death is counted by a
 * DEALLOC SENTINEL — an associated object, which the runtime releases when its host is
 * deallocated — not by instrumenting a release call site.
 *
 * That distinction is the whole point, and the first version of this got it wrong. Counting
 * releases (even "the release where the retain count was 1") only ever sees deaths WE cause.
 * Metal objects routinely die in someone else's hands: an allocation in a residency set is
 * retained by the set and released at `commit`, long after our `mtl_release` returned. A
 * release-site counter reports those objects as immortal forever, which reads exactly like a
 * leak. The retain-count version of this counter claimed MTLHeap `made == live` with zero
 * deallocs — and the very next thing it did was send me after the residency set for a leak
 * that number could not actually establish. The sentinel counts the death itself, whoever
 * held the final reference, so `made - freed` is now a real live count.
 *
 * Objects we never pass through an `mtl_new_*` are invisible here; a class whose live count is
 * flat is genuinely exonerated. The self-test proves the mechanism fires before any zero is
 * believed — an instrument that never fires reads identically to "nothing leaked".
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


/* Counting the release CALL proves nothing about the object dying — the last ref is often
 * dropped by someone else entirely (the residency set drops its ref at commit, not at our
 * mtl_release), and such a death is invisible to a release-site counter. So count the death
 * itself: an associated object is deallocated with its host, so a sentinel turns "did this
 * object actually die" into a number, whoever held the final reference. Same mechanism as the
 * import-texture sentinel in mtl_device.m. */
@interface LiminaMtlClassSentinel : NSObject
@property(nonatomic) int slot;
@property(nonatomic) bool selftest;
@end

static _Atomic unsigned long long g_limina_mtl_selftest_dealloc;

@implementation LiminaMtlClassSentinel
- (void)dealloc
{
   if (self.selftest) {
      atomic_fetch_add(&g_limina_mtl_selftest_dealloc, 1ull);
   } else {
      pthread_mutex_lock(&limina_mtl_census_lock);
      limina_mtl_census[self.slot].freed++;
      pthread_mutex_unlock(&limina_mtl_census_lock);
   }
   [super dealloc];
}
@end

static const char g_limina_mtl_sentinel_key; /* address only */

static void
limina_mtl_sentinel_attach(id obj, int slot, bool selftest)
{
   LiminaMtlClassSentinel *s = [[LiminaMtlClassSentinel alloc] init];
   s.slot = slot;
   s.selftest = selftest;
   objc_setAssociatedObject(obj, &g_limina_mtl_sentinel_key, s,
                            OBJC_ASSOCIATION_RETAIN_NONATOMIC);
   [s release]; /* the association holds the only ref; it dies with obj */
}

/* An instrument that never fires reads exactly like "nothing leaked", so prove the mechanism
 * on an object whose death we control before trusting a zero. */
static void
limina_mtl_sentinel_selftest(void)
{
   unsigned long long before = atomic_load(&g_limina_mtl_selftest_dealloc);
   @autoreleasepool {
      NSObject *victim = [[NSObject alloc] init];
      limina_mtl_sentinel_attach(victim, -1, true);
      [victim release];
   }
   fprintf(stderr, "[LIMINA-KK-MTLCLASS] dealloc sentinel self-test: %s\n",
           atomic_load(&g_limina_mtl_selftest_dealloc) > before ? "OK"
                                                               : "BROKEN (counts are lies)");
}

/* Cached: the census costs an associated object per Metal object, so it stays off unless asked. */
static int
limina_mtl_census_enabled(void)
{
   static _Atomic int cached = -1;
   int v = atomic_load(&cached);
   if (v < 0) {
      const char *e = getenv("LIMINA_KK_BOCENSUS");
      v = (e && *e && strtol(e, NULL, 10) > 0) ? 1 : 0;
      atomic_store(&cached, v);
      if (v) {
         static pthread_once_t once = PTHREAD_ONCE_INIT;
         pthread_once(&once, limina_mtl_sentinel_selftest);
      }
   }
   return v;
}

void *
limina_mtl_note_new(void *handle)
{
   if (handle == NULL || !limina_mtl_census_enabled())
      return handle;
   Class cls = object_getClass((id)handle);
   pthread_mutex_lock(&limina_mtl_census_lock);
   int i = limina_mtl_census_slot_locked(cls);
   if (i >= 0)
      limina_mtl_census[i].made++;
   pthread_mutex_unlock(&limina_mtl_census_lock);
   if (i >= 0)
      limina_mtl_sentinel_attach((id)handle, i, false);
   return handle;
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
      [obj release];
   }
}
