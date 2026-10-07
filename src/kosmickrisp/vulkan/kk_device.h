/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_DEVICE_H
#define KK_DEVICE_H 1

#include "kk_private.h"

#include "util/u_atomic.h"

#include "kk_query_table.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "kosmickrisp/clc/kk_precompiled_shader.h"
#include "libkk_shaders.h"

#include "util/u_dynarray.h"

#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct kk_bo;
struct kk_physical_device;
struct vk_pipeline_cache;

struct kk_residency_set {
   simple_mtx_t mutex;
   mtl_residency_set *handle;
};

struct mtl_sampler_packed {
   enum mtl_sampler_address_mode mode_u;
   enum mtl_sampler_address_mode mode_v;
   enum mtl_sampler_address_mode mode_w;
   enum mtl_sampler_border_color border_color;

   enum mtl_sampler_min_mag_filter min_filter;
   enum mtl_sampler_min_mag_filter mag_filter;
   enum mtl_sampler_mip_filter mip_filter;
   enum mtl_sampler_reduction_mode reduction_mode;

   enum mtl_compare_function compare_func;
   float min_lod;
   float max_lod;
   uint32_t max_anisotropy;
   bool normalized_coordinates;
};

struct kk_rc_sampler {
   struct mtl_sampler_packed key;

   mtl_sampler *handle;

   /* Reference count for this hardware sampler, protected by the heap mutex */
   uint16_t refcount;

   /* Index of this hardware sampler in the hardware sampler heap */
   uint16_t index;
};

struct kk_sampler_heap {
   simple_mtx_t lock;

   struct kk_query_table table;

   /* Map of mtl_sampler_packed to kk_rc_sampler */
   struct hash_table *ht;
};

struct kk_precompiled_cache {
   struct kk_precompiled_shader shaders[LIBKK_NUM_PROGRAMS];
};

/* limina: shared MTL4 command-allocator pool.
 *
 * Upstream gives each VkCommandPool its own free list of allocators and never resets them, and
 * two measured facts make that ratchet (spikes/vrend-region-leak/, spikes/kk-alloc-pool/):
 *   - reset() marks heaps for reuse and NEVER shrinks allocatedSize, so an allocator's size is
 *     non-decreasing for its whole lifetime; only releasing it returns its heaps (100%, measured);
 *   - each workload launch drives a different subset of allocators to a new high-water, so
 *     never-reset allocators climb with every launch and return nothing when it closes
 *     (663 -> 1492 MiB over eight aquarium launches on the vrend tier, 117 allocators throughout).
 * On the vrend tier the KK VkDevice belongs to host zink's screen and lives until the worker
 * exits, so nothing else ever gives that memory back.
 *
 * The pool bounds and returns it: allocators are borrowed device-wide for the span of one
 * recording, so the count follows concurrency; one that crosses a byte budget leaves service and
 * is released once every command buffer begun on it has completed; and an idle one past a decay
 * window is released above a floor. The count has a ceiling: an acquire that finds the pool full
 * waits briefly for a borrow to end or a drained allocator to be released, then fails, because a
 * guest's in-flight depth is not bounded by anything the host controls. Nothing is ever reset:
 * reset returns no memory, and reusing a reset allocator faults inside
 * -[IOGPUMetal4CommandBuffer fillCommandBufferArgs:] at commit under upstream's
 * reused-MTL4CommandBuffer model (spikes/kk-alloc-pool/RESULTS.md).
 */
struct kk_pooled_alloc {
   mtl_command_allocator *handle;
   /* Command buffers begun on this allocator that the GPU has not completed. Charged at
    * mtl_begin_command_buffer, NOT at commit: the borrow is returned at vkEndCommandBuffer while
    * the command buffer still sits uncommitted, and releasing the allocator in that window would
    * free heaps the GPU has not been handed yet. */
   uint32_t pending;
   bool in_use;   /* borrowed by a recording command buffer; Metal allows one at a time */
   bool draining; /* over budget: never borrowed again, released once pending hits 0 */
   uint32_t uses; /* recordings so far, for the stats */
   /* os_time_get_nano() when the last borrow ended: the decay clock. 0 = borrowed. */
   uint64_t idle_since;
};

struct kk_alloc_pool {
   mtx_t mtx;
   /* Signalled by release and discharge, the two events that can make room at the ceiling, and
    * only while someone waits: the common path takes no extra syscall. */
   cnd_t room;
   uint32_t waiters;
   struct util_dynarray allocs; /* struct kk_pooled_alloc * */
   uint64_t budget_bytes;
   uint64_t decay_ns;    /* idle this long before an under-budget allocator is surplus */
   uint32_t floor;       /* never decay below this many (drained over-budget ones always go) */
   /* Never mint past this many live allocators (0 = no ceiling). At the ceiling an acquire waits
    * up to wait_ns for a release or a drain, then fails. */
   uint32_t ceiling;
   uint64_t wait_ns;
   bool destroy_enabled; /* LIMINA_KK_ALLOC_DESTROY=0 kill switch */
   /* Stats, so neither growth nor reclaim is silent. */
   uint32_t live;
   uint32_t peak_live;
   uint32_t destroyed;
   uint32_t watermark_warned; /* growth past this warns, and doubles it */
   uint64_t ceiling_waits;    /* acquires that found the pool at its ceiling */
   uint64_t ceiling_fails;    /* ... and gave up */
   uint64_t acquires;
   uint64_t stats_every; /* LIMINA_KK_ALLOC_STATS=<n>: a stats line every n acquires; 0 = off */
   uint64_t retired;     /* crossed the budget */
   uint64_t retired_uses; /* sum of their recordings, for the mean lifetime */
};

/* util_dynarray's macros take a single type token, so give the pointer a name. */
typedef struct kk_pooled_alloc *kk_pooled_alloc_ptr;

void kk_alloc_pool_init(struct kk_device *dev);
void kk_alloc_pool_finish(struct kk_device *dev);
struct kk_pooled_alloc *kk_alloc_pool_acquire(struct kk_device *dev);
void kk_alloc_pool_release(struct kk_device *dev, struct kk_pooled_alloc *pa);
void kk_alloc_pool_charge(struct kk_device *dev, struct kk_pooled_alloc *pa);
void kk_alloc_pool_discharge(struct kk_device *dev, struct kk_pooled_alloc *pa);

struct kk_device {
   struct vk_device vk;

   /* limina: see struct kk_alloc_pool. */
   struct kk_alloc_pool alloc_pool;

   mtl_device *mtl_handle;
   mtl_compiler *mtl_compiler_handle;

   /* Dispatch table exposed to the user. Required since we need to record all
    * commands due to Metal limitations */
   struct vk_device_dispatch_table exposed_dispatch_table;

   struct kk_sampler_heap samplers;
   struct kk_query_table occlusion_queries;

   /* Track all heaps the user allocated so we can set them all as resident when
    * recording as required by Metal. */
   struct kk_residency_set residency_set;

   struct kk_precompiled_cache precompiled_cache;

   bool has_queue;
   struct kk_queue queue;

   struct vk_meta_device meta;

   /* Geomtry heap */
   struct kk_bo *heap;
   util_once_flag heap_init_once;

   /* Triangle-list indices for a triangle fan of up to KK_FAN_MAX_VERTICES vertices,
    * shared by every direct non-indexed fan draw. Created on first use. */
   struct kk_bo *fan_indices;
   util_once_flag fan_indices_once;

   /* Transform feedback counter-buffer shadow: command replay is sequential
    * on the queue thread and zink only consumes counter values through
    * vkCmdBeginTransformFeedbackEXT resume / vkCmdDrawIndirectByteCountEXT,
    * both of which we serve from this CPU-side map (the buffer itself is
    * not written -- TODO for full conformance). Small ring, newest wins.
    */
   struct {
      struct {
         mtl_buffer *buffer;
         uint64_t offset;
         uint64_t value;
      } entries[32];
      uint32_t next;
   } xfb_counters;
};

VK_DEFINE_HANDLE_CASTS(kk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct kk_physical_device *
kk_device_physical(const struct kk_device *dev)
{
   return (struct kk_physical_device *)dev->vk.physical;
}

VkResult kk_device_init_meta(struct kk_device *dev);
void kk_device_finish_meta(struct kk_device *dev);
VkResult kk_device_init_lib(struct kk_device *dev);
void kk_device_finish_lib(struct kk_device *dev);
/* limina: how many allocations the queue-attached residency set is carrying. A bindless read
 * dereferences a texture's MTLResourceID, and a resource that is not resident faults exactly as a
 * stale pointer would -- so the size of this set is a suspect in its own right once every
 * lifetime explanation is spent. Counted by kind because a view, a heap and a placed texture are
 * three different costs. */
extern uint32_t kk_limina_resident_heaps, kk_limina_resident_buffers,
   kk_limina_resident_textures;

/* limina: how many device sampler-table slots have been zeroed and returned to the free list. */
extern uint32_t kk_limina_sampler_retires;

void kk_device_add_heap_to_residency_set(struct kk_device *dev, mtl_heap *heap);
void kk_device_remove_heap_from_residency_set(struct kk_device *dev,
                                              mtl_heap *heap);
void kk_device_add_buffer_to_residency_set(struct kk_device *dev,
                                           mtl_buffer *buffer);
void kk_device_remove_buffer_from_residency_set(struct kk_device *dev,
                                                mtl_buffer *buffer);
void kk_device_add_texture_to_residency_set(struct kk_device *dev,
                                            mtl_texture *texture);
void kk_device_remove_texture_from_residency_set(struct kk_device *dev,
                                                 mtl_texture *texture);
void kk_device_make_resources_resident(struct kk_device *dev);

/* Required to create a sampler */
mtl_sampler *kk_sampler_create(struct kk_device *dev,
                               const struct mtl_sampler_packed *packed);
VkResult kk_sampler_heap_add(struct kk_device *dev,
                             struct mtl_sampler_packed desc,
                             struct kk_rc_sampler **out);
void kk_sampler_heap_remove(struct kk_device *dev, struct kk_rc_sampler *rc);

#endif // KK_DEVICE_H
