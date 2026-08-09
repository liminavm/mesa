/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_DEVICE_H
#define KK_DEVICE_H 1

#include "kk_private.h"

#include "kk_query_table.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "kosmickrisp/clc/kk_precompiled_shader.h"
#include "libkk_shaders.h"

#include "util/simple_mtx.h"
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
 * KK used to own three allocators per VkCommandBuffer, created in kk_create_cmd_buffer and
 * released only in kk_destroy_cmd_buffer, so the allocator population tracked command-buffer
 * POPULATION (~378-450 live) rather than encoding concurrency. Apple's model sizes them by
 * concurrency instead ("a command allocator for each frame's work"; their sample ships three).
 *
 * That mattered because of two measured facts (spikes/vrend-region-leak/):
 *   - reset() marks heaps for reuse and NEVER shrinks allocatedSize, so under a no-destroy
 *     policy an allocator's size is monotonically non-decreasing for its whole lifetime;
 *   - render encoding grows an allocator steadily for as long as it goes without a reset
 *     (compute plateaus; render does not), and KK reset only once per vkBeginCommandBuffer,
 *     i.e. once per epoch however many render passes that epoch contained.
 * Multiply those together across hundreds of independent allocators and you get the ~11 GiB
 * IOAccelerator ratchet, with a single allocator observed at 753 MiB.
 *
 * The pool bounds the product directly: allocators are shared, so the count follows concurrency,
 * and one is retired from service once it passes a byte budget, drained, and reset before reuse.
 * Nothing is destroyed until device teardown.
 *
 * Segregated by encoder kind because reuse does NOT cross types: a render pass encoded on a
 * compute-warmed allocator adds its own heaps rather than reusing them (measured +17.6 MiB), so
 * a mixed allocator would hold the union of both working sets.
 */
enum kk_alloc_class {
   KK_ALLOC_CLASS_RENDER = 0,
   KK_ALLOC_CLASS_COMPUTE,
   KK_ALLOC_CLASS_COUNT,
};

struct kk_pooled_alloc {
   mtl_command_allocator *handle;
   /* Command buffers begun on this allocator that have not yet completed on the GPU. Charged at
    * mtl_begin_command_buffer, NOT at commit: an allocator is returned to the pool at
    * kk_stop_encoder while its command buffers still sit uncommitted in submit_cmd_bufs, and a
    * reset in that window would discard commands the GPU has not even been handed. */
   uint32_t pending;
   bool in_use;   /* borrowed by an open encoder — Metal allows only one at a time */
   bool draining; /* over budget: no new work until pending hits 0 and it is reset */
   enum kk_alloc_class klass;
};

struct kk_alloc_pool {
   simple_mtx_t mtx;
   struct util_dynarray allocs[KK_ALLOC_CLASS_COUNT]; /* struct kk_pooled_alloc * */
   uint64_t budget_bytes;
   /* Stats, so pool growth is never silent. */
   uint32_t live[KK_ALLOC_CLASS_COUNT];
   uint32_t peak_live[KK_ALLOC_CLASS_COUNT];
   uint32_t watermark_warned;
};

/* util_dynarray's macros take a single type token, so give the pointer a name. */
typedef struct kk_pooled_alloc *kk_pooled_alloc_ptr;

struct kk_pooled_alloc *kk_alloc_pool_acquire(struct kk_device *dev,
                                              enum kk_alloc_class klass);
void kk_alloc_pool_release(struct kk_device *dev, struct kk_pooled_alloc *pa);
void kk_alloc_pool_charge(struct kk_device *dev, struct kk_pooled_alloc *pa);
void kk_alloc_pool_discharge(struct kk_device *dev, struct kk_pooled_alloc *pa);
void kk_alloc_pool_init(struct kk_device *dev);
void kk_alloc_pool_finish(struct kk_device *dev);

struct kk_device {
   struct vk_device vk;

   mtl_device *mtl_handle;
   struct kk_alloc_pool alloc_pool;
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

   uint64_t disabled_workarounds;
   bool gpu_capture_enabled;

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
