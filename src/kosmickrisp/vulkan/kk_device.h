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

#include "util/u_dynarray.h"

#include "vk_device.h"
#include "vk_meta.h"
#include "vk_queue.h"

struct kk_bo;
struct kk_physical_device;
struct kk_timestamp_batch;
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

/* Ordering for timestamp queries, whose report is written by the CPU when the
 * command buffer that sampled the GPU clock completes (see
 * kk_encoder_write_timestamp). That write lands OUTSIDE GPU command order, so
 * anything that observes the report -- an in-stream vkCmdResetQueryPool or
 * vkCmdCopyQueryPoolResults, a host vkResetQueryPool, destroying the pool --
 * has to be ordered against it explicitly.
 *
 * Every command buffer that carries timestamp samples takes a sequence number.
 * Its completion handler writes the reports and then retires that number, and
 * `event` is advanced to the largest CONTIGUOUS retired prefix -- so a waiter
 * on N is released only once 1..N have all been written, whatever order the
 * completion handlers happen to fire in. Consumers wait on `event`: on the GPU
 * with encodeWaitForEvent:, on the CPU with waitUntilSignaledValue:. */
struct kk_timestamp_sync {
   simple_mtx_t mutex;
   mtl_shared_event *event;
   uint64_t next_seq;  /* last sequence number handed out */
   uint64_t signalled; /* == event.signaledValue, the retired prefix */
   /* Sequence numbers retired ahead of `signalled + 1`; drained into the
    * prefix as the gaps fill. Normally empty. Array of uint64_t. */
   struct util_dynarray retired_out_of_order;
   /* Batches submitted but not yet retired, so a CPU reader can publish one
    * early instead of losing a race with its completion handler. Array of
    * struct kk_timestamp_batch *. */
   struct util_dynarray in_flight;
};

struct kk_device {
   struct vk_device vk;

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

   struct kk_timestamp_sync timestamps;
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

/* Take the sequence number for a command buffer that is about to carry
 * timestamp samples. Never returns 0 (0 means "no pending write"). */
uint64_t kk_timestamp_seq_alloc(struct kk_device *dev);

/* Called from the completion handler once `seq`'s reports have been written.
 * Advances the shared event over the newly contiguous prefix. */
void kk_timestamp_seq_retire(struct kk_device *dev, uint64_t seq);

/* The retired prefix, i.e. every timestamp report with a sequence number <=
 * this has already been written. Cheap; used to skip needless barriers. */
uint64_t kk_timestamp_seq_signalled(struct kk_device *dev);

/* Block until `seq` has been retired. For host-side observers of the report
 * (vkResetQueryPool, vkDestroyQueryPool). */
void kk_timestamp_wait_cpu(struct kk_device *dev, uint64_t seq);

/* Write out the reports of every in-flight batch up to `seq` whose sample has
 * already materialised, without waiting for its completion handler.
 *
 * Non-blocking and self-validating: a counter sample buffer is freshly
 * allocated per timestamp and resolves to 0 (or MTLCounterErrorValue) until the
 * GPU actually takes its sample, so a nonzero read is necessarily the real
 * thing and a zero simply means "not yet" -- there is no window in which this
 * can invent a value. Lets vkGetQueryPoolResults answer a poll issued the
 * moment the queue goes idle, which otherwise loses to the completion handler
 * every single time (0 successes in 10 runs; 4 in 5 with this). */
void kk_timestamp_publish_ready(struct kk_device *dev, uint64_t seq);

/* Track/untrack an in-flight batch. Internal to the timestamp path. */
void kk_timestamp_batch_track(struct kk_device *dev,
                              struct kk_timestamp_batch *batch);
void kk_timestamp_batch_untrack(struct kk_device *dev,
                                struct kk_timestamp_batch *batch);

/* The shared event to encodeWaitForEvent: on for GPU-side observers. */
mtl_shared_event *kk_timestamp_event(struct kk_device *dev);

/* Required to create a sampler */
mtl_sampler *kk_sampler_create(struct kk_device *dev,
                               const struct mtl_sampler_packed *packed);
VkResult kk_sampler_heap_add(struct kk_device *dev,
                             struct mtl_sampler_packed desc,
                             struct kk_rc_sampler **out);
void kk_sampler_heap_remove(struct kk_device *dev, struct kk_rc_sampler *rc);

#endif // KK_DEVICE_H
