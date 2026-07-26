/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_ENCODER_H
#define KK_ENCODER_H 1

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/u_dynarray.h"

#include "vulkan/vulkan.h"

struct kk_queue;
struct kk_cmd_buffer;

enum kk_encoder_type {
   KK_ENC_NONE = 0,
   KK_ENC_RENDER = BITFIELD_BIT(0),
   KK_ENC_COMPUTE = BITFIELD_BIT(1),
   KK_ENC_BLIT = BITFIELD_BIT(2),
   KK_ENC_ALL = (KK_ENC_RENDER | KK_ENC_COMPUTE | KK_ENC_BLIT),
   KK_ENC_COUNT = 3u,
};

struct kk_encoder_internal {
   mtl_command_buffer *cmd_buffer;
   mtl_command_encoder *encoder;

   /* Used to know if we need to make heaps resident again */
   uint32_t user_heap_hash;

   /* Need to track last used to we can converge at submission */
   enum kk_encoder_type last_used;

   /* Used to synchronize between passes inside the same command buffer */
   struct util_dynarray fences;
   /* Tracks if we need to wait on the last fence present in fences at the start
    * of the pass */
   bool wait_fence;
};

struct kk_copy_query_pool_results_info {
   uint64_t availability;
   uint64_t results;
   uint64_t indices;
   uint64_t dst_addr;
   uint64_t dst_stride;
   uint32_t first_query;
   VkQueryResultFlagBits flags;
   uint16_t reports_per_query;
   uint32_t query_count;
};

/* Where one vkCmdWriteTimestamp puts its result.
 *
 * Both mappings of the same 8 bytes are needed: the GPU one so the report can
 * be marked unavailable in command order at the point the clock is sampled, the
 * CPU one so the resolved value can be written when the command buffer
 * completes. See kk_encoder_write_timestamp. */
struct kk_timestamp_target {
   mtl_buffer *dst;       /* pool->bo->map */
   uint64_t offset;       /* byte offset of the query report within dst */
   uint64_t *report;      /* pool->bo->cpu + offset */
   uint64_t *pending_seq; /* &pool->ts_pending_seq, published on write */
};

struct kk_timestamp_sample {
   mtl_counter_sample_buffer *sb; /* owned */
   uint64_t *report;              /* CPU map of the 8-byte query report */
};

/* One command buffer's worth of timestamp samples, resolved on the CPU once it
 * completes. Self-contained: it owns everything it touches, so its completion
 * handler can run in any order against the encoder's own. */
struct kk_timestamp_batch {
   struct kk_device *dev;
   uint64_t seq;
   /* Array of struct kk_timestamp_sample. */
   struct util_dynarray samples;
};

/* Resolve this batch's samples and write them into their query reports. Safe to
 * call more than once, and safe to call before the completion handler: a sample
 * that has not been taken yet resolves to 0 and is skipped. */
void kk_timestamp_batch_publish(struct kk_timestamp_batch *batch);

struct kk_encoder {
   mtl_device *dev;
   struct kk_device *device;
   struct kk_encoder_internal main;
   /* Compute only for pre gfx required work */
   struct kk_encoder_internal pre_gfx;

   /* Queue `main` allocates command buffers from — kept so main can be split
    * mid-recording (see extra_cmd_buffers). */
   mtl_command_queue *main_queue;
   /* Retired `main` command buffers, oldest first, from splits. Committed in
    * order ahead of main at submission; released with it. Empty unless the GPU
    * needs a split counter resolve. Array of mtl_command_buffer *. */
   struct util_dynarray extra_cmd_buffers;

   /* Used to synchronize between main and pre_gfx encoders */
   mtl_event *event;
   uint64_t event_value;
   /* Track what values pre_gfx must wait/signal before starting the encoding */
   uint64_t wait_value_pre_gfx;
   uint64_t signal_value_pre_gfx;
   uint64_t last_signaled_value_pre_gfx;

   /* uint64_t pairs with first being the address, second being the value to
    * write */
   struct util_dynarray imm_writes;
   /* Array of kk_copy_quer_pool_results_info structs */
   struct util_dynarray copy_query_pool_result_infos;

   /* Timestamp samples taken into the CURRENT main command buffer, awaiting the
    * CPU resolve its completion handler will do. NULL when there are none.
    * Handed to that command buffer by kk_encoder_split_main / _submit. */
   struct kk_timestamp_batch *ts_batch;
   /* Timestamp writes issued inside a render pass, flushed at its end (we can
    * only sample at encoder boundaries). Array of kk_timestamp_target. */
   struct util_dynarray deferred_timestamps;
};

/* Allocates encoder and initialises/creates all resources required to start
 * recording commands into the multiple encoders */
VkResult kk_encoder_init(mtl_device *device, struct kk_queue *queue,
                         struct kk_encoder **encoder);

/* Submits all command buffers and releases encoder memory. Requires all command
 * buffers in the encoder to be linked to the last one used so the post
 * execution callback is called once all are done */
void kk_encoder_submit(struct kk_encoder *encoder);

mtl_render_encoder *
kk_encoder_start_render(struct kk_cmd_buffer *cmd,
                        mtl_render_pass_descriptor *descriptor,
                        uint32_t view_mask);

/* Ends encoding on all command buffers */
void kk_encoder_end(struct kk_cmd_buffer *cmd);

/* Creates a fence and signals it inside the encoder, then ends encoding */
void kk_encoder_signal_fence_and_end(struct kk_cmd_buffer *cmd);

mtl_render_encoder *kk_render_encoder(struct kk_cmd_buffer *cmd);

mtl_compute_encoder *kk_compute_encoder(struct kk_cmd_buffer *cmd);

mtl_blit_encoder *kk_blit_encoder(struct kk_cmd_buffer *cmd);

mtl_compute_encoder *kk_encoder_pre_gfx_encoder(struct kk_cmd_buffer *cmd);

void upload_queue_writes(struct kk_cmd_buffer *cmd);

/* Sample the GPU timestamp at the current command-stream point. The report is
 * marked unavailable in GPU command order here and written with the resolved
 * value when this command buffer completes; *target->pending_seq is set to the
 * sequence number to order that write against (see kk_timestamp_sync).
 *
 * Must NOT be called inside a render pass — use kk_encoder_defer_timestamp
 * there and flush at pass end. */
void kk_encoder_write_timestamp(struct kk_cmd_buffer *cmd,
                                const struct kk_timestamp_target *target);

/* Queue a timestamp write to be emitted when the enclosing render pass ends. */
void kk_encoder_defer_timestamp(struct kk_cmd_buffer *cmd,
                                const struct kk_timestamp_target *target);

/* Order everything encoded after this point in `cmd` against the timestamp
 * report writes up to `seq`. Cheap and no-op once they have landed. Must not be
 * called inside a render pass (it may end the current command buffer). */
void kk_encoder_timestamp_barrier(struct kk_cmd_buffer *cmd, uint64_t seq);

/* Emit all deferred (in-render-pass) timestamp writes. Call once the render
 * encoder has been/will be closed (i.e. outside the pass). */
void kk_encoder_flush_deferred_timestamps(struct kk_cmd_buffer *cmd);

#endif /* KK_ENCODER_H */
