/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_encoder.h"

#include "kk_bo.h"
#include "kk_cmd_buffer.h"
#include "kk_queue.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "kosmickrisp/libkk/kk_query.h"
#include "libkk_shaders.h"

/* limina LIMINA_KK_TS_TRACE=1: trace the timestamp query path. Off by default;
 * this is a debugging oracle for the M4 Pro zero-timestamp defect, not telemetry. */
static bool
kk_ts_trace(void)
{
   static int on = -1;
   if (on < 0) {
      const char *v = getenv("LIMINA_KK_TS_TRACE");
      on = v && v[0] != '0';
   }
   return on > 0;
}

static void
kk_encoder_start_internal(struct kk_encoder_internal *encoder,
                          mtl_device *device, mtl_command_queue *queue)
{
   encoder->cmd_buffer = mtl_new_command_buffer(queue);
   encoder->last_used = KK_ENC_NONE;
   encoder->fences = UTIL_DYNARRAY_INIT;
}

VkResult
kk_encoder_init(mtl_device *device, struct kk_queue *queue,
                struct kk_encoder **encoder)
{
   assert(encoder && device && queue);
   struct kk_encoder *enc = (struct kk_encoder *)malloc(sizeof(*enc));
   if (!enc)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   memset(enc, 0u, sizeof(*enc));
   enc->dev = device;
   enc->device = kk_queue_device(queue);
   kk_encoder_start_internal(&enc->main, device, queue->main.mtl_handle);
   kk_encoder_start_internal(&enc->pre_gfx, device, queue->pre_gfx.mtl_handle);
   enc->main_queue = queue->main.mtl_handle;
   enc->extra_cmd_buffers = UTIL_DYNARRAY_INIT;
   enc->event = mtl_new_event(device);
   enc->imm_writes = UTIL_DYNARRAY_INIT;
   enc->copy_query_pool_result_infos = UTIL_DYNARRAY_INIT;
   enc->deferred_timestamps = UTIL_DYNARRAY_INIT;

   *encoder = enc;
   return VK_SUCCESS;
}

mtl_render_encoder *
kk_encoder_start_render(struct kk_cmd_buffer *cmd,
                        mtl_render_pass_descriptor *descriptor,
                        uint32_t view_mask)
{
   struct kk_encoder *encoder = cmd->encoder;
   /* We must not already be in a render encoder */
   assert(encoder->main.last_used != KK_ENC_RENDER ||
          encoder->main.encoder == NULL);

   upload_queue_writes(cmd);
   if (encoder->main.last_used != KK_ENC_RENDER) {
      kk_encoder_signal_fence_and_end(cmd);

      /* Before we start any render operation we need to ensure we have the
       * requried signals to insert pre_gfx execution before the render encoder
       * in case we need to insert commands to massage input data for things
       * like triangle fans. For this, we signal the value pre_gfx will wait on,
       * and we wait on the value pre_gfx will signal once completed.
       */
      mtl_encode_signal_event(encoder->main.cmd_buffer, encoder->event,
                              ++encoder->event_value);
      encoder->wait_value_pre_gfx = encoder->event_value;
      mtl_encode_wait_for_event(encoder->main.cmd_buffer, encoder->event,
                                ++encoder->event_value);
      encoder->signal_value_pre_gfx = encoder->event_value;

      encoder->main.encoder = mtl_new_render_command_encoder_with_descriptor(
         encoder->main.cmd_buffer, descriptor);
      /* limina probe: a nil encoder here is Metal refusing the render pass — most
       * often because an attachment texture is not a valid render target. It is
       * silent: the crash lands later, in kk_encoder_internal_end_encoding's
       * assert, with no clue as to which pass. Name it at the source. */
      if (!encoder->main.encoder)
         fprintf(stderr, "[LIMINA-KK-ENC] render command encoder is NIL "
                         "(descriptor=%p view_mask=0x%x) — attachment likely not a "
                         "valid Metal render target\n",
                 (void *)descriptor, view_mask);
      if (encoder->main.wait_fence) {
         mtl_render_wait_for_fence(
            encoder->main.encoder,
            util_dynarray_top(&encoder->main.fences, mtl_fence *));
         encoder->main.wait_fence = false;
      }

      uint32_t layer_ids[KK_MAX_MULTIVIEW_VIEW_COUNT] = {};
      uint32_t count = 0u;
      u_foreach_bit(id, view_mask)
         layer_ids[count++] = id;
      if (view_mask == 0u) {
         layer_ids[count++] = 0;
      }
      mtl_set_vertex_amplification_count(encoder->main.encoder, layer_ids,
                                         count);
      encoder->main.user_heap_hash = UINT32_MAX;

      /* Bind read only data aka samplers' argument buffer. */
      struct kk_device *dev = kk_cmd_buffer_device(cmd);
      mtl_set_vertex_buffer(encoder->main.encoder, dev->samplers.table.bo->map,
                            0u, 1u);
      mtl_set_fragment_buffer(encoder->main.encoder,
                              dev->samplers.table.bo->map, 0u, 1u);

      /* limina: fresh encoder = fresh binding tables — drop the bind cache. */
      struct kk_graphics_state *gfx = &cmd->state.gfx;
      memset(&gfx->bind_cache_v0, 0, sizeof(gfx->bind_cache_v0));
      memset(&gfx->bind_cache_f0, 0, sizeof(gfx->bind_cache_f0));
      memset(&gfx->bind_cache_v2, 0, sizeof(gfx->bind_cache_v2));
      memset(&gfx->bind_cache_f2, 0, sizeof(gfx->bind_cache_f2));
   }
   encoder->main.last_used = KK_ENC_RENDER;
   return encoder->main.encoder;
}

void
kk_encoder_end(struct kk_cmd_buffer *cmd)
{
   assert(cmd);

   kk_encoder_signal_fence_and_end(cmd);

   /* Ensure all writes are done before we really end encoding. We could have
    * writes to be done in the following case: endRenderPasss->endQuery->cmdEnd */
   upload_queue_writes(cmd);
   kk_encoder_signal_fence_and_end(cmd);

   struct kk_encoder *encoder = cmd->encoder;
   if (encoder->last_signaled_value_pre_gfx != encoder->signal_value_pre_gfx) {
      mtl_encode_signal_event(encoder->pre_gfx.cmd_buffer, encoder->event,
                              encoder->signal_value_pre_gfx);
      encoder->last_signaled_value_pre_gfx = encoder->signal_value_pre_gfx;
   }
}

struct kk_imm_write_push {
   uint64_t buffer_address;
   uint32_t count;
};

void
upload_queue_writes(struct kk_cmd_buffer *cmd)
{
   struct kk_encoder *enc = cmd->encoder;

   if (enc->imm_writes.size == 0u &&
       enc->copy_query_pool_result_infos.size == 0u)
      return;

   uint32_t count =
      util_dynarray_num_elements(&enc->imm_writes, struct libkk_imm_write);
   if (count != 0) {
      struct kk_ptr data_gpu =
         kk_pool_upload(cmd, enc->imm_writes.data, enc->imm_writes.size,
                        sizeof(struct libkk_imm_write));
      /* kk_cmd_allocate_buffer sets the cmd buffer error so we can just exit */
      if (unlikely(!data_gpu.gpu))
         return;
      libkk_write_u32_array(cmd, kk_grid_1d(count), false, data_gpu.gpu);
      enc->imm_writes.size = 0u;
   }

   count = util_dynarray_num_elements(&enc->copy_query_pool_result_infos,
                                      struct libkk_copy_queries_args);
   if (count != 0u) {
      for (uint32_t i = 0u; i < count; ++i) {
         struct kk_copy_query_pool_results_info *push_data =
            util_dynarray_element(&enc->copy_query_pool_result_infos,
                                  struct kk_copy_query_pool_results_info, i);

         const struct libkk_copy_queries_args *data =
            (const struct libkk_copy_queries_args *)push_data;
         libkk_copy_queries_struct(cmd, kk_grid_1d(push_data->query_count),
                                   false, *data);
      }
      enc->copy_query_pool_result_infos.size = 0u;
   }
}

static void
kk_encoder_signal_fence(struct kk_encoder *encoder)
{
   assert(encoder);
   enum kk_encoder_type type = encoder->main.last_used;
   struct kk_encoder_internal *main_enc = &encoder->main;
   if (!main_enc->encoder)
      return;

   mtl_fence *fence = mtl_new_fence(encoder->dev);
   switch (type) {
   case KK_ENC_RENDER:
      mtl_render_update_fence(main_enc->encoder, fence);
      break;
   case KK_ENC_COMPUTE:
      mtl_compute_update_fence(main_enc->encoder, fence);
      break;
   case KK_ENC_BLIT:
      mtl_blit_update_fence(main_enc->encoder, fence);
      break;
   default:
      assert(0);
      break;
   }
   main_enc->wait_fence = true;
   util_dynarray_append(&main_enc->fences, fence);
}

static void
kk_encoder_internal_end_encoding(struct kk_encoder_internal *internal_encoder)
{
   assert(internal_encoder && internal_encoder->encoder);

   mtl_end_encoding(internal_encoder->encoder);
   mtl_release(internal_encoder->encoder);
   internal_encoder->encoder = NULL;
   internal_encoder->last_used = KK_ENC_NONE;
}

void
kk_encoder_signal_fence_and_end(struct kk_cmd_buffer *cmd)
{
   struct kk_encoder *encoder = cmd->encoder;
   assert(encoder);
   /* End pre_gfx */
   if (encoder->pre_gfx.encoder) {
      kk_encoder_internal_end_encoding(&encoder->pre_gfx);

      /* We can start rendering once all pre-graphics work is done */
      mtl_encode_signal_event(encoder->pre_gfx.cmd_buffer, encoder->event,
                              encoder->signal_value_pre_gfx);
      encoder->last_signaled_value_pre_gfx = encoder->signal_value_pre_gfx;
   }

   if (encoder->main.last_used != KK_ENC_NONE) {
      /* kk_encoder_internal_end_encoding will change this value to NONE */
      enum kk_encoder_type last = encoder->main.last_used;
      kk_encoder_signal_fence(encoder);
      kk_encoder_internal_end_encoding(&encoder->main);
      if (last == KK_ENC_RENDER) {
         mtl_encode_signal_event(encoder->main.cmd_buffer, encoder->event,
                                 ++encoder->event_value);
      }
   }

   if (cmd->drawable) {
      mtl_present_drawable(encoder->main.cmd_buffer, cmd->drawable);
      cmd->drawable = NULL;
   }
}

static void
kk_post_execution_release_internal(struct kk_encoder_internal *encoder)
{
   mtl_release(encoder->cmd_buffer);
   util_dynarray_foreach(&encoder->fences, mtl_fence *, fence)
      mtl_release(*fence);
   util_dynarray_fini(&encoder->fences);
}

void
kk_timestamp_batch_publish(struct kk_timestamp_batch *batch)
{
   util_dynarray_foreach(&batch->samples, struct kk_timestamp_sample, s) {
      uint64_t ns = mtl_counter_sample_buffer_resolve(s->sb, 0u);

      if (kk_ts_trace())
         fprintf(stderr, "[KKTS] resolve seq=%llu sb=%p report=%p value=%llu\n",
                 (unsigned long long)batch->seq, (void *)s->sb,
                 (void *)s->report, (unsigned long long)ns);

      /* A sample that never materialised leaves the report at the UINT64_MAX
       * the sampling encoder filled in, i.e. unavailable. Reporting nothing is
       * the honest answer and the one every caller can act on; a zero would
       * read back as a query that resolved to time zero. */
      if (ns != 0u && ns != UINT64_MAX)
         __atomic_store_n(s->report, ns, __ATOMIC_RELEASE);
   }
}

static void
kk_timestamp_batch_complete(void *data)
{
   struct kk_timestamp_batch *batch = data;

   /* Off the in-flight list first: from here on nobody else may touch it. */
   kk_timestamp_batch_untrack(batch->dev, batch);

   kk_timestamp_batch_publish(batch);

   util_dynarray_foreach(&batch->samples, struct kk_timestamp_sample, s)
      mtl_release(s->sb);
   util_dynarray_fini(&batch->samples);

   /* Only now are the reports guaranteed visible, so only now may a waiter
    * proceed. */
   kk_timestamp_seq_retire(batch->dev, batch->seq);
   free(batch);
}

/* Hand the pending samples to `cb`, whose completion resolves them. */
static void
kk_encoder_arm_timestamp_batch(struct kk_encoder *encoder,
                               mtl_command_buffer *cb)
{
   struct kk_timestamp_batch *batch = encoder->ts_batch;
   if (!batch)
      return;

   encoder->ts_batch = NULL;
   kk_timestamp_batch_track(batch->dev, batch);
   mtl_add_completed_handler(cb, kk_timestamp_batch_complete, batch);
}

static void
kk_post_execution_release(void *data)
{
   struct kk_encoder *encoder = data;
   util_dynarray_foreach(&encoder->extra_cmd_buffers, mtl_command_buffer *, cb)
      mtl_release(*cb);
   util_dynarray_fini(&encoder->extra_cmd_buffers);
   kk_post_execution_release_internal(&encoder->main);
   kk_post_execution_release_internal(&encoder->pre_gfx);
   mtl_release(encoder->event);
   util_dynarray_fini(&encoder->imm_writes);
   util_dynarray_fini(&encoder->copy_query_pool_result_infos);
   util_dynarray_fini(&encoder->deferred_timestamps);
   free(encoder);
}

void
kk_encoder_submit(struct kk_encoder *encoder)
{
   assert(encoder);

   /* Ahead of the release handler: the batch must resolve its samples before
    * anything else is torn down, and it is what unblocks GPU/CPU waiters. */
   kk_encoder_arm_timestamp_batch(encoder, encoder->main.cmd_buffer);

   mtl_add_completed_handler(encoder->main.cmd_buffer,
                             kk_post_execution_release, encoder);

   mtl_command_buffer_commit(encoder->pre_gfx.cmd_buffer);
   /* Command buffers execute in commit order on a queue, so committing the
    * retired ones first keeps a split transparent to command order. `main` is
    * committed last and carries the completion handler, so it is still the one
    * whose completion releases the encoder. */
   util_dynarray_foreach(&encoder->extra_cmd_buffers, mtl_command_buffer *, cb)
      mtl_command_buffer_commit(*cb);
   mtl_command_buffer_commit(encoder->main.cmd_buffer);
}

mtl_render_encoder *
kk_render_encoder(struct kk_cmd_buffer *cmd)
{
   struct kk_encoder *encoder = cmd->encoder;

   struct kk_graphics_state *gfx = &cmd->state.gfx;

   if (gfx->need_to_start_render_pass) {
      /* limina: Metal requires defaultRasterSampleCount >= 1; an attachment-less
       * render pass has no attachment to derive it from and a draw can reach here
       * before a pipeline sets it, leaving 0 -> mtl_new_render_command_encoder
       * returns nil -> the assert below. Clamp. */
      gfx->render.samples =
         gfx->pipeline_sample_count ? gfx->pipeline_sample_count : 1u;
      mtl_render_pass_descriptor_set_default_raster_sample_count(
         cmd->state.gfx.render_pass_descriptor, gfx->render.samples);
      gfx->need_to_start_render_pass = false;
      kk_encoder_start_render(cmd, gfx->render_pass_descriptor,
                              gfx->render.view_mask);
   }

   /* Render encoders are created at vkBeginRendering only */
   assert(encoder->main.last_used == KK_ENC_RENDER && encoder->main.encoder);
   return (mtl_render_encoder *)encoder->main.encoder;
}

mtl_compute_encoder *
kk_compute_encoder(struct kk_cmd_buffer *cmd)
{
   struct kk_encoder_internal *encoder = &cmd->encoder->main;

   if (encoder->last_used == KK_ENC_COMPUTE)
      return (mtl_compute_encoder *)encoder->encoder;

   kk_encoder_signal_fence_and_end(cmd);
   encoder->encoder = mtl_new_compute_command_encoder(encoder->cmd_buffer);
   if (encoder->wait_fence) {
      mtl_compute_wait_for_fence(
         encoder->encoder, util_dynarray_top(&encoder->fences, mtl_fence *));
      encoder->wait_fence = false;
   }
   encoder->user_heap_hash = UINT32_MAX;

   /* Bind read only data aka samplers' argument buffer. */
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   mtl_compute_set_buffer(encoder->encoder, dev->samplers.table.bo->map, 0u,
                          1u);
   encoder->last_used = KK_ENC_COMPUTE;
   upload_queue_writes(cmd);
   return (mtl_compute_encoder *)encoder->encoder;
}

mtl_blit_encoder *
kk_blit_encoder(struct kk_cmd_buffer *cmd)
{
   struct kk_encoder_internal *encoder = &cmd->encoder->main;

   if (encoder->last_used == KK_ENC_BLIT)
      return (mtl_blit_encoder *)encoder->encoder;

   upload_queue_writes(cmd);
   kk_encoder_signal_fence_and_end(cmd);
   encoder->encoder = mtl_new_blit_command_encoder(encoder->cmd_buffer);
   if (encoder->wait_fence) {
      mtl_compute_wait_for_fence(
         encoder->encoder, util_dynarray_top(&encoder->fences, mtl_fence *));
      encoder->wait_fence = false;
   }
   encoder->last_used = KK_ENC_BLIT;
   return (mtl_blit_encoder *)encoder->encoder;
}

/* Retire the current `main` command buffer and continue recording into a fresh
 * one. Everything already encoded keeps its place: the retired buffer is
 * committed ahead of the new one, and the fence signalled by the last encoder
 * before the split is waited on by the first encoder after it — the same
 * fence-across-command-buffers chaining kk_queue_submit already uses to link
 * consecutive submissions.
 *
 * Used to close a command buffer that carries timestamp samples so it can
 * complete — its completion is what makes those samples readable. The caller
 * must have ended encoding first. */
static void
kk_encoder_split_main(struct kk_encoder *encoder)
{
   assert(encoder->main.encoder == NULL);

   kk_encoder_arm_timestamp_batch(encoder, encoder->main.cmd_buffer);
   util_dynarray_append(&encoder->extra_cmd_buffers, encoder->main.cmd_buffer);

   encoder->main.cmd_buffer = mtl_new_command_buffer(encoder->main_queue);
   encoder->main.last_used = KK_ENC_NONE;
   /* A fresh command buffer makes no heap resident. */
   encoder->main.user_heap_hash = UINT32_MAX;
   if (util_dynarray_num_elements(&encoder->main.fences, mtl_fence *) > 0)
      encoder->main.wait_fence = true;
}

void
kk_encoder_write_timestamp(struct kk_cmd_buffer *cmd,
                           const struct kk_timestamp_target *target)
{
   struct kk_encoder *enc = cmd->encoder;
   struct kk_encoder_internal *main = &enc->main;

   /* Flush queued query/imm writes and close the current encoder so the GPU
    * clock is sampled at a clean stage boundary, after all prior work. */
   upload_queue_writes(cmd);
   kk_encoder_signal_fence_and_end(cmd);

   mtl_counter_sample_buffer *sb = mtl_new_timestamp_sample_buffer(enc->dev, 1u);
   if (!sb) {
      /* Out of resources: the report stays UINT64_MAX (unavailable). A missing
       * result is safer than a wrong one; this path has no way to fail the cmd
       * buffer, and timestamp queries are rare. */
      return;
   }

   if (!enc->ts_batch) {
      struct kk_timestamp_batch *batch = calloc(1u, sizeof(*batch));
      if (!batch) {
         mtl_release(sb);
         return;
      }
      batch->dev = enc->device;
      batch->seq = kk_timestamp_seq_alloc(enc->device);
      batch->samples = (struct util_dynarray)UTIL_DYNARRAY_INIT;
      enc->ts_batch = batch;
   }

   /* Sampling encoder: latches the GPU clock at its start boundary. */
   main->encoder =
      mtl_new_blit_command_encoder_timestamp(main->cmd_buffer, sb, 0u);

   /* Mark the report unavailable, in GPU command order, at exactly the point
    * the clock is sampled. Two jobs in one fill:
    *
    * 1. Correctness. The resolved value is written by the CPU when this command
    *    buffer completes, so between here and there the report must not read as
    *    a finished query. A TIMESTAMP pool has no availability word — the
    *    convention is value != UINT64_MAX (libkk/kk_query.cl), so 0xff bytes
    *    say "not yet". Ordering it here also means an earlier in-stream
    *    vkCmdResetQueryPool cannot clobber it, and a later one wins over it.
    *
    * 2. The sampling encoder MUST carry real work. Metal elides a blit encoder
    *    that encodes nothing, and an elided encoder never takes its counter
    *    sample — the report then keeps whatever it held, with no hint that
    *    anything went wrong. Fence waits and signals are synchronisation, not
    *    data movement, and do not count. Measured on M4 Pro: with no work here
    *    a CPU resolveCounterRange: of the sample buffer reads 0 in 20 runs out
    *    of 20; with this fill it reads a real timestamp 20 out of 20. M1 Max
    *    does not elide it, which is why this went unnoticed there. */
   mtl_blit_fill_buffer(main->encoder, target->dst, target->offset,
                        sizeof(uint64_t), 0xffu);

   if (main->wait_fence) {
      mtl_blit_wait_for_fence(main->encoder,
                              util_dynarray_top(&main->fences, mtl_fence *));
      main->wait_fence = false;
   }
   main->last_used = KK_ENC_BLIT;
   kk_encoder_signal_fence(enc);
   kk_encoder_internal_end_encoding(main);

   /* No GPU resolve. -[MTLBlitCommandEncoder resolveCounters:] cannot be
    * trusted: the sample only materialises when the command buffer that took it
    * COMPLETES, and a resolve encoded before that silently writes zero rather
    * than failing. On M4 Pro that is 94% of the time from the same command
    * buffer and still 18% from a later one (50 runs each) — and since KK's
    * availability test for a timestamp is "value != UINT64_MAX", a zero reads
    * back as a query that legitimately resolved to zero. Nothing downstream can
    * tell the two apart, which is what made this cost a day of guest-side
    * investigation.
    *
    * So the value is read on the CPU in the completion handler instead
    * (mtl_counter_sample_buffer_resolve — 50 runs out of 50 on both M1 Max and
    * M4 Pro), and everything that could observe the report earlier is ordered
    * against that write through kk_encoder_timestamp_barrier. */
   struct kk_timestamp_sample sample = {.sb = sb, .report = target->report};
   util_dynarray_append_typed(&enc->ts_batch->samples,
                              struct kk_timestamp_sample, sample);

   /* Publish before returning: from here on, a consumer of this pool must
    * barrier against this sequence number. Monotonic max rather than a plain
    * store -- two threads may be recording timestamps into the same pool, and
    * the pool has to end up naming the LATEST write to wait for. */
   uint64_t seen = __atomic_load_n(target->pending_seq, __ATOMIC_RELAXED);
   while (seen < enc->ts_batch->seq &&
          !__atomic_compare_exchange_n(target->pending_seq, &seen,
                                       enc->ts_batch->seq, true,
                                       __ATOMIC_RELEASE, __ATOMIC_RELAXED))
      ;

   if (kk_ts_trace())
      fprintf(stderr, "[KKTS] write_timestamp seq=%llu sb=%p dst=%p off=%llu\n",
              (unsigned long long)enc->ts_batch->seq, (void *)sb,
              (void *)target->dst, (unsigned long long)target->offset);
}

void
kk_encoder_timestamp_barrier(struct kk_cmd_buffer *cmd, uint64_t seq)
{
   struct kk_encoder *enc = cmd->encoder;
   struct kk_device *dev = enc->device;

   /* Already written, and a CPU write that has landed is visible to the GPU
    * (the report lives in shared storage). Nothing to order. */
   if (seq == 0u || seq <= kk_timestamp_seq_signalled(dev))
      return;

   /* The pending write happens in a completion handler, so it cannot happen
    * while the command buffer that holds the samples is still running. Waiting
    * for it from inside that same command buffer would deadlock — close it
    * first. Splits are committed ahead of `main`, so command order survives. */
   kk_encoder_signal_fence_and_end(cmd);
   if (enc->ts_batch)
      kk_encoder_split_main(enc);

   mtl_encode_wait_for_event(enc->main.cmd_buffer, kk_timestamp_event(dev),
                             seq);

   if (kk_ts_trace())
      fprintf(stderr, "[KKTS] barrier seq=%llu (signalled=%llu)\n",
              (unsigned long long)seq,
              (unsigned long long)kk_timestamp_seq_signalled(dev));
}

void
kk_encoder_defer_timestamp(struct kk_cmd_buffer *cmd,
                           const struct kk_timestamp_target *target)
{
   util_dynarray_append_typed(&cmd->encoder->deferred_timestamps,
                              struct kk_timestamp_target, *target);
}

void
kk_encoder_flush_deferred_timestamps(struct kk_cmd_buffer *cmd)
{
   struct util_dynarray *arr = &cmd->encoder->deferred_timestamps;
   util_dynarray_foreach(arr, struct kk_timestamp_target, ts)
      kk_encoder_write_timestamp(cmd, ts);
   util_dynarray_clear(arr);
}

mtl_compute_encoder *
kk_encoder_pre_gfx_encoder(struct kk_cmd_buffer *cmd)
{
   struct kk_encoder *encoder = cmd->encoder;
   if (!encoder->pre_gfx.encoder) {
      /* Fast-forward all previous render encoders and wait for the last one */
      uint32_t last_signaled = (encoder->wait_value_pre_gfx - 1u);
      if (encoder->wait_value_pre_gfx != 0u &&
          encoder->last_signaled_value_pre_gfx != last_signaled) {
         mtl_encode_signal_event(encoder->pre_gfx.cmd_buffer, encoder->event,
                                 last_signaled);
         encoder->last_signaled_value_pre_gfx = last_signaled;
      }
      mtl_encode_wait_for_event(encoder->pre_gfx.cmd_buffer, encoder->event,
                                encoder->wait_value_pre_gfx);
      encoder->pre_gfx.encoder =
         mtl_new_compute_command_encoder(encoder->pre_gfx.cmd_buffer);
      struct kk_device *dev = kk_cmd_buffer_device(cmd);
      /* Xcode GPU capture doesn't like it when we declare an input buffer but
       * none is bound. Even if we never use it... */
      mtl_compute_set_buffer(encoder->pre_gfx.encoder,
                             dev->samplers.table.bo->map, 0u, 1u);
   }

   return encoder->pre_gfx.encoder;
}
