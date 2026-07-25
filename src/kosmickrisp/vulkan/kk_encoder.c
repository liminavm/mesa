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
   kk_encoder_start_internal(&enc->main, device, queue->main.mtl_handle);
   kk_encoder_start_internal(&enc->pre_gfx, device, queue->pre_gfx.mtl_handle);
   enc->main_queue = queue->main.mtl_handle;
   enc->extra_cmd_buffers = UTIL_DYNARRAY_INIT;
   enc->event = mtl_new_event(device);
   enc->imm_writes = UTIL_DYNARRAY_INIT;
   enc->copy_query_pool_result_infos = UTIL_DYNARRAY_INIT;
   enc->timestamp_sample_buffers = UTIL_DYNARRAY_INIT;
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
   util_dynarray_foreach(&encoder->timestamp_sample_buffers,
                         mtl_counter_sample_buffer *, sb)
      mtl_release(*sb);
   util_dynarray_fini(&encoder->timestamp_sample_buffers);
   util_dynarray_fini(&encoder->deferred_timestamps);
   free(encoder);
}

void
kk_encoder_submit(struct kk_encoder *encoder)
{
   assert(encoder);

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
 * Only used to work around GPUs that cannot resolve a counter sample from the
 * command buffer that took it (mtl_device_needs_split_counter_resolve). The
 * caller must have ended encoding first. */
static void
kk_encoder_split_main(struct kk_encoder *encoder)
{
   assert(encoder->main.encoder == NULL);

   util_dynarray_append(&encoder->extra_cmd_buffers, encoder->main.cmd_buffer);

   encoder->main.cmd_buffer = mtl_new_command_buffer(encoder->main_queue);
   encoder->main.last_used = KK_ENC_NONE;
   /* A fresh command buffer makes no heap resident. */
   encoder->main.user_heap_hash = UINT32_MAX;
   if (util_dynarray_num_elements(&encoder->main.fences, mtl_fence *) > 0)
      encoder->main.wait_fence = true;
}

void
kk_encoder_write_timestamp(struct kk_cmd_buffer *cmd, mtl_buffer *dst,
                           uint64_t dst_offset)
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
   util_dynarray_append(&enc->timestamp_sample_buffers, sb);

   /* Sampling encoder: latches the GPU clock at its start boundary. */
   main->encoder =
      mtl_new_blit_command_encoder_timestamp(main->cmd_buffer, sb, 0u);
   if (main->wait_fence) {
      mtl_blit_wait_for_fence(main->encoder,
                              util_dynarray_top(&main->fences, mtl_fence *));
      main->wait_fence = false;
   }
   main->last_used = KK_ENC_BLIT;
   kk_encoder_signal_fence(enc);
   kk_encoder_internal_end_encoding(main);

   /* Resolve encoder: MUST be distinct from the sampling encoder (resolving a
    * slot in its own sampling encoder reads zero). Writes the ns value into the
    * query report in GPU command order, so it orders correctly against an
    * in-stream CmdResetQueryPool / CmdCopyQueryPoolResults.
    *
    * On some GPUs a distinct ENCODER is not enough — the sample is not visible
    * to a resolve encoded in the same COMMAND BUFFER, which writes zero without
    * failing (measured on M4 Pro; M1 Max is unaffected). There, split the
    * command buffer so the resolve lands in the next one. Command buffers run
    * in commit order, so the in-stream ordering above is preserved. */
   if (mtl_device_needs_split_counter_resolve(enc->dev))
      kk_encoder_split_main(enc);

   main->encoder = mtl_new_blit_command_encoder(main->cmd_buffer);
   if (main->wait_fence) {
      mtl_blit_wait_for_fence(main->encoder,
                              util_dynarray_top(&main->fences, mtl_fence *));
      main->wait_fence = false;
   }
   main->last_used = KK_ENC_BLIT;
   mtl_blit_resolve_timestamp(main->encoder, sb, 0u, dst, dst_offset);
   kk_encoder_signal_fence(enc);
   kk_encoder_internal_end_encoding(main);
}

void
kk_encoder_defer_timestamp(struct kk_cmd_buffer *cmd, mtl_buffer *dst,
                           uint64_t dst_offset)
{
   struct kk_deferred_timestamp entry = {.dst = dst, .offset = dst_offset};
   util_dynarray_append(&cmd->encoder->deferred_timestamps, entry);
}

void
kk_encoder_flush_deferred_timestamps(struct kk_cmd_buffer *cmd)
{
   struct util_dynarray *arr = &cmd->encoder->deferred_timestamps;
   util_dynarray_foreach(arr, struct kk_deferred_timestamp, ts)
      kk_encoder_write_timestamp(cmd, ts->dst, ts->offset);
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
