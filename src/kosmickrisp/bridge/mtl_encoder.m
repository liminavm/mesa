/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_encoder.h"

#include <Metal/MTLBlitCommandEncoder.h>
#include <Metal/MTLBlitPass.h>
#include <Metal/MTLCaptureManager.h>
#include <Metal/MTLCaptureScope.h>
#include <Metal/MTLComputeCommandEncoder.h>
#include <Metal/MTLCounters.h>
#include <Metal/MTLRenderCommandEncoder.h>

/* limina: LIMINA_KK_STATS=1 — once-per-second aggregate counters to stderr.
 * Measures render-pass split rate (renc + Load-action reloads) vs draw rate. */
#include <stdatomic.h>
#include <time.h>

/* limina: RTLOG knob, cached — a getenv here sat on the per-draw path (round 24:
 * ~7% of the hot ring core in __findenv_locked). */
static inline bool
limina_kk_rtlog_cached(void)
{
   static int v = -1;
   if (v < 0)
      v = getenv("LIMINA_KK_RTLOG") != NULL;
   return v;
}
static _Atomic uint64_t limina_st_renc, limina_st_benc, limina_st_cenc,
   limina_st_draw, limina_st_loadc, limina_st_loadds;
static bool
limina_stats_on(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("LIMINA_KK_STATS") != NULL;
   return enabled;
}
static void
limina_stats_bump(_Atomic uint64_t *ctr)
{
   if (!limina_stats_on())
      return;
   atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
   /* clock_gettime per bump sat on the per-draw path (round 25: ~3% of the
    * hot ring thread); only poll the clock every 1024 bumps — at >1k events/s
    * the once-per-second print cadence is unaffected. */
   static _Atomic uint64_t bumps;
   if (atomic_fetch_add_explicit(&bumps, 1, memory_order_relaxed) & 1023)
      return;
   static _Atomic long last_sec;
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   long prev = atomic_load_explicit(&last_sec, memory_order_relaxed);
   if (ts.tv_sec != prev &&
       atomic_compare_exchange_strong(&last_sec, &prev, ts.tv_sec)) {
      fprintf(stderr,
              "[LIMINA-KK-STATS] renc=%llu loadC=%llu loadDS=%llu benc=%llu "
              "cenc=%llu draw=%llu (last interval)\n",
              (unsigned long long)atomic_exchange(&limina_st_renc, 0),
              (unsigned long long)atomic_exchange(&limina_st_loadc, 0),
              (unsigned long long)atomic_exchange(&limina_st_loadds, 0),
              (unsigned long long)atomic_exchange(&limina_st_benc, 0),
              (unsigned long long)atomic_exchange(&limina_st_cenc, 0),
              (unsigned long long)atomic_exchange(&limina_st_draw, 0));
   }
}

/* Common encoder utils */
void
mtl_end_encoding(void *encoder)
{
   @autoreleasepool {
      id<MTLCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      [enc endEncoding];
   }
}

/* MTLBlitEncoder */
mtl_blit_encoder *
mtl_new_blit_command_encoder(mtl_command_buffer *cmd_buffer)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>)cmd_buffer;
      limina_stats_bump(&limina_st_benc);
      return [[cmd_buf blitCommandEncoder] retain];
   }
}

void
mtl_blit_update_fence(mtl_blit_encoder *encoder,
                      mtl_fence *fence)
{
   @autoreleasepool {
      id<MTLBlitCommandEncoder> enc = (id<MTLBlitCommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc updateFence:f];
   }
}

void
mtl_blit_wait_for_fence(mtl_blit_encoder *encoder,
                            mtl_fence *fence)
{
   @autoreleasepool {
      id<MTLBlitCommandEncoder> enc = (id<MTLBlitCommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc waitForFence:f];
   }
}

/* A blit encoder that samples the GPU timestamp counter at its START stage
 * boundary into `sample_index` of `sample_buffer` (the only sampling point
 * Apple GPUs support). The sample is taken when the encoder begins, i.e. after
 * all prior (fence-ordered) work — a valid latch point for vkCmdWriteTimestamp. */
mtl_blit_encoder *
mtl_new_blit_command_encoder_timestamp(mtl_command_buffer *cmd_buffer,
                                       mtl_counter_sample_buffer *sample_buffer,
                                       uint32_t sample_index)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>)cmd_buffer;
      id<MTLCounterSampleBuffer> sb =
         (id<MTLCounterSampleBuffer>)sample_buffer;

      MTLBlitPassDescriptor *desc = [MTLBlitPassDescriptor blitPassDescriptor];
      MTLBlitPassSampleBufferAttachmentDescriptor *att =
         desc.sampleBufferAttachments[0];
      att.sampleBuffer = sb;
      att.startOfEncoderSampleIndex = sample_index;
      att.endOfEncoderSampleIndex = MTLCounterDontSample;
      return [[cmd_buf blitCommandEncoderWithDescriptor:desc] retain];
   }
}

/* Resolve one timestamp sample into `dst` at `dst_offset` (8 bytes: the ns
 * value). MUST be a different encoder than the one that took the sample —
 * resolving a slot in its own sampling encoder reads zero (verified). */
void
mtl_blit_resolve_timestamp(mtl_blit_encoder *encoder,
                           mtl_counter_sample_buffer *sample_buffer,
                           uint32_t sample_index, mtl_buffer *dst,
                           uint64_t dst_offset)
{
   @autoreleasepool {
      id<MTLBlitCommandEncoder> blit = (id<MTLBlitCommandEncoder>)encoder;
      id<MTLCounterSampleBuffer> sb =
         (id<MTLCounterSampleBuffer>)sample_buffer;
      id<MTLBuffer> dst_buf = (id<MTLBuffer>)dst;
      [blit resolveCounters:sb
                    inRange:NSMakeRange(sample_index, 1)
          destinationBuffer:dst_buf
          destinationOffset:dst_offset];
   }
}

void
mtl_copy_from_buffer_to_buffer(mtl_blit_encoder *blit_enc_handle,
                               mtl_buffer *src_buf, size_t src_offset,
                               mtl_buffer *dst_buf, size_t dst_offset,
                               size_t size)
{
   @autoreleasepool {
      id<MTLBlitCommandEncoder> blit = (id<MTLBlitCommandEncoder>)blit_enc_handle;
      id<MTLBuffer> mtl_src_buffer = (id<MTLBuffer>)src_buf;
      id<MTLBuffer> mtl_dst_buffer = (id<MTLBuffer>)dst_buf;
      if (limina_kk_rtlog_cached())
         fprintf(stderr, "[LIMINA-KK-B2B] src=%p+%zu dst=%p+%zu size=%zu\n",
                 (void *)mtl_src_buffer.contents, src_offset,
                 (void *)mtl_dst_buffer.contents, dst_offset, size);
      [blit copyFromBuffer:mtl_src_buffer sourceOffset:src_offset toBuffer:mtl_dst_buffer destinationOffset:dst_offset size:size];
   }
}

void
mtl_copy_from_buffer_to_texture(mtl_blit_encoder *blit_enc_handle,
                                struct mtl_buffer_image_copy *data)
{
   @autoreleasepool {
      const MTLSize size = MTLSizeMake(data->image_size.x, data->image_size.y, data->image_size.z);
      const MTLOrigin origin = MTLOriginMake(data->image_origin.x, data->image_origin.y, data->image_origin.z);
      id<MTLBlitCommandEncoder> blit = (id<MTLBlitCommandEncoder>)blit_enc_handle;
      id<MTLBuffer> buffer = (id<MTLBuffer>)data->buffer;
      id<MTLTexture> image = (id<MTLTexture>)data->image;
      [blit copyFromBuffer:buffer
              sourceOffset:data->buffer_offset_B
         sourceBytesPerRow:data->buffer_stride_B
       sourceBytesPerImage:data->buffer_2d_image_size_B
                sourceSize:size
                 toTexture:image
          destinationSlice:data->image_slice
          destinationLevel:data->image_level
         destinationOrigin:origin
                   options:(MTLBlitOption)data->options];
   }
}

void
mtl_copy_from_texture_to_buffer(mtl_blit_encoder *blit_enc_handle,
                                struct mtl_buffer_image_copy *data)
{
   @autoreleasepool {
      const MTLSize size = MTLSizeMake(data->image_size.x, data->image_size.y, data->image_size.z);
      const MTLOrigin origin = MTLOriginMake(data->image_origin.x, data->image_origin.y, data->image_origin.z);
      id<MTLBlitCommandEncoder> blit = (id<MTLBlitCommandEncoder>)blit_enc_handle;
      id<MTLBuffer> buffer = (id<MTLBuffer>)data->buffer;
      id<MTLTexture> image = (id<MTLTexture>)data->image;
      [blit copyFromTexture:image
                sourceSlice:data->image_slice
                sourceLevel:data->image_level
               sourceOrigin:origin
                 sourceSize:size
                   toBuffer:buffer
          destinationOffset:data->buffer_offset_B
     destinationBytesPerRow:data->buffer_stride_B
   destinationBytesPerImage:data->buffer_2d_image_size_B
                    options:(MTLBlitOption)data->options];
   }
}

void
mtl_copy_from_texture_to_texture(mtl_blit_encoder *blit_enc_handle,
                                 mtl_texture *src_tex_handle, size_t src_slice,
                                 size_t src_level, struct mtl_origin src_origin,
                                 struct mtl_size src_size,
                                 mtl_texture *dst_tex_handle, size_t dst_slice,
                                 size_t dst_level, struct mtl_origin dst_origin)
{
   @autoreleasepool {
      MTLOrigin mtl_src_origin = MTLOriginMake(src_origin.x, src_origin.y, src_origin.z);
      MTLSize mtl_src_size = MTLSizeMake(src_size.x, src_size.y, src_size.z);
      MTLOrigin mtl_dst_origin = MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z);
      id<MTLTexture> mtl_dst_tex = (id<MTLTexture>)dst_tex_handle;
      id<MTLBlitCommandEncoder> blit = (id<MTLBlitCommandEncoder>)blit_enc_handle;
      id<MTLTexture> mtl_src_tex = (id<MTLTexture>)src_tex_handle;
      [blit copyFromTexture:mtl_src_tex
                sourceSlice:src_slice
                sourceLevel:src_level
               sourceOrigin:mtl_src_origin
                 sourceSize:mtl_src_size
                  toTexture:mtl_dst_tex
           destinationSlice:dst_slice
           destinationLevel:dst_level
          destinationOrigin:mtl_dst_origin];
   }
}

/* MTLComputeEncoder */
mtl_compute_encoder *
mtl_new_compute_command_encoder(mtl_command_buffer *cmd_buffer)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>)cmd_buffer;
      limina_stats_bump(&limina_st_cenc);
      return [[cmd_buf computeCommandEncoder] retain];
   }
}

void
mtl_compute_update_fence(mtl_compute_encoder *encoder, mtl_fence *fence)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc updateFence:f];
   }
}

void
mtl_compute_wait_for_fence(mtl_compute_encoder *encoder, mtl_fence *fence)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc waitForFence:f];
   }
}

void
mtl_compute_set_pipeline_state(mtl_compute_encoder *encoder,
                               mtl_compute_pipeline_state *state_handle)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      id<MTLComputePipelineState> state = (id<MTLComputePipelineState>)state_handle;
      [enc setComputePipelineState:state];
   }
}

void
mtl_compute_set_buffer(mtl_compute_encoder *encoder,
                       mtl_buffer *buffer, size_t offset, size_t index)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      [enc setBuffer:buf offset:offset atIndex:index];
   }
}

void
mtl_compute_use_heaps(mtl_compute_encoder *encoder, mtl_heap **heaps,
                      uint32_t count)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      id<MTLHeap> *handles = (id<MTLHeap>*)heaps;
      [enc useHeaps:handles count:count];
   }
}

void
mtl_dispatch_threads(mtl_compute_encoder *encoder,
                     struct mtl_size grid_size, struct mtl_size local_size)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      MTLSize thread_count = MTLSizeMake(grid_size.x, grid_size.y, grid_size.z);
      MTLSize threads_per_threadgroup = MTLSizeMake(local_size.x, local_size.y,
                                                    local_size.z);

      // TODO_KOSMICKRISP can we rely on nonuniform threadgroup size support?
      [enc dispatchThreads:thread_count threadsPerThreadgroup:threads_per_threadgroup];
   }
}

void
mtl_dispatch_threadgroups_with_indirect_buffer(mtl_compute_encoder *encoder,
                                               mtl_buffer *buffer,
                                               uint32_t offset,
                                               struct mtl_size local_size)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      MTLSize threads_per_threadgroup = MTLSizeMake(local_size.x,
                                                    local_size.y,
                                                    local_size.z);

      [enc dispatchThreadgroupsWithIndirectBuffer:buf indirectBufferOffset:offset threadsPerThreadgroup:threads_per_threadgroup];
   }
}

void
mtl_memory_barrier_with_scope(mtl_compute_encoder *encoder,
                              enum mtl_barrier_scope scope)
{
   @autoreleasepool {
      id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder;
      [enc memoryBarrierWithScope:(MTLBarrierScope)scope];
   }
}

/* MTLRenderEncoder */

/* Encoder commands */
mtl_render_encoder *
mtl_new_render_command_encoder_with_descriptor(
   mtl_command_buffer *command_buffer, mtl_render_pass_descriptor *descriptor)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)command_buffer;
      MTLRenderPassDescriptor *desc = (MTLRenderPassDescriptor *)descriptor;
      /* limina: targeted single-pass GPU capture — arm with LIMINA_KK_CAPTURE=<W>;
       * captures the first render pass whose color0 is a buffer-backed (linear)
       * texture of width W AND has a depth attachment; writes /tmp/kk-pass.gputrace. */
      static int limina_captured = 0;
      const char *limina_capw = getenv("LIMINA_KK_CAPTURE");
      bool limina_capturing = false;
      if (limina_capw && !limina_captured) {
         id<MTLTexture> c0 = desc.colorAttachments[0].texture;
         id<MTLTexture> root = c0;
         while (root && !root.buffer && root.parentTexture)
            root = root.parentTexture;
         if (c0 && root && root.buffer && desc.depthAttachment.texture &&
             c0.width == (NSUInteger)atoi(limina_capw)) {
            MTLCaptureManager *mgr = [MTLCaptureManager sharedCaptureManager];
            MTLCaptureDescriptor *cd = [[MTLCaptureDescriptor alloc] init];
            cd.captureObject = [(id<MTLCommandBuffer>)cmd device];
            cd.destination = MTLCaptureDestinationGPUTraceDocument;
            cd.outputURL = [NSURL fileURLWithPath:@"/tmp/kk-pass.gputrace"];
            NSError *err = nil;
            if ([mgr startCaptureWithDescriptor:cd error:&err]) {
               limina_captured = 1;
               limina_capturing = true;
               fprintf(stderr, "[LIMINA-KK-CAPTURE] started for %sx pass\n", limina_capw);
            } else {
               fprintf(stderr, "[LIMINA-KK-CAPTURE] failed: %s\n",
                       err ? [[err description] UTF8String] : "?");
            }
            [cd release];
         }
      }
      limina_stats_bump(&limina_st_renc);
      if (limina_stats_on()) {
         if (desc.colorAttachments[0].texture &&
             desc.colorAttachments[0].loadAction == MTLLoadActionLoad)
            atomic_fetch_add_explicit(&limina_st_loadc, 1, memory_order_relaxed);
         if (desc.depthAttachment.texture &&
             desc.depthAttachment.loadAction == MTLLoadActionLoad)
            atomic_fetch_add_explicit(&limina_st_loadds, 1, memory_order_relaxed);
      }
      id<MTLRenderCommandEncoder> enc =
         [[cmd renderCommandEncoderWithDescriptor:desc] retain];
      if (limina_capturing) {
         [(id<MTLCommandBuffer>)cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull d) {
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            fprintf(stderr, "[LIMINA-KK-CAPTURE] stopped, trace at /tmp/kk-pass.gputrace\n");
         }];
      }
      /* limina probe: render-pass structure. A texture VIEW of a buffer-backed
       * (linear) texture reports .buffer == nil — walk parentTexture too. */
      if (limina_kk_rtlog_cached()) {
         id<MTLTexture> c0 = desc.colorAttachments[0].texture;
         id<MTLTexture> root = c0;
         while (root && !root.buffer && root.parentTexture)
            root = root.parentTexture;
         if (c0) {
            static int rp_count = 0;
            fprintf(stderr,
                    "[LIMINA-KK-RP] #%d enc=%p %lux%lu linear=%d view=%d load=%lu "
                    "store=%lu ds=%p dsload=%lu\n",
                    ++rp_count, (void *)enc, (unsigned long)c0.width,
                    (unsigned long)c0.height, root && root.buffer ? 1 : 0,
                    c0.parentTexture ? 1 : 0,
                    (unsigned long)desc.colorAttachments[0].loadAction,
                    (unsigned long)desc.colorAttachments[0].storeAction,
                    (void *)desc.depthAttachment.texture,
                    (unsigned long)desc.depthAttachment.loadAction);
         }
      }
      return enc;
   }
}

void
mtl_render_update_fence(mtl_render_encoder *encoder, mtl_fence *fence)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc updateFence:f afterStages:MTLRenderStageFragment];
   }
}

void
mtl_render_wait_for_fence(mtl_render_encoder *encoder, mtl_fence *fence)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc waitForFence:f beforeStages:MTLRenderStageVertex];
   }
}

void
mtl_set_viewports(mtl_render_encoder *encoder, struct mtl_viewport *viewports,
                  uint32_t count)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      MTLViewport *vps = (MTLViewport *)viewports;
      if (limina_kk_rtlog_cached())
         for (uint32_t i = 0; i < count; i++)
            fprintf(stderr, "[LIMINA-KK-VP] enc=%p vp[%u]=(%.1f,%.1f %.1fx%.1f z=%.2f..%.2f)\n",
                    (void *)enc, i, vps[i].originX, vps[i].originY, vps[i].width,
                    vps[i].height, vps[i].znear, vps[i].zfar);
      [enc setViewports:vps count:count];
   }
}

void
mtl_set_scissor_rects(mtl_render_encoder *encoder,
                      struct mtl_scissor_rect *scissor_rects, uint32_t count)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      MTLScissorRect *rects = (MTLScissorRect *)scissor_rects;
      if (limina_kk_rtlog_cached())
         for (uint32_t i = 0; i < count; i++)
            fprintf(stderr, "[LIMINA-KK-SC] enc=%p sc[%u]=(%lu,%lu %lux%lu)\n",
                    (void *)enc, i, (unsigned long)rects[i].x, (unsigned long)rects[i].y,
                    (unsigned long)rects[i].width, (unsigned long)rects[i].height);
      [enc setScissorRects:rects count:count];
   }
}

void
mtl_render_set_pipeline_state(mtl_render_encoder *encoder,
                              mtl_render_pipeline_state *pipeline)
{
   @autoreleasepool {
      if (limina_kk_rtlog_cached()) fprintf(stderr, "[LIMINA-KK-ST] mtl_render_set_pipeline_state enc=%p pso=%p\n", (void *)encoder, (void *)pipeline);
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLRenderPipelineState> pipe = (id<MTLRenderPipelineState>)pipeline;
      [enc setRenderPipelineState:pipe];
   }
}

void
mtl_set_depth_stencil_state(mtl_render_encoder *encoder,
                            mtl_depth_stencil_state *state)
{
   @autoreleasepool {
      if (limina_kk_rtlog_cached()) fprintf(stderr, "[LIMINA-KK-ST] mtl_set_depth_stencil_state enc=%p dss=%p\n", (void *)encoder, (void *)state);
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLDepthStencilState> s = (id<MTLDepthStencilState>)state;
      [enc setDepthStencilState:s];
   }
}

void
mtl_set_stencil_references(mtl_render_encoder *encoder, uint32_t front,
                           uint32_t back)
{
   @autoreleasepool {
      if (limina_kk_rtlog_cached()) fprintf(stderr, "[LIMINA-KK-ST] mtl_set_stencil_references enc=%p\n", (void *)encoder);
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setStencilFrontReferenceValue:front backReferenceValue:back];
   }
}

void
mtl_set_front_face_winding(mtl_render_encoder *encoder,
                           enum mtl_winding winding)
{
   @autoreleasepool {
      if (limina_kk_rtlog_cached()) fprintf(stderr, "[LIMINA-KK-ST] mtl_set_front_face_winding enc=%p wind=%lu\n", (void *)encoder, (unsigned long)winding);
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setFrontFacingWinding:(MTLWinding)winding];
   }
}

void
mtl_set_cull_mode(mtl_render_encoder *encoder, enum mtl_cull_mode mode)
{
   @autoreleasepool {
      if (limina_kk_rtlog_cached()) fprintf(stderr, "[LIMINA-KK-ST] mtl_set_cull_mode enc=%p mode=%lu\n", (void *)encoder, (unsigned long)mode);
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setCullMode:(MTLCullMode)mode];
   }
}

void
mtl_set_visibility_result_mode(mtl_render_encoder *encoder,
                               enum mtl_visibility_result_mode mode,
                               size_t offset)
{
   @autoreleasepool {
      if (limina_kk_rtlog_cached()) fprintf(stderr, "[LIMINA-KK-ST] mtl_set_visibility_result_mode enc=%p mode=%lu off=%zu\n", (void *)encoder, (unsigned long)mode, offset);
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:offset];
   }
}

void
mtl_set_depth_bias(mtl_render_encoder *encoder, float depth_bias,
                   float slope_scale, float clamp)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setDepthBias:depth_bias slopeScale:slope_scale clamp:clamp];
   }
}

void
mtl_set_depth_clip_mode(mtl_render_encoder *encoder,
                        enum mtl_depth_clip_mode mode)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setDepthClipMode:(MTLDepthClipMode)mode];
   }
}

void
mtl_set_vertex_amplification_count(mtl_render_encoder *encoder,
                                   uint32_t *layer_ids, uint32_t id_count)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      MTLVertexAmplificationViewMapping mappings[32];
      for (uint32_t i = 0u; i < id_count; ++i) {
         mappings[i].renderTargetArrayIndexOffset = layer_ids[i];
         mappings[i].viewportArrayIndexOffset = 0u;
      }
      [enc setVertexAmplificationCount:id_count viewMappings:mappings];
   }
}

void
mtl_set_vertex_buffer(mtl_render_encoder *encoder, mtl_buffer *buffer,
                      uint32_t offset, uint32_t index)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      if (limina_kk_rtlog_cached()) {
         const float *f = buf.contents ? (const float *)((const char *)buf.contents + offset) : NULL;
         size_t nz = 0, len = buf.length, firstnz[8], nfound = 0;
         const unsigned char *raw = (const unsigned char *)buf.contents;
         if (raw)
            for (size_t i = 0; i < len; i++)
               if (raw[i]) { if (nfound < 8) firstnz[nfound++] = i; nz++; }
         fprintf(stderr,
                 "[LIMINA-KK-VB] enc=%p idx=%u off=%lu len=%lu base=%p nonzero=%zu "
                 "nzoff=[%zd %zd %zd %zd %zd %zd %zd %zd] first4=(%.2f %.2f %.2f %.2f)\n",
                 (void *)enc, index, (unsigned long)offset, (unsigned long)len,
                 (void *)raw, nz,
                 nfound > 0 ? (ssize_t)firstnz[0] : -1, nfound > 1 ? (ssize_t)firstnz[1] : -1,
                 nfound > 2 ? (ssize_t)firstnz[2] : -1, nfound > 3 ? (ssize_t)firstnz[3] : -1,
                 nfound > 4 ? (ssize_t)firstnz[4] : -1, nfound > 5 ? (ssize_t)firstnz[5] : -1,
                 nfound > 6 ? (ssize_t)firstnz[6] : -1, nfound > 7 ? (ssize_t)firstnz[7] : -1,
                 f ? f[0] : -999, f ? f[1] : -999, f ? f[2] : -999, f ? f[3] : -999);
         if (raw && nfound > 0) {
            size_t s0 = firstnz[0] & ~(size_t)3;
            const float *g = (const float *)(raw + s0);
            fprintf(stderr, "[LIMINA-KK-VBF] @%zu: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
                    s0, g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], g[10], g[11]);
         }
      }
      [enc setVertexBuffer:buf offset:offset atIndex:index];
   }
}

/* limina: offset-only rebinds — valid only while a buffer is bound at index on
 * this encoder (the caller's bind cache guarantees it). Much cheaper than
 * setBuffer (no residency/binding-table update); no autoreleasepool needed
 * (creates no objects). */
void
mtl_set_vertex_buffer_offset(mtl_render_encoder *encoder, uint32_t offset,
                             uint32_t index)
{
   id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
   [enc setVertexBufferOffset:offset atIndex:index];
}

void
mtl_set_fragment_buffer_offset(mtl_render_encoder *encoder, uint32_t offset,
                               uint32_t index)
{
   id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
   [enc setFragmentBufferOffset:offset atIndex:index];
}

void
mtl_set_vertex_bytes(mtl_render_encoder *encoder, const void *bytes,
                     uint32_t length, uint32_t index)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setVertexBytes:bytes length:length atIndex:index];
   }
}

void
mtl_set_fragment_buffer(mtl_render_encoder *encoder, mtl_buffer *buffer,
                        uint32_t offset, uint32_t index)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      [enc setFragmentBuffer:buf offset:offset atIndex:index];
   }
}

void
mtl_set_fragment_bytes(mtl_render_encoder *encoder, const void *bytes,
                       uint32_t length, uint32_t index)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      [enc setFragmentBytes:bytes length:length atIndex:index];
   }
}

void
mtl_draw_primitives(mtl_render_encoder *encoder,
                    enum mtl_primitive_type primitve_type, uint32_t vertexStart,
                    uint32_t vertexCount, uint32_t instanceCount,
                    uint32_t baseInstance)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      MTLPrimitiveType type = (MTLPrimitiveType)primitve_type;
      if (limina_kk_rtlog_cached())
         fprintf(stderr, "[LIMINA-KK-DRAW] enc=%p type=%lu count=%u\n", (void *)enc,
                 (unsigned long)type, vertexCount);
      limina_stats_bump(&limina_st_draw);
      [enc drawPrimitives:type vertexStart:vertexStart vertexCount:vertexCount instanceCount:instanceCount baseInstance:baseInstance];
   }
}

void
mtl_draw_indexed_primitives(
   mtl_render_encoder *encoder, enum mtl_primitive_type primitve_type,
   uint32_t index_count, enum mtl_index_type index_type,
   mtl_buffer *index_buffer, uint32_t index_buffer_offset,
   uint32_t instance_count, int32_t base_vertex, uint32_t base_instance)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLBuffer> buf = (id<MTLBuffer>)index_buffer;
      MTLIndexType ndx_type = (MTLIndexType)index_type;
      MTLPrimitiveType primitive = (MTLPrimitiveType)primitve_type;
      limina_stats_bump(&limina_st_draw);
      [enc drawIndexedPrimitives:primitive indexCount:index_count indexType:ndx_type indexBuffer:buf indexBufferOffset:index_buffer_offset instanceCount:instance_count baseVertex:base_vertex baseInstance:base_instance];
   }
}

void
mtl_draw_primitives_indirect(mtl_render_encoder *encoder,
                             enum mtl_primitive_type primitve_type,
                             mtl_buffer *indirect_buffer,
                             uint64_t indirect_buffer_offset)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLBuffer> buf = (id<MTLBuffer>)indirect_buffer;
      MTLPrimitiveType type = (MTLPrimitiveType)primitve_type;
      limina_stats_bump(&limina_st_draw);
      [enc drawPrimitives:type indirectBuffer:buf indirectBufferOffset:indirect_buffer_offset];
   }
}

void
mtl_draw_indexed_primitives_indirect(mtl_render_encoder *encoder,
                                     enum mtl_primitive_type primitve_type,
                                     enum mtl_index_type index_type,
                                     mtl_buffer *index_buffer,
                                     uint32_t index_buffer_offset,
                                     mtl_buffer *indirect_buffer,
                                     uint64_t indirect_buffer_offset)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLBuffer> buf = (id<MTLBuffer>)indirect_buffer;
      id<MTLBuffer> ndx_buf = (id<MTLBuffer>)index_buffer;
      MTLPrimitiveType type = (MTLPrimitiveType)primitve_type;
      MTLIndexType ndx_type = (MTLIndexType)index_type;
      limina_stats_bump(&limina_st_draw);
      [enc drawIndexedPrimitives:type indexType:ndx_type indexBuffer:ndx_buf indexBufferOffset:index_buffer_offset indirectBuffer:buf indirectBufferOffset:indirect_buffer_offset];
   }
}

void
mtl_render_use_resource(mtl_compute_encoder *encoder, mtl_resource *res_handle,
                        uint32_t usage)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLResource> res = (id<MTLResource>)res_handle;
      [enc useResource:res usage:(MTLResourceUsage)usage stages:MTLRenderStageVertex|MTLRenderStageFragment];
   }
}

void
mtl_render_use_resources(mtl_render_encoder *encoder,
                         mtl_resource **resource_handles, uint32_t count,
                         enum mtl_resource_usage usage)
{
   @autoreleasepool {
      // id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLResource> *handles = (id<MTLResource>*)resource_handles;
      for (uint32_t i = 0u; i < count; ++i) {
         if (handles[i] != NULL)
            mtl_render_use_resource(encoder, handles[i], usage);
      }
      /* TODO_KOSMICKRISP No null values in the array or Metal complains */
      // [enc useResources:handles count:count usage:(MTLResourceUsage)usage];
   }
}

void
mtl_render_use_heaps(mtl_render_encoder *encoder, mtl_heap **heaps,
                     uint32_t count)
{
   @autoreleasepool {
      id<MTLRenderCommandEncoder> enc = (id<MTLRenderCommandEncoder>)encoder;
      id<MTLHeap> *handles = (id<MTLHeap>*)heaps;
      [enc useHeaps:handles count:count stages:MTLRenderStageVertex|MTLRenderStageFragment];
   }
}

void
mtl_blit_fill_buffer(mtl_blit_encoder *blit_enc_handle, mtl_buffer *buf,
                     size_t offset, size_t length, uint8_t value)
{
   @autoreleasepool {
      id<MTLBlitCommandEncoder> enc = (id<MTLBlitCommandEncoder>)blit_enc_handle;
      id<MTLBuffer> b = (id<MTLBuffer>)buf;
      [enc fillBuffer:b range:NSMakeRange(offset, length) value:value];
   }
}
