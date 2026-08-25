/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_queue.h"

/* limina: per-class allocation census (limina_mtl_note_new). */
#include "mtl_bridge.h"

#include <Metal/MTLDevice.h>
#include <Metal/MTLCommandQueue.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* LIMINA A/B lever, KK_LIMINA_SERIALIZE=1: chain every command buffer on a queue event so no
 * two can ever execute concurrently or out of order.
 *
 * Metal 4 removed automatic hazard tracking, so nothing orders two command buffers on a queue
 * except the barriers KosmicKrisp encodes by hand. KK submits an upload (pre_gfx, a compute
 * dispatch -- there is no blit encoder) and the draws that sample it (gfx) as SEPARATE command
 * buffers. This lever removes every possible ordering fault between them at once: if the
 * corruption survives full serialisation, no ordering bug in KK can be its cause and the fault
 * is in what is encoded rather than when it runs. It is catastrophically slow by design. */
static bool
limina_serialize_submits(void)
{
   static int on = -1;
   if (on < 0) {
      const char *env = getenv("KK_LIMINA_SERIALIZE");
      on = env && strcmp(env, "0") != 0;
      fprintf(stderr, "[LIMINA] KK submit serialisation %s\n",
              on ? "ON (KK_LIMINA_SERIALIZE) -- one command buffer at a time, GPU-synchronous"
                 : "off (default)");
      fflush(stderr);
   }
   return on;
}

/* MTL4CommitOptions */
mtl_commit_options *
mtl_new_commit_options(void)
{
   @autoreleasepool {
      return (mtl_commit_options *)limina_mtl_note_new([[MTL4CommitOptions new] init]);
   }
}

void
mtl_commit_options_add_feedback_handler(mtl_commit_options *options,
                                        mtl_feedback_handler_callback callback,
                                        void *data)
{
   @autoreleasepool {
      MTL4CommitOptions* opt = (MTL4CommitOptions *)options;
      [opt addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
         @autoreleasepool {
            NSError *error = feedback.error;
            struct mtl_feedback_data feedback_data = (struct mtl_feedback_data){
               .user_data = data,
               .error_message =
                  error ? error.localizedDescription.UTF8String : NULL,
               .gpu_start = feedback.GPUStartTime,
               .gpu_end = feedback.GPUEndTime,
               .error = error
                           ? (enum mtl_command_queue_error)error.code
                           : MTL_COMMAND_QUEUE_ERROR_NONE,
            };
            callback(&feedback_data);
         }
      }];
   }
}

/* MTL4CommandQueue */
mtl_command_queue *
mtl_new_command_queue(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return (mtl_command_queue *)limina_mtl_note_new([dev newMTL4CommandQueue]);
   }
}

void
mtl_command_queue_add_residency_set(mtl_command_queue *cmd_queue,
                                    mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue addResidencySet:s];
   }
}

void
mtl_command_queue_remove_residency_set(mtl_command_queue *cmd_queue,
                                       mtl_residency_set *set)
{
   @autoreleasepool {
      id<MTL4CommandQueue> queue = (id<MTL4CommandQueue>)cmd_queue;
      id<MTLResidencySet> s = (id<MTLResidencySet>)set;
      return [queue removeResidencySet:s];
   }
}

void
mtl_signal_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q signalEvent:e value:value];
   }
}

void
mtl_wait_for_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLEvent> e = (id<MTLEvent>)event;
      [q waitForEvent:e value:value];
   }
}

void
mtl_command_queue_commit(mtl_command_queue *queue,
                         mtl_command_buffer **command_buffers, uint32_t count,
                         mtl_commit_options *options)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTL4CommandBuffer> *cmds = (id<MTL4CommandBuffer> *)command_buffers;
      MTL4CommitOptions *opt = (MTL4CommitOptions *)options;
      /* limina probe: charge these command buffers to their allocators, and discharge them when
       * the GPU says this commit completed. See mtl_command_buffer.m. */
      uint64_t batch = limina_kk_alloc_track_commit((void **)command_buffers, count);
      if (batch) {
         [opt addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
           limina_kk_alloc_track_complete(batch);
         }];
      }
      if (limina_serialize_submits() && count > 0u) {
         /* Chain the command buffers on a queue event so buffer i+1 cannot begin until i has
          * fully completed. `opt` is reused rather than replaced: it carries the charge and
          * discharge feedback handlers the allocator pool depends on, and dropping those would
          * trade a rendering bug for a use-after-free. */
         static id<MTLEvent> ev = nil;
         static MTL4CommitOptions *bare = nil;
         static uint64_t val = 0u;
         if (ev == nil) {
            ev = [[q device] newEvent];
            bare = [[MTL4CommitOptions new] init];
         }
         for (uint32_t i = 0; i < count; ++i) {
            if (val)
               [q waitForEvent:ev value:val];
            /* Only the final buffer carries `opt`, so the charge/discharge handlers fire exactly
             * once for the batch instead of once per buffer. They then land at the completion of
             * the last buffer -- strictly later than before, which is the safe direction. */
            [q commit:&cmds[i] count:1 options:(i + 1u == count ? opt : bare)];
            [q signalEvent:ev value:++val];
         }
      } else {
         [q commit:cmds count:count options:opt];
      }
   }
}

void
mtl_command_queue_wait_for_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q waitForDrawable:d];
   }
}

void
mtl_command_queue_signal_drawable(mtl_command_queue *queue, void *drawable)
{
   @autoreleasepool {
      id<MTL4CommandQueue> q = (id<MTL4CommandQueue>)queue;
      id<MTLDrawable> d = (id<MTLDrawable>)drawable;
      [q signalDrawable:d];
   }
}
