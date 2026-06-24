/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_buffer.h"

#include <Metal/MTLCommandBuffer.h>
#include <QuartzCore/CAMetalLayer.h>

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

void
mtl_encode_signal_event(mtl_command_buffer *cmd_buf_handle,
                        mtl_event *event_handle, uint64_t value)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>)cmd_buf_handle;
      id<MTLEvent> event = (id<MTLEvent>)event_handle;
      [cmd_buf encodeSignalEvent:event value:value];
   }
}

void
mtl_encode_wait_for_event(mtl_command_buffer *cmd_buf_handle,
                          mtl_event *event_handle, uint64_t value)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>)cmd_buf_handle;
      id<MTLEvent> event = (id<MTLEvent>)event_handle;
      [cmd_buf encodeWaitForEvent:event value:value];
   }
}

void
mtl_add_completed_handler(mtl_command_buffer *cmd, void (*callback)(void *data),
                          void *data)
{
   @autoreleasepool {
      id<MTLCommandBuffer> mtl_cmd = (id<MTLCommandBuffer>)cmd;
      [mtl_cmd addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull cmd_buf) {
         @autoreleasepool {
            if (callback)
               callback(data);
         }
      }];
   }
}

void
mtl_command_buffer_commit(mtl_command_buffer *cmd_buffer)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>)cmd_buffer;
      if (limina_kk_rtlog_cached()) {
         [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull done) {
            if (done.error || done.status != MTLCommandBufferStatusCompleted)
               fprintf(stderr, "[LIMINA-KK-CBERR] status=%lu error=%s\n",
                       (unsigned long)done.status,
                       done.error ? [[done.error description] UTF8String] : "nil");
         }];
      }
      [cmd_buf commit];
   }
}

void
mtl_present_drawable(mtl_command_buffer *cmd_buf, void *drawable_ptr)
{
   @autoreleasepool {
      id<MTLCommandBuffer> cmd = (id<MTLCommandBuffer>)cmd_buf;
      id<CAMetalDrawable> drawable = [(id<CAMetalDrawable>)drawable_ptr autorelease];
      [cmd presentDrawable:drawable];
   }
}
