/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_buffer.h"

/* TODO_KOSMICKRISP Remove */
#include "kk_image_layout.h"

#include <Metal/MTLBuffer.h>
#include <Metal/MTLTexture.h>

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

uint64_t
mtl_buffer_get_length(mtl_buffer *buffer)
{
   @autoreleasepool {
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      return buf.length;
   }
}

uint64_t
mtl_buffer_get_gpu_address(mtl_buffer *buffer)
{
   @autoreleasepool {
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      return [buf gpuAddress];
   }
}

void *
mtl_get_contents(mtl_buffer *buffer)
{
   @autoreleasepool {
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      return [buf contents];
   }
}

/* TODO_KOSMICKRISP This is a duplicate, but both should be removed once we move kk_image_layout to the bridge. */
static MTLTextureDescriptor *
mtl_new_texture_descriptor(const struct kk_image_layout *layout)
{
   @autoreleasepool {
      MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
      descriptor.textureType = (MTLTextureType)layout->type;
      descriptor.pixelFormat = layout->format.mtl;
      descriptor.width = layout->width_px;
      descriptor.height = layout->height_px;
      descriptor.depth = layout->depth_px;
      descriptor.mipmapLevelCount = layout->levels;
      descriptor.sampleCount = layout->sample_count_sa;
      descriptor.arrayLength = layout->layers;
      descriptor.allowGPUOptimizedContents = layout->optimized_layout;
      descriptor.usage = (MTLTextureUsage)layout->usage;
      /* We don't set the swizzle because Metal complains when the usage has store or render target with swizzle... */
      
      return descriptor;
   }
}

mtl_texture *
mtl_new_texture_with_descriptor_linear(mtl_buffer *buffer,
                                       const struct kk_image_layout *layout,
                                       uint64_t offset)
{
   @autoreleasepool {
      id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
      MTLTextureDescriptor *descriptor = [mtl_new_texture_descriptor(layout) autorelease];
      descriptor.resourceOptions = buf.resourceOptions;
      /* limina: Metal requires buffer-backed (linear) textures to be MTLTextureType2D;
       * vk_image_to_mtl_texture_type promotes single-layer images to 2DArray for input
       * attachments, which trips _mtlValidateStrideTextureParameters. Demote — array
       * views of a 2D texture remain creatable via newTextureView. */
      if (descriptor.textureType == MTLTextureType2DArray && descriptor.arrayLength == 1)
         descriptor.textureType = MTLTextureType2D;
      if (limina_kk_rtlog_cached())
         fprintf(stderr,
                 "[LIMINA-KK-LINTEX] %lux%lu type=%lu(layout %u) usage=0x%lx fmt=%lu stride=%u off=%llu\n",
                 (unsigned long)descriptor.width, (unsigned long)descriptor.height,
                 (unsigned long)descriptor.textureType, layout->type,
                 (unsigned long)descriptor.usage, (unsigned long)descriptor.pixelFormat,
                 layout->linear_stride_B, (unsigned long long)offset);
      id<MTLTexture> texture = [buf newTextureWithDescriptor:descriptor offset:offset bytesPerRow:layout->linear_stride_B];

      return texture;
   }
}

