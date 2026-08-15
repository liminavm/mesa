/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_texture.h"

/* limina: per-class allocation census (limina_mtl_note_new). */
#include "mtl_bridge.h"

/* TODO_LUNARG Remove */
#include "kk_image_layout.h"

/* TODO_LUNARG Remove */
#include "vulkan/vulkan.h"

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
mtl_texture_get_gpu_resource_id(mtl_texture *texture)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      return (uint64_t)[tex gpuResourceID]._impl;
   }
}

void
mtl_texture_get_props(mtl_texture *texture, struct mtl_texture_props *props)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      props->width = [tex width];
      props->height = [tex height];
      props->pixel_format = (uint32_t)[tex pixelFormat];
      props->texture_type = (uint32_t)[tex textureType];
      props->sample_count = (uint32_t)[tex sampleCount];
      props->mip_levels = (uint32_t)[tex mipmapLevelCount];
      props->array_length = (uint32_t)[tex arrayLength];
      props->usage = (uint32_t)[tex usage];
   }
}

/* TODO_KOSMICKRISP This should be part of the mapping */
static uint32_t
mtl_texture_view_type(uint32_t type, uint8_t sample_count)
{
   switch (type) {
   case VK_IMAGE_VIEW_TYPE_1D:
      return MTLTextureType1D;
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
      return MTLTextureType1DArray;
   case VK_IMAGE_VIEW_TYPE_2D:
      return sample_count > 1u ? MTLTextureType2DMultisample : MTLTextureType2D;;
   case VK_IMAGE_VIEW_TYPE_CUBE:
      return MTLTextureTypeCube;
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return MTLTextureTypeCubeArray;
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
      return sample_count > 1u ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
   case VK_IMAGE_VIEW_TYPE_3D:
      return MTLTextureType3D;
   default:
      assert(false && "Unsupported VkViewType");
      return MTLTextureType1D;
   }
}

static MTLTextureSwizzle
mtl_texture_swizzle(enum pipe_swizzle swizzle)
{
   const MTLTextureSwizzle map[] =
      {
         [PIPE_SWIZZLE_X] = MTLTextureSwizzleRed,
         [PIPE_SWIZZLE_Y] = MTLTextureSwizzleGreen,
         [PIPE_SWIZZLE_Z] = MTLTextureSwizzleBlue,
         [PIPE_SWIZZLE_W] = MTLTextureSwizzleAlpha,
         [PIPE_SWIZZLE_0] = MTLTextureSwizzleZero,
         [PIPE_SWIZZLE_1] = MTLTextureSwizzleOne,
      };

   return map[swizzle];
}

mtl_texture *
mtl_new_texture_view_with(mtl_texture *texture, const struct kk_view_layout *layout)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      MTLTextureType type = mtl_texture_view_type(layout->view_type, layout->sample_count_sa);
      NSRange levels = NSMakeRange(layout->base_level, layout->num_levels);
      NSRange slices = NSMakeRange(layout->base_array_layer, layout->array_len);
      MTLTextureSwizzleChannels swizzle = MTLTextureSwizzleChannelsMake(mtl_texture_swizzle(layout->swizzle.red),
                                                                        mtl_texture_swizzle(layout->swizzle.green),
                                                                        mtl_texture_swizzle(layout->swizzle.blue),
                                                                        mtl_texture_swizzle(layout->swizzle.alpha));
      id<MTLTexture> v = [tex newTextureViewWithPixelFormat:layout->format.mtl textureType:type levels:levels slices:slices swizzle:swizzle];
      if (limina_kk_rtlog_cached() && (!v || tex.buffer))
         fprintf(stderr, "[LIMINA-KK-VIEW] %s parent=%p(linear=%d %lux%lu) type=%lu fmt=%lu -> %p\n",
                 v ? "ok" : "NIL", (void *)tex, tex.buffer ? 1 : 0,
                 (unsigned long)tex.width, (unsigned long)tex.height,
                 (unsigned long)type, (unsigned long)layout->format.mtl, (void *)v);
      return (mtl_texture *)limina_mtl_note_new(v);
   }
}

mtl_texture *
mtl_new_texture_view_with_no_swizzle(mtl_texture *texture, const struct kk_view_layout *layout)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      MTLTextureType type = mtl_texture_view_type(layout->view_type, layout->sample_count_sa);
      NSRange levels = NSMakeRange(layout->base_level, layout->num_levels);
      NSRange slices = NSMakeRange(layout->base_array_layer, layout->array_len);
      id<MTLTexture> v = [tex newTextureViewWithPixelFormat:layout->format.mtl textureType:type levels:levels slices:slices];
      if (limina_kk_rtlog_cached() && (!v || tex.buffer))
         fprintf(stderr, "[LIMINA-KK-VIEW] %s(nosw) parent=%p(linear=%d %lux%lu) type=%lu fmt=%lu -> %p\n",
                 v ? "ok" : "NIL", (void *)tex, tex.buffer ? 1 : 0,
                 (unsigned long)tex.width, (unsigned long)tex.height,
                 (unsigned long)type, (unsigned long)layout->format.mtl, (void *)v);
      return (mtl_texture *)limina_mtl_note_new(v);
   }
}

mtl_texture *
mtl_new_texture_view_with_format(mtl_texture *texture, uint32_t pixel_format)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      id<MTLTexture> v =
         [tex newTextureViewWithPixelFormat:(MTLPixelFormat)pixel_format
                               textureType:tex.textureType
                                    levels:NSMakeRange(0, tex.mipmapLevelCount)
                                    slices:NSMakeRange(0, tex.arrayLength)];
      if (!v)
         fprintf(stderr,
                 "[LIMINA-KK-VIEW] NIL reformat parent=%p(%lux%lu fmt=%lu usage=0x%lx) -> fmt=%lu\n",
                 (void *)tex, (unsigned long)tex.width, (unsigned long)tex.height,
                 (unsigned long)tex.pixelFormat, (unsigned long)tex.usage,
                 (unsigned long)pixel_format);
      return (mtl_texture *)limina_mtl_note_new(v);
   }
}

void
mtl_texture_get_bytes(mtl_texture *texture, void *host_ptr,
                      struct mtl_texture_memory_copy *data)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      MTLRegion region = MTLRegionMake3D(data->image_origin.x, data->image_origin.y, data->image_origin.z,
                                         data->image_size.x, data->image_size.y, data->image_size.z);
      return [tex getBytes:host_ptr
               bytesPerRow:data->buffer_stride_B
             bytesPerImage:data->buffer_2d_image_size_B
                fromRegion:region
               mipmapLevel:data->image_level
                     slice:data->image_slice];
   }
}

void
mtl_texture_replace_region(mtl_texture *texture, const void *host_ptr,
                           struct mtl_texture_memory_copy *data)
{
   @autoreleasepool {
      id<MTLTexture> tex = (id<MTLTexture>)texture;
      MTLRegion region = MTLRegionMake3D(data->image_origin.x, data->image_origin.y, data->image_origin.z,
                                         data->image_size.x, data->image_size.y, data->image_size.z);
      return [tex replaceRegion:region
                    mipmapLevel:data->image_level
                          slice:data->image_slice
                      withBytes:host_ptr
                    bytesPerRow:data->buffer_stride_B
                  bytesPerImage:data->buffer_2d_image_size_B];
   }
}
