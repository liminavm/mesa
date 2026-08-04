/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_TEXTURE_H
#define MTL_TEXTURE_H 1

#include "mtl_types.h"

#include <inttypes.h>

/* TODO_KOSMICKRISP Move this to bridge. */
struct kk_view_layout;

/* Utils*/
uint64_t mtl_texture_get_gpu_resource_id(mtl_texture *texture);

/* Properties of an externally-created texture (VK_EXT_external_memory_metal
 * import). The importer must check these against the VkImageCreateInfo it was
 * handed: Vulkan requires the imported handle to be compatible with the image,
 * and Metal will not tell us later — a mismatched texture just renders wrong. */
struct mtl_texture_props {
   uint64_t width;
   uint64_t height;
   uint32_t pixel_format;
   uint32_t texture_type;
   uint32_t sample_count;
   uint32_t mip_levels;
   uint32_t array_length;
   uint32_t usage;
};
void mtl_texture_get_props(mtl_texture *texture,
                           struct mtl_texture_props *props);

/* Texture view creation */
mtl_texture *mtl_new_texture_view_with(mtl_texture *texture,
                                       const struct kk_view_layout *layout);
mtl_texture *
mtl_new_texture_view_with_no_swizzle(mtl_texture *texture,
                                     const struct kk_view_layout *layout);

void mtl_texture_get_bytes(mtl_texture *texture, void *host_ptr,
                           struct mtl_texture_memory_copy *data);

void mtl_texture_replace_region(mtl_texture *texture, const void *host_ptr,
                                struct mtl_texture_memory_copy *data);

#endif /* MTL_TEXTURE_H */
