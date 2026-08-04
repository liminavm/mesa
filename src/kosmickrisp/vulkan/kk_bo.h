/*
 * Copyright © 2025 LunarG, Inc
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_BO_H
#define KK_BO_H 1

#include "kosmickrisp/bridge/mtl_types.h"

#include "vulkan/vulkan_core.h"

#include <inttypes.h>

struct kk_device;
struct vk_object_base;

struct kk_bo {
   mtl_heap *mtl_handle;
   mtl_buffer *map;
   /* An imported MTLTexture (VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT).
    * Unlike a heap or buffer import this is not memory we can suballocate from:
    * the texture IS the storage, so mtl_handle/map/gpu/cpu all stay NULL and the
    * single image bound to this memory adopts the texture verbatim. */
   mtl_texture *texture;
   uint64_t size_B;
   uint64_t gpu; // GPU address
   void *cpu;    // CPU address
};

struct kk_ptr {
   void *cpu;
   uint64_t gpu;

   /* Pointer in terms of a Metal buffer and offset */
   mtl_buffer *buffer;
   uint32_t offset;
};

VkResult kk_alloc_bo(struct kk_device *dev, struct vk_object_base *log_obj,
                     uint64_t size_B, uint64_t align_B, struct kk_bo **bo_out);

void kk_destroy_bo(struct kk_device *dev, struct kk_bo *bo);

#endif /* KK_BO_H */
