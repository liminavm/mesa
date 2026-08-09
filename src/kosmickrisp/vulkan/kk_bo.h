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
#include <stdbool.h>

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

VkResult kk_bo_map_placed(struct kk_device *dev, struct kk_bo *bo, void **addr);
VkResult kk_bo_unmap(struct kk_device *dev, struct kk_bo *bo, void *addr,
                     bool reserved);

/* limina: allocation census (LIMINA_KK_BOCENSUS=<secs>). BOs are counted inside kk_bo.c;
 * textures have to be counted where images acquire and release them. Increment on every site
 * that makes a plane hold an MTLTexture — freshly created OR adopted via mtl_retain — and
 * decrement on release, so the pair stays balanced regardless of which path produced it. */
#include <stdatomic.h>
extern atomic_ullong kk_tex_census_acquire, kk_tex_census_release;
#define KK_TEX_CENSUS_ACQUIRE() atomic_fetch_add(&kk_tex_census_acquire, 1ull)
#define KK_TEX_CENSUS_RELEASE() atomic_fetch_add(&kk_tex_census_release, 1ull)

#endif /* KK_BO_H */
