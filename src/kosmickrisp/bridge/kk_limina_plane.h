/*
 * Copyright 2026 the limina authors
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_LIMINA_PLANE_H
#define KK_LIMINA_PLANE_H

#include <vulkan/vulkan_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * limina: which plane of an imported IOSurface backs this memory.
 *
 * VK_EXT_external_memory_metal imports a surface as a bare
 * VkImportMemoryMetalHandleInfoEXT::handle and has nowhere to say "plane 1" —
 * that is the whole gap this struct fills. Metal's own
 * newTextureWithDescriptor:iosurface:plane: takes the index, so without a
 * carrier every import is plane 0 and a biplanar (NV12) decode target can only
 * ever expose its luma.
 *
 * Chained onto the import info by zink's IOSurface path and read by KK's
 * vkAllocateMemory. Both ends are on this branch, so this is an internal
 * contract, not an ABI: keep the two halves in this one header.
 *
 * Absent, the import means plane 0, which is what every pre-existing caller
 * (the venus scanout and shared-buffer imports) intends.
 *
 * Shaped after VK_EXT_metal_objects' VkImportMetalTextureInfoEXT, which carries
 * exactly this as an aspect bit — that extension has the field and the one we
 * use does not. Matching it keeps an eventual migration mechanical.
 *
 * The sType is private. A validation layer walking the import chain would flag
 * it as unknown; nothing in this stack loads one, and that report would be
 * cosmetic rather than a defect to repair.
 */
#define VK_STRUCTURE_TYPE_IMPORT_IOSURFACE_PLANE_LIMINA \
   ((VkStructureType)1000999000)

typedef struct VkImportIOSurfacePlaneLIMINA {
   VkStructureType sType;
   const void *pNext;
   /* VK_IMAGE_ASPECT_PLANE_0_BIT, _PLANE_1_BIT or _PLANE_2_BIT. */
   VkImageAspectFlagBits plane;
} VkImportIOSurfacePlaneLIMINA;

/* The aspect bit as the index Metal wants. Anything else means plane 0, which
 * keeps an unset or malformed chain on the pre-existing behavior. */
static inline uint32_t
kk_limina_plane_index(VkImageAspectFlagBits aspect)
{
   switch (aspect) {
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
      return 1;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
      return 2;
   default:
      return 0;
   }
}

#ifdef __cplusplus
}
#endif

#endif /* KK_LIMINA_PLANE_H */
