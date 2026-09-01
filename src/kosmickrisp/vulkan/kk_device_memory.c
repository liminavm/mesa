/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_device_memory.h"

#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_image.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/kk_limina_plane.h"
#include "kosmickrisp/bridge/mtl_bridge.h"

#include "vulkan/vulkan_metal.h"

#include "util/u_atomic.h"
#include "util/u_memory.h"

#include <inttypes.h>
#include <sys/mman.h>

/* Supports mtlheap only */
const VkExternalMemoryProperties kk_mtlheap_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT,
   .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT,
};

/* An MTLTexture handle is a *dedicated* allocation: the texture is the storage,
 * so it can back exactly one image and cannot be suballocated. It is therefore
 * not interchangeable with the heap handle type — keep the compatible set to
 * itself alone. */
const VkExternalMemoryProperties kk_mtltexture_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT |
                             VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
                             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT,
   .compatibleHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT,
};

const VkExternalMemoryProperties kk_host_mem_props = {
   .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
   .exportFromImportedHandleTypes = 0,
   .compatibleHandleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT |
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT,
};

#ifdef VK_USE_PLATFORM_METAL_EXT
VKAPI_ATTR VkResult VKAPI_CALL
kk_GetMemoryMetalHandlePropertiesEXT(
   VkDevice device, VkExternalMemoryHandleTypeFlagBits handleType,
   const void *pHandle,
   VkMemoryMetalHandlePropertiesEXT *pMemoryMetalHandleProperties)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   /* Heaps back all of our own allocations; textures are additionally accepted
    * as dedicated imports (a texture cannot be suballocated, so it backs exactly
    * one image). */
   switch (handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT:
      break;
   default:
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }
   pMemoryMetalHandleProperties->memoryTypeBits =
      BITFIELD_MASK(pdev->mem_type_count);

   return VK_SUCCESS;
}
#endif /* VK_USE_PLATFORM_METAL_EXT */

VKAPI_ATTR VkResult VKAPI_CALL
kk_GetMemoryHostPointerPropertiesEXT(
   VkDevice device, VkExternalMemoryHandleTypeFlagBits handleType,
   UNUSED const void *pHandle,
   VkMemoryHostPointerPropertiesEXT *pMemoryHostPointerProperties)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   struct kk_physical_device *pdev = kk_device_physical(dev);

   switch (handleType) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT:
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT:
      break;
   default:
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   }
   pMemoryHostPointerProperties->memoryTypeBits =
      BITFIELD_MASK(pdev->mem_type_count);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_AllocateMemory(VkDevice device, const VkMemoryAllocateInfo *pAllocateInfo,
                  const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   struct kk_device_memory *mem;
   VkResult result = VK_SUCCESS;
   const VkImportMemoryMetalHandleInfoEXT *metal_info = vk_find_struct_const(
      pAllocateInfo->pNext, IMPORT_MEMORY_METAL_HANDLE_INFO_EXT);
   const VkImportMemoryHostPointerInfoEXT *host_info = vk_find_struct_const(
      pAllocateInfo->pNext, IMPORT_MEMORY_HOST_POINTER_INFO_EXT);

   // TODO_KOSMICKRISP Do the actual memory allocation with alignment requirements
   uint32_t alignment = (1ULL << 12);

   const uint64_t aligned_size =
      align64(pAllocateInfo->allocationSize, alignment);

   mem = vk_device_memory_create(&dev->vk, pAllocateInfo, pAllocator,
                                 sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   bool import_metal = metal_info && metal_info->handleType;
   bool import_host = host_info && host_info->handleType;

   if (import_metal || import_host) {
      /* limina: metal_info is NULL on a host-pointer-only import — don't deref it. */
      const bool import_texture =
         import_metal && metal_info->handleType ==
                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT;
      if (import_metal && !import_texture &&
          metal_info->handleType !=
             VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT) {
         result = vk_errorf(&dev->vk.base, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                            "unsupported Metal handle type 0x%x",
                            metal_info->handleType);
         goto fail_alloc;
      }
      mem->bo = CALLOC_STRUCT(kk_bo);
      if (!mem->bo) {
         result = vk_errorf(&dev->vk.base, VK_ERROR_OUT_OF_DEVICE_MEMORY, "%m");
         goto fail_alloc;
      }

      if (import_texture) {
         /* A texture is the storage, not a pool to carve from: there is no heap
          * and no buffer, so gpu/cpu stay 0 and the image bound to this memory
          * adopts the texture verbatim (kk_image_plane_bind). Anything that
          * tries to treat this memory as a buffer — vkMapMemory, a VkBuffer
          * bind — must fail rather than deref a NULL map. */
         /* limina: the MTLTEXTURE handle also accepts an IOSurfaceRef (CFTypeID
          * dispatch). We build the texture HERE, from the dedicated image's own
          * kk_image_layout, via newTextureWithDescriptor:iosurface:plane: — so
          * the bind-time verbatim-adoption checks pass by construction. This is
          * the import half of the vrend "IOSurface world": zink (under vrend)
          * imports a venus-exported window buffer's IOSurface as a GL texture. */
         if (mtl_handle_is_iosurface(metal_info->handle)) {
            const VkMemoryDedicatedAllocateInfo *ded = vk_find_struct_const(
               pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
            struct kk_image *ded_image =
               (ded && ded->image) ? kk_image_from_handle(ded->image) : NULL;
            if (!ded_image || ded_image->plane_count != 1) {
               result = vk_errorf(&dev->vk.base, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                                  "IOSurface import requires a dedicated "
                                  "single-plane image");
               FREE(mem->bo);
               goto fail_alloc;
            }
            /* Hand-walked rather than vk_find_struct_const: the plane struct is
             * limina-private and has no entry in the generated sType tables the
             * macro resolves against. Absent, the import means plane 0. */
            uint32_t io_plane = 0;
            for (const VkBaseInStructure *s = pAllocateInfo->pNext; s;
                 s = s->pNext) {
               if (s->sType == VK_STRUCTURE_TYPE_IMPORT_IOSURFACE_PLANE_LIMINA) {
                  io_plane = kk_limina_plane_index(
                     ((const VkImportIOSurfacePlaneLIMINA *)s)->plane);
                  break;
               }
            }
            mem->bo->texture = mtl_new_texture_with_descriptor_iosurface(
               dev->mtl_handle, &ded_image->planes[0].layout,
               metal_info->handle, io_plane);
            if (!mem->bo->texture) {
               result = vk_errorf(&dev->vk.base, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                                  "IOSurface texture creation failed");
               FREE(mem->bo);
               goto fail_alloc;
            }
         } else {
            mem->bo->texture = mtl_retain(metal_info->handle);
         }
         mem->bo->size_B = mem->vk.size;
         kk_device_add_texture_to_residency_set(dev, mem->bo->texture);
      } else if (import_metal) {
         /* We only support heaps since that's the backing for all our memory
          * and simplifies implementation */
         mem->bo->mtl_handle = mtl_retain(metal_info->handle);
         mem->bo->map =
            mtl_new_buffer_with_length(mem->bo->mtl_handle, mem->vk.size, 0u);
         mem->bo->size_B = mtl_heap_get_size(mem->bo->mtl_handle);
         kk_device_add_heap_to_residency_set(dev, mem->bo->mtl_handle);
      } else if (import_host) {
         /* We can't create a heap from a host pointer. The imported memory will
          * only be usable for buffers */
         mem->bo->map = mtl_new_buffer_with_bytes_no_copy(
            dev->mtl_handle, host_info->pHostPointer, mem->vk.size);
         /* limina: newBufferWithBytesNoCopy returns nil on a rejected pointer/length;
          * adding nil to the residency set makes the AGX residency commit segfault
          * at the next submit. Fail the allocation loudly instead. */
         if (!mem->bo->map) {
            fprintf(stderr,
                    "[LIMINA-KK-IMPORT] newBufferWithBytesNoCopy FAILED ptr=%p "
                    "size=%llu (page-aligned ptr + page-multiple size required)\n",
                    host_info->pHostPointer, (unsigned long long)mem->vk.size);
            FREE(mem->bo);
            result = vk_errorf(&dev->vk.base, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                               "host pointer import rejected by Metal");
            goto fail_alloc;
         }
         mem->bo->size_B = mtl_buffer_get_length(mem->bo->map);
         kk_device_add_buffer_to_residency_set(dev, mem->bo->map);
      }

      /* A texture import has no buffer to take an address or contents from. */
      if (mem->bo->map) {
         mem->bo->gpu = mtl_buffer_get_gpu_address(mem->bo->map);
         mem->bo->cpu = mtl_get_contents(mem->bo->map);
      }
   } else {
      result =
         kk_alloc_bo(dev, &dev->vk.base, aligned_size, alignment, &mem->bo);
      if (result != VK_SUCCESS)
         goto fail_alloc;
   }

   *pMem = kk_device_memory_to_handle(mem);

   return VK_SUCCESS;

fail_alloc:
   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
kk_FreeMemory(VkDevice device, VkDeviceMemory _mem,
              const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_device_memory, mem, _mem);

   if (!mem)
      return;

   /* According to spec, memory is implicitly unmapped upon free */
   if (mem->map)
      kk_bo_unmap(dev, mem->bo, mem->map, false);

   kk_destroy_bo(dev, mem->bo);

   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_MapMemory2KHR(VkDevice device, const VkMemoryMapInfoKHR *pMemoryMapInfo,
                 void **ppData)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_device_memory, mem, pMemoryMapInfo->memory);
   VkResult result = VK_SUCCESS;

   if (mem == NULL) {
      *ppData = NULL;
      return VK_SUCCESS;
   }

   const VkDeviceSize offset = pMemoryMapInfo->offset;
   const VkDeviceSize size = vk_device_memory_range(
      &mem->vk, pMemoryMapInfo->offset, pMemoryMapInfo->size);

   void *fixed_addr = NULL;
   if (pMemoryMapInfo->flags & VK_MEMORY_MAP_PLACED_BIT_EXT) {
      const VkMemoryMapPlacedInfoEXT *placed_info = vk_find_struct_const(
         pMemoryMapInfo->pNext, MEMORY_MAP_PLACED_INFO_EXT);
      fixed_addr = placed_info->pPlacedAddress;
   }

   /* From the Vulkan spec version 1.0.32 docs for MapMemory:
    *
    *  * If size is not equal to VK_WHOLE_SIZE, size must be greater than 0
    *    assert(size != 0);
    *  * If size is not equal to VK_WHOLE_SIZE, size must be less than or
    *    equal to the size of the memory minus offset
    */
   assert(size > 0);
   assert(offset + size <= mem->bo->size_B);

   if (size != (size_t)size) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "requested size 0x%" PRIx64 " does not fit in %u bits",
                       size, (unsigned)(sizeof(size_t) * 8));
   }

   /* From the Vulkan 1.2.194 spec:
    *
    *    "memory must not be currently host mapped"
    */
   if (mem->map != NULL) {
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                       "Memory object already mapped.");
   }

   void *mapped_addr = fixed_addr;
   result = kk_bo_map_placed(dev, mem->bo, &mapped_addr);
   if (result != VK_SUCCESS)
      return result;

   assert(!fixed_addr || mapped_addr == fixed_addr);
   mem->map = mapped_addr;
   *ppData = mem->map + offset;

   if (getenv("LIMINA_KK_RTLOG"))
      fprintf(stderr, "[LIMINA-KK-MAP] mem=%p size=%llu cpu=%p heap=%d\n", (void *)mem,
              (unsigned long long)mem->vk.size, mem->bo->cpu,
              mem->bo->mtl_handle ? 1 : 0);

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_UnmapMemory2KHR(VkDevice device,
                   const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_device_memory, mem, pMemoryUnmapInfo->memory);
   VkResult result = VK_SUCCESS;

   if (mem == NULL)
      return result;

   bool reserved = pMemoryUnmapInfo->flags & VK_MEMORY_UNMAP_RESERVE_BIT_EXT;

   result = kk_bo_unmap(dev, mem->bo, mem->map, reserved);
   if (result != VK_SUCCESS)
      return result;

   /* According to spec, memory is not unmapped on failure, so we only clear to
    * NULL on success */
   mem->map = NULL;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_FlushMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                           const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_InvalidateMappedMemoryRanges(VkDevice device, uint32_t memoryRangeCount,
                                const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_GetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory _mem,
                             VkDeviceSize *pCommittedMemoryInBytes)
{
   VK_FROM_HANDLE(kk_device_memory, mem, _mem);

   *pCommittedMemoryInBytes = mem->bo->size_B;
}

#ifdef VK_USE_PLATFORM_METAL_EXT
VKAPI_ATTR VkResult VKAPI_CALL
kk_GetMemoryMetalHandleEXT(
   VkDevice device, const VkMemoryGetMetalHandleInfoEXT *pGetMetalHandleInfo,
   void **pHandle)
{
   VK_FROM_HANDLE(kk_device_memory, mem, pGetMetalHandleInfo->memory);

   /* A texture-backed allocation can only ever hand back its texture, and a
    * heap-backed one its heap — asking for the other is an invalid handle, not
    * an assert: the request comes from the application. */
   if (pGetMetalHandleInfo->handleType ==
       VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLTEXTURE_BIT_EXT) {
      if (!mem->bo->texture)
         return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      *pHandle = mem->bo->texture;
      return VK_SUCCESS;
   }

   if (pGetMetalHandleInfo->handleType !=
       VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT)
      return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   /* In some cases, such as host pointer imports, we don't have a MTLHeap */
   if (!mem->bo->mtl_handle)
      return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   /* From the Vulkan spec of vkGetMemoryMetalHandleEXT:
    *
    *    "Unless the app retains the handle object returned by the call,
    *     the lifespan will be the same as the associated VkDeviceMemory"
    */
   *pHandle = mem->bo->mtl_handle;
   return VK_SUCCESS;
}
#endif /* VK_USE_PLATFORM_METAL_EXT */

VKAPI_ATTR uint64_t VKAPI_CALL
kk_GetDeviceMemoryOpaqueCaptureAddress(
   UNUSED VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   VK_FROM_HANDLE(kk_device_memory, mem, pInfo->memory);

   return mem->bo->gpu;
}
