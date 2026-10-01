/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_cmd_pool.h"

#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_physical_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "vk_common_entrypoints.h"

#include <stdlib.h>

/* limina: KK_CMD_POOL_BO_MAX (32 BOs = 4 MiB) thrashes on draw-heavy replay —
 * a 10k-draw frame needs ~350 BOs of root/push uploads, so ~300 Metal buffers
 * get created AND destroyed every frame (kk_cmd_bo_create was 8.5% of the
 * submit thread, plus residency-set churn; each kk_bo is a full MTLHeap, so a
 * create/destroy is an IOGPU kernel round trip AT DRAW TIME). Default cap is
 * now 512 (2026-07-28 drawstorm decomposition: with the old 32 default the
 * heap churn was the single largest host cost of a command-heavy frame — the
 * cache only grows from BOs a workload actually returned, so light users
 * retain little). LIMINA_KK_BOCACHE overrides: a numeric value >= 0 is the
 * cap (0 disables caching, for debugging), anything else keeps 512. */
static uint32_t
kk_cmd_pool_bo_cache_cap(void)
{
   static int cap = -1;
   if (cap < 0) {
      const char *env = getenv("LIMINA_KK_BOCACHE");
      cap = 512;
      if (env) {
         char *end = NULL;
         long v = strtol(env, &end, 10);
         if (end != env && *end == '\0' && v >= 0)
            cap = (int)v;
      }
   }
   return (uint32_t)cap;
}

static VkResult
kk_cmd_bo_create(struct kk_cmd_pool *pool, struct kk_cmd_bo **bo_out)
{
   struct kk_device *dev = kk_cmd_pool_device(pool);

   struct kk_cmd_bo *bo = vk_zalloc(&pool->vk.alloc, sizeof(*bo), 8,
                                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (bo == NULL)
      return vk_error(pool, VK_ERROR_OUT_OF_HOST_MEMORY);

   const VkResult result =
      kk_alloc_bo(dev, &pool->vk.base, KK_CMD_BO_SIZE, 0, &bo->bo);
   if (result != VK_SUCCESS) {
      vk_free(&pool->vk.alloc, bo);
      return vk_error(pool, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   *bo_out = bo;
   return VK_SUCCESS;
}

static void
kk_cmd_bo_destroy(struct kk_cmd_pool *pool, struct kk_cmd_bo *bo)
{
   struct kk_device *dev = kk_cmd_pool_device(pool);
   kk_destroy_bo(dev, bo->bo);
   vk_free(&pool->vk.alloc, bo);
}

static void
kk_cmd_pool_destroy_bos(struct kk_cmd_pool *pool)
{
   list_for_each_entry_safe(struct kk_cmd_bo, bo, &pool->free_bos, link)
      kk_cmd_bo_destroy(pool, bo);

   list_inithead(&pool->free_bos);
}

VkResult
kk_cmd_pool_alloc_bo(struct kk_cmd_pool *pool, struct kk_cmd_bo **bo_out)
{
   struct kk_cmd_bo *bo = NULL;
   if (!list_is_empty(&pool->free_bos)) {
      bo = list_first_entry(&pool->free_bos, struct kk_cmd_bo, link);
      pool->num_free_bos--;
   }

   if (bo) {
      list_del(&bo->link);
      *bo_out = bo;
      return VK_SUCCESS;
   }

   return kk_cmd_bo_create(pool, bo_out);
}

void
kk_cmd_pool_free_bo_list(struct kk_cmd_pool *pool, struct list_head *bos)
{
   list_for_each_entry_safe(struct kk_cmd_bo, bo, bos, link) {
      list_del(&bo->link);
      if (pool->num_free_bos > kk_cmd_pool_bo_cache_cap()) {
         kk_cmd_bo_destroy(pool, bo);
      } else {
         list_addtail(&bo->link, &pool->free_bos);
         pool->num_free_bos++;
      }
   }
}

mtl_command_buffer *
kk_cmd_pool_get_cmd_buf(struct kk_cmd_pool *pool)
{
   if (util_dynarray_num_elements(&pool->metal.free_cmd_bufs,
                                  mtl_command_buffer *) > 0u)
      return util_dynarray_pop(&pool->metal.free_cmd_bufs,
                               mtl_command_buffer *);

   struct kk_device *dev = kk_cmd_pool_device(pool);
   return mtl_new_command_buffer(dev->mtl_handle);
}

void
kk_cmd_pool_return_cmd_buf(struct kk_cmd_pool *pool,
                           mtl_command_buffer *cmd_buf)
{
   util_dynarray_append(&pool->metal.free_cmd_bufs, cmd_buf);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateCommandPool(VkDevice _device,
                     const VkCommandPoolCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator,
                     VkCommandPool *pCmdPool)
{
   VK_FROM_HANDLE(kk_device, device, _device);
   struct kk_cmd_pool *pool;

   pool = vk_alloc2(&device->vk.alloc, pAllocator, sizeof(*pool), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (pool == NULL)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      vk_command_pool_init(&device->vk, &pool->vk, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free2(&device->vk.alloc, pAllocator, pool);
      return result;
   }

   pool->metal.free_cmd_bufs = UTIL_DYNARRAY_INIT;

   list_inithead(&pool->free_bos);

   *pCmdPool = kk_cmd_pool_to_handle(pool);

   return VK_SUCCESS;
}

static void
kk_cmd_pool_release_mtl_objects(struct kk_cmd_pool *pool)
{
   util_dynarray_foreach(&pool->metal.free_cmd_bufs, mtl_command_buffer *,
                         cmd_buf) {
      mtl_release(*cmd_buf);
   }
   util_dynarray_clear(&pool->metal.free_cmd_bufs);
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyCommandPool(VkDevice _device, VkCommandPool commandPool,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, device, _device);
   VK_FROM_HANDLE(kk_cmd_pool, pool, commandPool);

   if (!pool)
      return;

   vk_command_pool_finish(&pool->vk);
   kk_cmd_pool_destroy_bos(pool);
   kk_cmd_pool_release_mtl_objects(pool);
   util_dynarray_fini(&pool->metal.free_cmd_bufs);
   vk_free2(&device->vk.alloc, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_ResetCommandPool(VkDevice _device, VkCommandPool commandPool,
                    VkCommandPoolResetFlags flags)
{
   VK_FROM_HANDLE(kk_cmd_pool, pool, commandPool);

   VkResult result = vk_common_ResetCommandPool(_device, commandPool, flags);
   if (result != VK_SUCCESS)
      return result;

   if (flags & VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) {
      kk_cmd_pool_release_mtl_objects(pool);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_TrimCommandPool(VkDevice device, VkCommandPool commandPool,
                   VkCommandPoolTrimFlags flags)
{
   VK_FROM_HANDLE(kk_cmd_pool, pool, commandPool);

   vk_command_pool_trim(&pool->vk, flags);
   kk_cmd_pool_destroy_bos(pool);
}

void
kk_cmd_pool_set_label(struct kk_cmd_pool *pool, const char *label)
{
   list_for_each_entry(struct kk_cmd_bo, bo, &pool->free_bos, link)
      kk_bo_set_label(bo->bo, label);
}
