/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_wsi.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_dispatch_trampolines.h"
#include "kk_entrypoints.h"
#include "kk_format.h"
#include "kk_image.h"
#include "kk_instance.h"
#include "wsi_common.h"
#include "vk_format.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#if !DETECT_OS_APPLE
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

static PFN_vkVoidFunction
kk_instance_get_proc_addr_unchecked(const struct vk_instance *instance,
                                    const char *name)
{
   PFN_vkVoidFunction func;

   if (instance == NULL || name == NULL)
      return NULL;

   func = vk_instance_dispatch_table_get(&instance->dispatch_table, name);
   if (func != NULL)
      return func;

   func = vk_physical_device_dispatch_table_get(&kk_physical_device_trampolines,
                                                name);
   if (func != NULL)
      return func;

   func = vk_device_dispatch_table_get(&kk_device_trampolines, name);
   if (func != NULL)
      return func;

   return NULL;
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
kk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);
   return kk_instance_get_proc_addr_unchecked(pdev->vk.instance, pName);
}

#if !DETECT_OS_APPLE
/* Only formats a host surface takes reach the compositor's driver. */
static bool
kk_wsi_can_share_format(VkPhysicalDevice physicalDevice, VkFormat format)
{
   const struct kk_va_format *va = kk_get_va_format(vk_format_to_pipe_format(format));
   return va && mtl_pixel_format_is_shareable(va->mtl_pixel_format);
}

/* The sysfs device behind DRM node major:minor, which a GPU's primary and render nodes share. */
static bool
kk_drm_node_device(dev_t node, char out[PATH_MAX])
{
   char link[64];
   snprintf(link, sizeof(link), "/sys/dev/char/%u:%u/device", major(node), minor(node));
   return realpath(link, out) != NULL;
}

/* Whether the X server's DRI3 device is the GPU this driver renders on. Mesa's own check only
 * matches PCI devices, and virtio-gpu is not one: answered no, X11 WSI copies every frame into an
 * exported buffer, which reaches the X server as a blob its driver cannot see. */
static bool
kk_wsi_can_present_on_device(VkPhysicalDevice physicalDevice, int fd)
{
   VK_FROM_HANDLE(kk_physical_device, pdev, physicalDevice);
   int64_t maj, min;
   struct stat st;
   if (!mtl_device_get_drm_render_node(pdev->mtl_dev_handle, &maj, &min) || fstat(fd, &st) != 0 ||
       !S_ISCHR(st.st_mode))
      return false;
   char ours[PATH_MAX], theirs[PATH_MAX];
   return kk_drm_node_device(makedev(maj, min), ours) && kk_drm_node_device(st.st_rdev, theirs) &&
          strcmp(ours, theirs) == 0;
}
#endif

static VkResult
kk_bind_drawable_to_vkimage(VkImage vk_image, void *drawable)
{
   VK_FROM_HANDLE(kk_image, image, vk_image);
   mtl_texture *texture = mtl_drawable_get_texture(drawable);

   /* This should only be called for swapchain binding. */
   assert(image->plane_count == 1);
   struct kk_image_plane *plane = &image->planes[0];
   if (plane->mtl_handle)
      mtl_release(plane->mtl_handle);
   if (plane->mtl_handle_array)
      mtl_release(plane->mtl_handle_array);
   plane->mtl_handle = mtl_retain(texture);
   plane->mtl_handle_array = NULL;
   plane->addr = mtl_texture_get_gpu_resource_id(texture);

   return VK_SUCCESS;
}

static void
kk_wsi_cmd_blit_image(VkCommandBuffer vk_cmd, const VkBlitImageInfo2 *blit_info,
                      bool opaque_alpha)
{
   VK_FROM_HANDLE(kk_image, src, blit_info->srcImage);

   /* The blit samples the source through a view the meta path creates for
    * itself, and that view is where the alpha gets dropped. The value is a
    * property of the swapchain, so it never changes for a given image. */
   src->wsi_opaque_alpha = opaque_alpha;

   kk_CmdBlitImage2(vk_cmd, blit_info);
}

static void
kk_encode_drawable_present(VkCommandBuffer vk_cmd, void *drawable)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, vk_cmd);
   mtl_retain(drawable);
   cmd->drawable = drawable;
}

static void *
kk_get_mtl4_command_queue(VkDevice device)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   if (dev->has_queue) {
      return dev->queue.mtl_handle;
   }

   return NULL;
}

static struct vk_queue *
kk_get_blit_queue(VkDevice device)
{
   /* We only have one queue, so just return that one. */
   VK_FROM_HANDLE(kk_device, dev, device);
   if (!dev->has_queue)
      return NULL;

   return &dev->queue.vk;
}

VkResult
kk_init_wsi(struct kk_physical_device *pdev)
{
   /* Off macOS, a swapchain image reaches the compositor as a dma-buf only on a bridge that can
    * make shareable textures; otherwise it is copied to host memory. */
#if DETECT_OS_APPLE
   struct wsi_device_options wsi_options = {.sw_device = false};
#else
   struct wsi_device_options wsi_options = {
      .sw_device = !pdev->shareable_textures,
   };
#endif
   struct wsi_device *wsi = &pdev->wsi_device;
   VkResult result =
      wsi_device_init(wsi, kk_physical_device_to_handle(pdev), kk_wsi_proc_addr,
                      &pdev->vk.instance->alloc,
                      -1,   // No KMS display: not relevant for metal wsi
                      NULL, // Not relevant for metal
                      &wsi_options);
   if (result != VK_SUCCESS)
      return result;

   wsi->metal.bind_drawable_to_vkimage = kk_bind_drawable_to_vkimage;
#if !DETECT_OS_APPLE
   wsi->can_share_format = kk_wsi_can_share_format;
   if (pdev->shareable_textures)
      wsi->can_present_on_device = kk_wsi_can_present_on_device;
#endif
   wsi->metal.encode_drawable_present = kk_encode_drawable_present;
   wsi->metal.cmd_blit_image = kk_wsi_cmd_blit_image;
   wsi->metal.get_mtl4_command_queue = kk_get_mtl4_command_queue;
   wsi->get_blit_queue = kk_get_blit_queue;

   pdev->vk.wsi_device = wsi;

   return result;
}

void
kk_finish_wsi(struct kk_physical_device *pdev)
{
   pdev->vk.wsi_device = NULL;
   wsi_device_finish(&pdev->wsi_device, &pdev->vk.instance->alloc);
}
