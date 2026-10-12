/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_sync.h"

#include "kk_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "vk_sync_binary.h"

#if !DETECT_OS_APPLE
#include <errno.h>
#include <poll.h>
#endif

static VkResult
kk_timeline_init(struct vk_device *device, struct vk_sync *sync,
                 uint64_t initial_value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);

   struct kk_device *dev = container_of(device, struct kk_device, vk);
   timeline->mtl_handle = mtl_new_shared_event(dev->mtl_handle);
   mtl_shared_event_set_signaled_value(timeline->mtl_handle, initial_value);

   return VK_SUCCESS;
}

static void
kk_timeline_finish(struct vk_device *device, struct vk_sync *sync)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   mtl_release(timeline->mtl_handle);
}

static VkResult
kk_timeline_signal(struct vk_device *device, struct vk_sync *sync,
                   uint64_t value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   mtl_shared_event_set_signaled_value(timeline->mtl_handle, value);
   return VK_SUCCESS;
}

static VkResult
kk_timeline_get_value(struct vk_device *device, struct vk_sync *sync,
                      uint64_t *value)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   *value = mtl_shared_event_get_signaled_value(timeline->mtl_handle);
   return VK_SUCCESS;
}

static VkResult
kk_timeline_wait(struct vk_device *device, struct vk_sync *sync,
                 uint64_t wait_value, enum vk_sync_wait_flags wait_flags,
                 uint64_t abs_timeout_ns)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);

   /* abs_timeout_ns is the point in time when we should stop waiting, not the
    * absolute time in ns. Therefore, we need to compute the delta from now to
    * when we should stop waiting and convert to ms for Metal to be happy
    * (Similar to what dzn does).
    */
   uint64_t timeout_ms = 0u;
   if (abs_timeout_ns == OS_TIMEOUT_INFINITE) {
      timeout_ms = OS_TIMEOUT_INFINITE;
   } else {
      uint64_t cur_time = os_time_get_nano();
      uint64_t rel_timeout_ns =
         abs_timeout_ns > cur_time ? abs_timeout_ns - cur_time : 0;

      timeout_ms =
         (rel_timeout_ns / 1000000) + (rel_timeout_ns % 1000000 ? 1 : 0);
   }
   int completed = mtl_shared_event_wait_until_signaled_value(
      timeline->mtl_handle, wait_value, timeout_ms);

   return completed != 0 ? VK_SUCCESS : VK_TIMEOUT;
}

#if !DETECT_OS_APPLE
VkResult
kk_binary_export_sync_file(struct vk_device *device, struct vk_sync *sync, int *pFd)
{
   struct vk_sync_binary *binary = container_of(sync, struct vk_sync_binary, sync);
   struct kk_sync_timeline *timeline =
      container_of(&binary->timeline, struct kk_sync_timeline, base);

   /* The runtime has waited for the signal to be submitted (WAIT_PENDING). */
   int fd = mtl_shared_event_export_sync_file(timeline->mtl_handle, binary->next_point);
   if (fd >= 0) {
      *pFd = fd;
      return VK_SUCCESS;
   }
   VkResult result = vk_sync_wait(device, &binary->timeline, binary->next_point, 0,
                                  OS_TIMEOUT_INFINITE);
   /* Reached now, so the bridge's answer is a signaled fence: a real fd, which consumers like the
    * window system pass on to a dma-buf, where -1 is refused. */
   *pFd = result == VK_SUCCESS
             ? mtl_shared_event_export_sync_file(timeline->mtl_handle, binary->next_point)
             : -1;
   return result;
}

/* A queue cannot wait on a guest fence yet, so the fence is waited for here, on the CPU, and the
 * binary is signaled once it has passed. */
VkResult
kk_binary_import_sync_file(struct vk_device *device, struct vk_sync *sync, int fd)
{
   struct vk_sync_binary *binary = container_of(sync, struct vk_sync_binary, sync);
   struct pollfd p = {.fd = fd, .events = POLLIN};
   int r;
   do {
      r = poll(&p, 1, -1);
   } while (r < 0 && (errno == EINTR || errno == EAGAIN));
   if (r < 0 || (p.revents & (POLLERR | POLLNVAL)))
      return vk_error(device, VK_ERROR_INVALID_EXTERNAL_HANDLE);
   binary->next_point++;
   return vk_sync_signal(device, &binary->timeline, binary->next_point);
}
#endif

const struct vk_sync_type kk_sync_type = {
   .size = sizeof(struct kk_sync_timeline),
   .features = VK_SYNC_FEATURE_TIMELINE | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_PENDING |
               VK_SYNC_FEATURE_WAIT_BEFORE_SIGNAL,
   .init = kk_timeline_init,
   .finish = kk_timeline_finish,
   .signal = kk_timeline_signal,
   .get_value = kk_timeline_get_value,
   .reset = NULL,
   .move = NULL,
   .wait = kk_timeline_wait,
   .wait_many = NULL,
   .import_opaque_fd = NULL,
   .export_opaque_fd = NULL,
   .import_sync_file = NULL,
   .export_sync_file = NULL,
   .import_win32_handle = NULL,
   .export_win32_handle = NULL,
   .set_win32_export_params = NULL,
};
