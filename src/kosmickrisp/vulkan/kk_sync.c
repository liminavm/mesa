/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_sync.h"

#include "kk_device.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

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
   /* The runtime signals non-timeline (binary) syncs with value 0; the
    * signaled state of a binary kk sync is event value 1 (waits map 0→1). */
   if (!(sync->flags & VK_SYNC_IS_TIMELINE) && value == 0)
      value = 1;
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
   /* Binary use of this type (kk_sync_type_binary, threaded submit): the
    * runtime passes wait_value 0 for non-timeline syncs; the signal side
    * always sets 1 (see kk_queue_submit / vk_sync signal mapping). */
   if (!(sync->flags & VK_SYNC_IS_TIMELINE) && wait_value == 0)
      wait_value = 1;

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

static VkResult
kk_timeline_reset(struct vk_device *device, struct vk_sync *sync)
{
   struct kk_sync_timeline *timeline =
      container_of(sync, struct kk_sync_timeline, base);
   mtl_shared_event_set_signaled_value(timeline->mtl_handle, 0);
   return VK_SUCCESS;
}

/* Binary-semaphore payload move (threaded submit consumes the wait payload so
 * the app can immediately re-signal the semaphore): dst takes src's shared
 * event — any GPU-side signal already encoded against it follows the handle —
 * and src gets a fresh, unsignaled event. Mirrors dzn_sync_move (the other
 * shared-fence-backed vk_sync).
 */
static VkResult
kk_timeline_move(struct vk_device *device, struct vk_sync *dst,
                 struct vk_sync *src)
{
   struct kk_device *dev = container_of(device, struct kk_device, vk);
   struct kk_sync_timeline *tdst = container_of(dst, struct kk_sync_timeline, base);
   struct kk_sync_timeline *tsrc = container_of(src, struct kk_sync_timeline, base);

   mtl_shared_event *new_event = mtl_new_shared_event(dev->mtl_handle);
   if (new_event == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   mtl_shared_event_set_signaled_value(new_event, 0);

   mtl_release(tdst->mtl_handle);
   tdst->mtl_handle = tsrc->mtl_handle;
   tsrc->mtl_handle = new_event;
   return VK_SUCCESS;
}

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
   .move = kk_timeline_move,
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

/* sync_file shims mirroring vk_sync_binary's: there is no native sync_file on
 * macOS, so "export" waits for the payload and hands back -1 (the
 * already-signaled sync_file), and import of a real fd is rejected (fd == -1
 * is handled by vk_sync_import_sync_file before reaching here). zink creates
 * its binary semaphores with SYNC_FD export info, so the type must offer
 * these hooks to be selectable at all.
 */
static VkResult
kk_binary_import_sync_file(struct vk_device *device, struct vk_sync *sync,
                           int fd)
{
   return VK_ERROR_INVALID_EXTERNAL_HANDLE;
}

static VkResult
kk_binary_export_sync_file(struct vk_device *device, struct vk_sync *sync,
                           int *pFd)
{
   VkResult result =
      kk_timeline_wait(device, sync, 1, 0, OS_TIMEOUT_INFINITE);
   *pFd = -1;
   return result;
}

/* limina (threaded submit): the same shared-event sync serving BINARY
 * semaphores/fences natively — signaled = value 1, reset = value 0, move =
 * handle swap. Registered ahead of the vk_sync_binary wrapper only when
 * LIMINA_KK_SUBMIT_THREAD is on: the wrapper has no `move`, which forced
 * threaded submission trips over (vk_semaphore.c asserts move for permanent
 * binary payloads), while under threaded submit every binary wait moves the
 * payload into a temporary — so each signal/wait cycle runs on a fresh event
 * and the stale-signaled-value hazard of 0/1 binaries never arises. (In
 * immediate mode the wrapper's monotonic-point scheme remains the right
 * mechanism, which is why this type is not registered then.)
 */
const struct vk_sync_type kk_sync_type_binary = {
   .size = sizeof(struct kk_sync_timeline),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_WAIT_ANY |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = kk_timeline_init,
   .finish = kk_timeline_finish,
   .signal = kk_timeline_signal,
   .get_value = kk_timeline_get_value,
   .reset = kk_timeline_reset,
   .move = kk_timeline_move,
   .wait = kk_timeline_wait,
   .wait_many = NULL,
   .import_opaque_fd = NULL,
   .export_opaque_fd = NULL,
   .import_sync_file = kk_binary_import_sync_file,
   .export_sync_file = kk_binary_export_sync_file,
   .import_win32_handle = NULL,
   .export_win32_handle = NULL,
   .set_win32_export_params = NULL,
};
