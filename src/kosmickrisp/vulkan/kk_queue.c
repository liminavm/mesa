/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_limina_capture.h"
#include "kk_queue.h"
#include "kk_buffer.h"
#include "kk_cmd_buffer.h"
#include "kk_device.h"
#include "kk_physical_device.h"
#include "kk_sync.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_cmd_queue.h"

static void
commit_callback(struct mtl_feedback_data *data)
{
   if (data->error != MTL_COMMAND_QUEUE_ERROR_NONE) {
      struct kk_device *dev = (struct kk_device *)data->user_data;
      vk_device_set_lost(
         &dev->vk, "Command queue error: %s, with message \"%s\"",
         mtl_command_queue_error_to_string(data->error), data->error_message);
   }
}

/* limina: GPU completion for one submitted batch — discharge the allocator borrows it charged,
 * which is what lets an over-budget allocator leave DRAINING and be reset. Metal snapshots the
 * feedback-handler list per commit (measured, mtl4-repro T4), so this fires exactly once. */
struct kk_submit_discharge {
   struct kk_device *dev;
   uint32_t count;
   struct kk_pooled_alloc *allocs[];
};

static void
discharge_callback(struct mtl_feedback_data *data)
{
   struct kk_submit_discharge *d = (struct kk_submit_discharge *)data->user_data;
   for (uint32_t i = 0; i < d->count; ++i)
      kk_alloc_pool_discharge(d->dev, d->allocs[i]);
   free(d);
}

/* limina: fault injection for the discharge-payload allocation.
 *
 * That allocation is a few dozen bytes and effectively never fails, which is exactly why its
 * recovery path carried a use-after-free unnoticed for as long as it did. An untestable error
 * path is an unreviewed one, so make it reachable on demand:
 * LIMINA_KK_FAIL_DISCHARGE_ALLOC=<n> fails the first <n> payload allocations.
 *
 * The counter is deliberately plain: it is a debug knob, and a race can only change how many
 * injections land, never whether the path under test is correct. */
static bool
kk_discharge_alloc_should_fail(void)
{
   static int remaining = -1;
   if (remaining < 0) {
      const char *e = getenv("LIMINA_KK_FAIL_DISCHARGE_ALLOC");
      remaining = (e && *e) ? atoi(e) : 0;
   }
   if (remaining > 0) {
      remaining--;
      fprintf(stderr, "[LIMINA-KK] injecting discharge-payload allocation failure\n");
      return true;
   }
   return false;
}

static void
rerecord_cmd_buffer(struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   kk_reset_cmd_buffer_internal(cmd);

   vk_cmd_queue_execute(&cmd->vk.cmd_queue, kk_cmd_buffer_to_handle(cmd),
                        &dev->vk.dispatch_table);

   cs_end(cmd);
   cs_end(cmd);

   /* Need to ensure the new buffers allocated at record are resident. */
   kk_device_make_resources_resident(dev);
}

static VkResult
kk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct kk_queue *queue = container_of(vk_queue, struct kk_queue, vk);
   struct kk_device *dev = kk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   for (struct vk_sync_wait *wait = submit->waits,
                            *end = submit->waits + submit->wait_count;
        wait != end; ++wait) {
      struct kk_sync_timeline *sync =
         container_of(wait->sync, struct kk_sync_timeline, base);
      mtl_wait_for_event(queue->mtl_handle, sync->mtl_handle, wait->wait_value);
   }

   /* Ensure any changes to residency are propagated before we submit any
    * work. All resources should have been allocated before submission.
    * Otherwise, users are playing with fire. */
   kk_device_make_resources_resident(dev);

   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      struct kk_cmd_buffer *cmd_buffer =
         container_of(submit->command_buffers[i], struct kk_cmd_buffer, vk);

      /* Submitted command buffers require re-recording since Metal does not
       * support multiple submissions. */
      if (cmd_buffer->submitted)
         rerecord_cmd_buffer(cmd_buffer);
      cmd_buffer->submitted = true;

      mtl_command_buffer **cmds =
         util_dynarray_begin(&cmd_buffer->submit_cmd_bufs);
      uint32_t count = util_dynarray_num_elements(&cmd_buffer->submit_cmd_bufs,
                                                  mtl_command_buffer *);

      if (cmd_buffer->drawable) {
         mtl_command_queue_wait_for_drawable(queue->mtl_handle,
                                             cmd_buffer->drawable);
      }

      /* Metal complains with empty submissions. */
      if (count > 0u) {
         /* limina: build the discharge payload BEFORE anything is committed or cleared.
          *
          * The charge for a command buffer must be released at GPU COMPLETION. The old code
          * committed first and, if this allocation failed, left the charges in charged_allocs to
          * be discharged by kk_cmd_release_resources at Vulkan command-buffer reset instead —
          * which rerecord_cmd_buffer (just above) reaches on a SIMULTANEOUS_USE resubmit while
          * the first submission is still executing. No app-side spec violation is needed to get
          * there, so on both tiers a guest could drive pending to 0 with the GPU still reading
          * the heaps: harmless while the pool only ever RESET a drained allocator, a
          * use-after-free once it may DESTROY one.
          *
          * Allocating up-front removes the window rather than narrowing it. If it fails we
          * return before committing, so the GPU never receives this work and discharging later
          * at reset is then exactly right. Nothing between the dynarray clear and the commit can
          * fail, so a charge can never be dropped on the floor either. */
         uint32_t ncharged = util_dynarray_num_elements(
            &cmd_buffer->charged_allocs, kk_pooled_alloc_ptr);
         struct kk_submit_discharge *d = NULL;
         if (ncharged) {
            d = kk_discharge_alloc_should_fail()
                   ? NULL
                   : malloc(sizeof(*d) + ncharged * sizeof(kk_pooled_alloc_ptr));
            if (d == NULL)
               return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
            d->dev = dev;
            d->count = ncharged;
            memcpy(d->allocs, util_dynarray_begin(&cmd_buffer->charged_allocs),
                   ncharged * sizeof(kk_pooled_alloc_ptr));
         }

         mtl_commit_options_add_feedback_handler(queue->commit_options,
                                                 commit_callback, dev);
         if (d) {
            util_dynarray_clear(&cmd_buffer->charged_allocs);
            mtl_commit_options_add_feedback_handler(queue->commit_options,
                                                    discharge_callback, d);
         }

         mtl_command_queue_commit(queue->mtl_handle, cmds, count,
                                  queue->commit_options);

         /* limina: a triggered GPU capture closes here, after the commit, never mid-encode --
          * stopping with an encoded-but-uncommitted command buffer truncates the trace. */
         kk_limina_capture_after_commit();
      }

      if (cmd_buffer->drawable) {
         mtl_command_queue_signal_drawable(queue->mtl_handle,
                                           cmd_buffer->drawable);
         mtl_release(cmd_buffer->drawable);
         cmd_buffer->drawable = NULL;
      }
   }

   for (uint32_t i = 0u; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];
      struct kk_sync_timeline *sync =
         container_of(signal->sync, struct kk_sync_timeline, base);
      mtl_signal_event(queue->mtl_handle, sync->mtl_handle,
                       signal->signal_value);
   }

   return VK_SUCCESS;
}

VkResult
kk_queue_init(struct kk_device *dev, struct kk_queue *queue,
              const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   const VkDeviceQueueGlobalPriorityCreateInfo *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO);
   const VkQueueGlobalPriority global_priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM;

   /* From the Vulkan 1.3.295 spec:
    *
    *    "If the globalPriorityQuery feature is enabled and the requested
    *    global priority is not reported via
    *    VkQueueFamilyGlobalPriorityPropertiesKHR, the driver implementation
    *    must fail the queue creation. In this scenario,
    *    VK_ERROR_INITIALIZATION_FAILED is returned."
    */
   if (dev->vk.enabled_features.globalPriorityQuery &&
       global_priority != VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_INITIALIZATION_FAILED;

   if (global_priority > VK_QUEUE_GLOBAL_PRIORITY_MEDIUM)
      return VK_ERROR_NOT_PERMITTED;

   VkResult result;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS)
      return result;

   queue->mtl_handle = mtl_new_command_queue(dev->mtl_handle);
   mtl_command_queue_add_residency_set(queue->mtl_handle,
                                       dev->residency_set.handle);
   queue->commit_options = mtl_new_commit_options();

   queue->vk.driver_submit = kk_queue_submit;

   return VK_SUCCESS;
}

void
kk_queue_finish(struct kk_device *dev, struct kk_queue *queue)
{
   mtl_command_queue_remove_residency_set(queue->mtl_handle,
                                          dev->residency_set.handle);
   mtl_release(queue->commit_options);
   mtl_release(queue->mtl_handle);
   vk_queue_finish(&queue->vk);
}
