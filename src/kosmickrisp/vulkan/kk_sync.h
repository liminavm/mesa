/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_SYNC_TYPES_H
#define KK_SYNC_TYPES_H 1

#include "kosmickrisp/bridge/mtl_types.h"

#include "util/detect_os.h"
#include "vk_sync.h"

struct kk_queue;

struct kk_sync_timeline {
   struct vk_sync base;
   mtl_shared_event *mtl_handle;
};

extern const struct vk_sync_type kk_sync_type;

#if !DETECT_OS_APPLE
/* The binary semaphores' sync_file export: a kernel fence for the commit the signal follows,
 * where the generic one waits on the CPU. */
VkResult kk_binary_export_sync_file(struct vk_device *device, struct vk_sync *sync, int *pFd);
VkResult kk_binary_import_sync_file(struct vk_device *device, struct vk_sync *sync, int fd);
#endif

#endif
