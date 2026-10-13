/*
 * Copyright © 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 */

/* Whether a DRM device is a limina transport this build speaks to, without creating a context:
 * one ioctl, cheap enough for every entry point to decline on. Header-only so the DRI loader can
 * share it with the bridge. */

#ifndef LMN_PROBE_H
#define LMN_PROBE_H

#include <stdbool.h>
#include <string.h>
#include <xf86drm.h>

#include "limina_gpu_wire.h"
#include "virtgpu_limina_drm.h"

static inline bool
lmn_probe_fd(int fd)
{
   /* The ioctl number is in the driver range: only virtio-gpu gives it this meaning. */
   drmVersionPtr v = drmGetVersion(fd);
   bool virtio = v && v->name && !strcmp(v->name, "virtio_gpu");
   drmFreeVersion(v);
   if (!virtio)
      return false;
   struct drm_virtgpu_limina_info info = {0};
   return drmIoctl(fd, DRM_IOCTL_VIRTGPU_LIMINA_INFO, &info) == 0 &&
          info.wire_version == LMN_WIRE_VERSION;
}

#endif
