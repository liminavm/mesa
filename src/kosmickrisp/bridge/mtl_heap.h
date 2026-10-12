/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_HEAP_H
#define MTL_HEAP_H 1

#include "mtl_types.h"

#include <inttypes.h>

/* TODO_KOSMICKRISP We should move this struct to the bridge side. */
struct kk_image_layout;

/* Creation */
mtl_heap *mtl_new_heap(mtl_device *device, uint64_t size,
                       enum mtl_resource_options resource_options);

/* Utils */
uint64_t mtl_heap_get_size(mtl_heap *heap);

/* Allocation from heap */
mtl_buffer *mtl_new_buffer_with_length(mtl_heap *heap, uint64_t size_B,
                                       uint64_t offset_B);
mtl_texture *mtl_new_texture_with_descriptor(
   mtl_heap *heap, const struct kk_image_layout *layout, uint64_t offset);

void mtl_heap_set_label(mtl_heap *heap, const char *label);

/* Maps the first `size_B` bytes of a heap's memory at `addr` (anywhere when NULL) and returns the
 * mapping, or NULL. Only for bridges whose heap memory is a file the process can map again: the
 * Metal bridge remaps Metal's own mapping in kk_bo.c instead. */
void *mtl_heap_map_at(mtl_heap *heap, void *addr, uint64_t size_B);

/* Heaps other processes share as dma-bufs. Only for bridges on a DRM device; the Metal bridge
 * does not define them. */
/* A heap of exactly `size` bytes (rounded up by the device) that can be exported. */
mtl_heap *mtl_new_heap_exportable(mtl_device *device, uint64_t size);
/* A new dma-buf fd for an exportable or imported heap, or -1. */
int mtl_heap_export_fd(mtl_heap *heap);
/* The heap another heap exported as dma-buf `fd`, or NULL. The fd stays the caller's. */
mtl_heap *mtl_new_heap_from_fd(mtl_device *device, int fd);
/* A texture whose storage is a surface the host makes to share beyond this driver: what another
 * GPU driver in the guest (a compositor's) imports as a plain buffer. Only linear 2D images of one
 * level, one layer and one sample, in a 32-bit BGRA or RGBA format, laid out as `layout` says;
 * NULL for anything else, or when the bridge cannot. */
mtl_texture *mtl_new_texture_shareable(mtl_device *device, const struct kk_image_layout *layout);
/* A new dma-buf fd for a shareable texture, or -1. */
int mtl_texture_export_fd(mtl_texture *texture);
/* The texture over the surface dma-buf `fd` names, whichever driver made it, laid out as `layout`
 * says; NULL when it is no surface or not one `layout` describes. The fd stays the caller's. */
mtl_texture *mtl_new_texture_from_fd(mtl_device *device, int fd,
                                     const struct kk_image_layout *layout);
void mtl_resource_set_label(mtl_resource *resource, const char *label);

#endif /* MTL_HEAP_H */
