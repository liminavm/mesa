/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef MTL_COMMAND_BUFFER_H
#define MTL_COMMAND_BUFFER_H 1

#include "mtl_types.h"

#include <stdint.h>

void mtl_command_allocator_reset(mtl_command_allocator *allocator);

/* Size of the allocator's internal encoding heaps, per Apple's -[MTL4CommandAllocator
 * allocatedSize]. See the LIMINA_KK_ALLOC_STATS block in mtl_command_buffer.m. */
uint64_t mtl_command_allocator_allocated_size(mtl_command_allocator *allocator);

void mtl_begin_command_buffer(mtl_command_buffer *command_buffer,
                              mtl_command_allocator *allocator);
void mtl_end_command_buffer(mtl_command_buffer *command_buffer);

void mtl_command_resolve_counter_heap(mtl_command_buffer *command_buffer,
                                      mtl_counter_heap *heap,
                                      uint32_t first_index, uint32_t count,
                                      uint64_t dst_addr);

#endif /* MTL_COMMAND_BUFFER_H */
