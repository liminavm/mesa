/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_command_buffer.h"

void
mtl_command_allocator_reset(mtl_command_allocator *allocator)
{
}

uint64_t
mtl_command_allocator_allocated_size(mtl_command_allocator *allocator)
{
   return 0u;
}

int
limina_kk_alloc_stats_on(void)
{
   return 0;
}

void *
limina_kk_alloc_track_commit(void **cbs, uint32_t count)
{
   return NULL;
}

void
limina_kk_alloc_track_complete(void *batch)
{
}

void
mtl_begin_command_buffer(mtl_command_buffer *command_buffer,
                         mtl_command_allocator *allocator)
{
}

void
mtl_end_command_buffer(mtl_command_buffer *command_buffer)
{
}

void
mtl_command_resolve_counter_heap(mtl_command_buffer *command_buffer,
                                 mtl_counter_heap *heap, uint32_t first_index,
                                 uint32_t count, uint64_t dst_addr)
{
}
