/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "poly/tessellator.h"

KERNEL(1024)
libkk_prefix_sum_tess(global struct poly_tess_params *p)
{
   local uint scratch[32];
   poly_prefix_sum(scratch, p->counts, p->nr_patches, 1 /* words */,
                   0 /* word */, 1024);

   /* After prefix summing, we know the total # of indices, so allocate the
    * index buffer now. Elect a thread for the allocation.
    */
   barrier(CLK_LOCAL_MEM_FENCE);
   if (cl_local_id.x != 0)
      return;

   /* The last element of an inclusive prefix sum is the total sum */
   uint total = p->nr_patches > 0 ? p->counts[p->nr_patches - 1] : 0;

   /* Allocate 4-byte indices */
   uint32_t elsize_B = sizeof(uint32_t);
   uint32_t size_B = total * elsize_B;
   uint alloc_B = poly_heap_alloc_offs(p->heap, size_B);
   p->index_buffer = (global uint32_t *)(((uintptr_t)p->heap->base) + alloc_B);

   /* ...and now we can generate the API indexed draw */
   global uint32_t *desc = p->out_draws;

   desc[0] = total;              /* count */
   desc[1] = 1;                  /* instance_count */
   desc[2] = alloc_B / elsize_B; /* start */
   desc[3] = 0;                  /* index_bias */
   desc[4] = 0;                  /* start_instance */
}

/* The threadgroups that cover a grid's threads, after them
 * (KK_GRID_INDIRECT_THREADS). */
static void
libkk_grid_threadgroups(global uint32_t *grid, uint32_t local_size_x)
{
   grid[3] = (grid[0] + local_size_x - 1) / local_size_x;
   grid[4] = grid[1];
   grid[5] = grid[2];
}

/* Grids, vertex buffer and count buffer for a geometry shader draw whose counts
 * live on the GPU. The grids are KK_GRID_INDIRECT_THREADS: the kernels they
 * dispatch bounds-check the partial last threadgroup. */
KERNEL(1)
libkk_gs_setup_indirect(uint64_t index_buffer, constant uint *draw,
                        global struct poly_vertex_params *vp /* output */,
                        global struct poly_geometry_params *p /* output */,
                        global struct poly_heap *heap,
                        uint64_t vs_outputs /* Vertex (TES) output mask */,
                        uint32_t index_size_B /* 0 if no index bffer */,
                        uint32_t index_buffer_range_el,
                        uint32_t prim /* Input primitive type, enum mesa_prim */,
                        int is_prefix_summing, uint max_indices, uint32_t shape,
                        uint32_t vs_local_size, uint32_t gs_local_size)
{
   poly_gs_setup_indirect(index_buffer, draw, vp, p, heap, vs_outputs,
                          index_size_B, index_buffer_range_el, prim,
                          is_prefix_summing, max_indices,
                          (enum poly_gs_shape)shape);

   libkk_grid_threadgroups(vp->grid, vs_local_size);
   libkk_grid_threadgroups(p->grid, gs_local_size);
}

/* One workgroup per count word. */
KERNEL(1024)
libkk_prefix_sum_geom(constant struct poly_geometry_params *p)
{
   local uint scratch[32];
   poly_prefix_sum(scratch, p->count_buffer, p->input_primitives,
                   p->count_buffer_stride / 4, cl_group_id.x, 1024);
}

/* Adds the primitives a geometry shader draw generated and wrote (counts[0],
 * counts[1]) to the accumulators of the queries active around it. */
KERNEL(1)
libkk_gs_accumulate_counts(constant uint32_t *counts,
                           global uint32_t *generated /* nullable */,
                           global uint32_t *written_needed /* nullable */)
{
   if (generated)
      generated[0] += counts[0];

   if (written_needed) {
      written_needed[0] += counts[1];
      written_needed[1] += counts[0];
   }
}

/* Copies count 32-bit words, for transform feedback offsets moving between a
 * counter buffer and the GPU-tracked append offsets. */
KERNEL(1)
libkk_copy_u32(global uint32_t *dst, constant uint32_t *src)
{
   dst[cl_global_id.x] = src[cl_global_id.x];
}

/* The draw a vkCmdDrawIndirectByteCountEXT makes from a counter buffer whose
 * value only the GPU holds: a session that captured from a geometry shader
 * wrote it there. Laid out as a VkDrawIndirectCommand. */
KERNEL(1)
libkk_xfb_byte_count_draw(global uint32_t *out, constant uint32_t *counter,
                          uint32_t counter_offset, uint32_t vertex_stride,
                          uint32_t instance_count, uint32_t first_instance)
{
   uint32_t bytes = *counter;
   out[0] =
      bytes > counter_offset ? (bytes - counter_offset) / vertex_stride : 0;
   out[1] = instance_count;
   out[2] = 0;
   out[3] = first_instance;
}
