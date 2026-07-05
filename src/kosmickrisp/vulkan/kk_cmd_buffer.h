/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_CMD_BUFFER_H
#define KK_CMD_BUFFER_H 1

#include "kk_private.h"

#include "kk_descriptor_set.h"
#include "kk_image.h"
#include "kk_nir_lower_vbo.h"
#include "kk_shader.h"

#include "kosmickrisp/bridge/mtl_types.h"

#include "kosmickrisp/libkk/kk_query.h"

#include "util/u_dynarray.h"

#include "vk_command_buffer.h"

#include <stdio.h>

struct kk_query_pool;

struct kk_root_descriptor_table {
   struct kk_ptr root_buffer;

   union {
      struct {
         /* Vertex input state */
         uint32_t buffer_strides[KK_MAX_VBUFS];
         uint64_t attrib_base[KK_MAX_ATTRIBS];
         uint32_t attrib_clamps[KK_MAX_ATTRIBS];

         float blend_constant[4];
         float clip_z_coeff;

         /* Transform feedback: per-buffer write base for THIS draw,
          * pre-adjusted so the vertex shader stores at
          * xfb_base[b] + vertex_id * stride + output_offset
          * (bound base + current append offset - firstVertex * stride).
          * xfb_active_mask has bit b set when capture to buffer b is on.
          */
         uint64_t xfb_base[4];
         uint32_t xfb_active_mask;
         /* For instanced capture: slot = (instance_id - first_instance) *
          * verts_per_instance + vertex_id. */
         uint32_t xfb_verts_per_instance;
         uint32_t xfb_first_instance;
      } draw;
      struct {
         uint32_t base_group[3];
      } cs;
   };

   /* Client push constants */
   uint8_t push[KK_MAX_PUSH_SIZE];

   /* Descriptor set base addresses */
   uint64_t sets[KK_MAX_SETS];

   /* Dynamic buffer bindings */
   struct kk_buffer_address dynamic_buffers[KK_MAX_DYNAMIC_BUFFERS];

   /* Start index in dynamic_buffers where each set starts */
   uint8_t set_dynamic_buffer_start[KK_MAX_SETS];
};

struct kk_descriptor_state {
   bool root_dirty;
   struct kk_root_descriptor_table root;

   uint32_t set_sizes[KK_MAX_SETS];
   struct kk_descriptor_set *sets[KK_MAX_SETS];

   uint32_t push_dirty;
   struct kk_push_descriptor_set *push[KK_MAX_SETS];
};

struct kk_per_draw_data {
   uint32_t draw_id;
   uint32_t index_size;
   /* Mask of outputs flowing VS->TCS, VS->GS, or TES->GS . */
   uint64_t vertex_outputs;

   /* Address of vertex param buffer if geom/tess is used, else 0 */
   uint64_t vertex_params;

   /* Address of tessellation param buffer if tessellation used, else 0 */
   uint64_t tess_params;

   uint64_t base_vertex_addr;
   uint64_t base_instance_addr;
};

struct kk_attachment {
   VkRenderingAttachmentFlagsKHR flags;
   VkFormat vk_format;
   struct kk_image_view *iview;

   VkResolveModeFlagBits resolve_mode;
   struct kk_image_view *resolve_iview;

   /* Needed to track the value of storeOp in case we need to copy images for
    * the DRM_FORMAT_MOD_LINEAR case */
   VkAttachmentStoreOp store_op;
};

struct kk_rendering_state {
   VkRenderingFlagBits flags;

   VkRect2D area;
   uint32_t layer_count;
   uint32_t view_mask;
   uint32_t samples;

   uint32_t color_att_count;
   struct kk_attachment color_att[KK_MAX_RTS];
   struct kk_attachment depth_att;
   struct kk_attachment stencil_att;
   struct kk_attachment fsr_att;

   bool ms_bresenham_lines;
   bool sample_locations_enable;
   uint32_t sample_locations_count;
   VkSampleLocationEXT sample_locations[KK_MAX_SAMPLES];
};

/* Dirty tracking bits for state not tracked by vk_dynamic_graphics_state or
 * shaders_dirty.
 */
enum kk_dirty {
   KK_DIRTY_VB = BITFIELD_BIT(0),
   KK_DIRTY_OCCLUSION = BITFIELD_BIT(1),
};

struct kk_graphics_state {
   struct kk_rendering_state render;
   struct kk_descriptor_state descriptors;
   struct kk_per_draw_data per_draw_data;

   mtl_depth_stencil_state *depth_stencil_state;
   mtl_render_pass_descriptor *render_pass_descriptor;
   bool is_depth_stencil_dynamic;
   bool is_cull_front_and_back;
   bool need_to_start_render_pass;

   enum kk_dirty dirty;
   uint32_t pipeline_sample_count;

   struct {
      enum mtl_visibility_result_mode mode;

      /* If enabled, index of the current occlusion query in the occlusion heap.
       * There can only be one active at a time (hardware constraint).
       */
      uint16_t index;
   } occlusion;

   /* Index buffer */
   struct {
      mtl_buffer *handle;
      uint64_t buffer_size;
      uint64_t range;
      uint64_t offset;
      uint8_t bytes_per_index;
   } index;

   /* Vertex buffers */
   struct {
      struct kk_addr_range addr_range[KK_MAX_VBUFS];
      mtl_buffer *handles[KK_MAX_VBUFS];
   } vb;

   /* limina: per-attribute format block size, cached when VI dirties so the
    * per-VB-bind clamp recompute (per draw on zink streams) skips the
    * util_format_description chain. */
   uint8_t attrib_elsize_B[KK_MAX_ATTRIBS];

   /* limina: per-encoder binding cache for the hot per-draw buffer binds
    * (root table at index 0, per-draw data at index 2). The bound buffers
    * live in the upload pool, so consecutive draws rebind the SAME MTLBuffer
    * at a new offset — Metal's setBufferOffset skips the residency/binding-
    * table work of setBuffer, and a fully unchanged bind is skipped outright.
    * Zeroed when a new render encoder is created (fresh binding tables). */
   struct kk_bound_buf {
      mtl_buffer *buf;
      uint32_t offset;
   } bind_cache_v0, bind_cache_f0, bind_cache_v2, bind_cache_f2;

   /* limina: last-uploaded per-draw data + its pool location; non-tess draws
    * usually repeat the same content (draw_id varies only across multidraw),
    * so the per-draw pool upload can be skipped (pool data is immutable).
    * .gpu == 0 = invalid (cmd->state is zeroed at Begin). */
   struct kk_per_draw_data last_per_draw_data;
   struct kk_ptr per_draw_gpu;

   /* Tessellation state */
   struct {
      /* Grid buffer for when the draw is indirect */
      struct kk_ptr indirect_ptr;
      mtl_buffer *out_draws_buffer;
      uint64_t out_draws_offset;
      struct kk_tess_info info;
      enum mesa_prim prim;
   } tess;

   /* Transform feedback state (CPU-tracked; command replay is sequential).
    * Capture is lowered to vertex-shader global stores, so it is correct
    * for non-indexed list topologies — exactly the surface GLES3 permits
    * while transform feedback is active.
    */
   struct {
      bool enabled;
      struct {
         uint64_t gpu_base; /* bound buffer GPU address (0 = unbound) */
         uint64_t size;
         mtl_buffer *counter_handle; /* counter buffer from Begin, for End */
         uint64_t counter_offset;
         uint64_t offset_B; /* current append offset within the buffer */
      } buf[4];

      /* Query accumulation while queries are active (CPU-side; counts are
       * known at replay time for direct draws). */
      struct kk_query_pool *pg_pool; /* PRIMITIVES_GENERATED / pipeline stats */
      uint32_t pg_query;
      uint64_t pg_count;
      struct kk_query_pool *tf_pool; /* TRANSFORM_FEEDBACK_STREAM */
      uint32_t tf_query;
      uint64_t tf_written, tf_needed;
      bool warned_indirect;
   } xfb;

   /* Needed by vk_command_buffer::dynamic_graphics_state */
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
};

struct kk_compute_state {
   struct kk_descriptor_state descriptors;
};

struct kk_conditional_rendering_state {
   uint64_t address;
   bool inverted;
   bool enabled;
};

struct kk_encoder;

struct kk_uploader {
   /** List of kk_cmd_bo */
   struct list_head bos;

   /* Current addresses */
   struct kk_bo *bo;
   uint32_t offset;
};

struct kk_cmd_buffer {
   struct vk_command_buffer vk;

   struct kk_encoder *encoder;
   void *drawable;

   struct {
      struct kk_graphics_state gfx;
      struct kk_compute_state cs;
      struct kk_conditional_rendering_state cond_render;
      struct kk_shader *shaders[MESA_SHADER_STAGES];
      /* Only tracks graphics shaders since compute is always bound for now. */
      uint32_t dirty_shaders;
   } state;

   struct kk_uploader uploader;

   /* Owned large BOs */
   struct util_dynarray large_bos;

   /* Does the command buffer use the geometry heap? */
   bool uses_heap;

   /* Inside a vkCmdBeginRendering/EndRendering pair (replay order). Timestamp
    * writes here are deferred to pass end — we can only sample the GPU clock at
    * encoder boundaries, and ending a render encoder mid-pass would re-run its
    * loadOps. */
   bool in_render_pass;
};

VK_DEFINE_HANDLE_CASTS(kk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops kk_cmd_buffer_ops;

static inline struct kk_device *
kk_cmd_buffer_device(struct kk_cmd_buffer *cmd)
{
   return (struct kk_device *)cmd->vk.base.device;
}

static inline struct kk_cmd_pool *
kk_cmd_buffer_pool(struct kk_cmd_buffer *cmd)
{
   return (struct kk_cmd_pool *)cmd->vk.pool;
}

static inline struct kk_descriptor_state *
kk_get_descriptors_state(struct kk_cmd_buffer *cmd,
                         VkPipelineBindPoint bind_point)
{
   switch (bind_point) {
   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      return &cmd->state.gfx.descriptors;
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmd->state.cs.descriptors;
   default:
      UNREACHABLE("Unhandled bind point");
   }
};

void kk_cmd_release_resources(struct kk_device *dev, struct kk_cmd_buffer *cmd);

static void
kk_cmd_buffer_dirty_all_gfx(struct kk_cmd_buffer *cmd)
{
   /* Ensure we flush all graphics state */
   vk_dynamic_graphics_state_dirty_all(&cmd->vk.dynamic_graphics_state);
   cmd->state.dirty_shaders = ~0u;
   cmd->state.gfx.dirty = ~0u;
   cmd->state.gfx.descriptors.root_dirty = true;

   /* We just flushed out the heap use. If we want to use it again, we'll need
    * to queue a free for it again.
    */
   cmd->uses_heap = false;
}

void kk_cmd_release_dynamic_ds_state(struct kk_cmd_buffer *cmd);

mtl_depth_stencil_state *
kk_compile_depth_stencil_state(struct kk_device *device,
                               const struct vk_depth_stencil_state *ds,
                               bool has_depth, bool has_stencil);

void kk_meta_resolve_rendering(struct kk_cmd_buffer *cmd,
                               const VkRenderingInfo *pRenderingInfo);

void kk_cmd_buffer_write_descriptor_buffer(struct kk_cmd_buffer *cmd,
                                           struct kk_descriptor_state *desc,
                                           size_t size, size_t offset);

struct kk_ptr kk_pool_alloc(struct kk_cmd_buffer *cmd, uint32_t size,
                            uint32_t alignment);

struct kk_ptr kk_pool_upload(struct kk_cmd_buffer *cmd, const void *data,
                             uint32_t size, uint32_t alignment);

uint64_t kk_upload_descriptor_root(struct kk_cmd_buffer *cmd,
                                   VkPipelineBindPoint bind_point);

void kk_cmd_buffer_flush_push_descriptors(struct kk_cmd_buffer *cmd,
                                          struct kk_descriptor_state *desc);

enum kk_grid_mode {
   KK_GRID_DIRECT = 0u,
   KK_GRID_INDIRECT,
};
struct kk_grid {
   enum kk_grid_mode mode;
   union {
      struct {
         uint32_t offset;
         mtl_buffer *indirect;
      };
      struct mtl_size size;
   };
};

static struct kk_grid
kk_grid_3d(uint32_t x, uint32_t y, uint32_t z)
{
   return (struct kk_grid){
      .mode = KK_GRID_DIRECT,
      .size = {x, y, z},
   };
}

static struct kk_grid
kk_grid_2d(uint32_t x, uint32_t y)
{
   return kk_grid_3d(x, y, 1u);
}

static struct kk_grid
kk_grid_1d(uint32_t x)
{
   return kk_grid_3d(x, 1u, 1u);
}

static struct kk_grid
kk_grid_indirect(mtl_buffer *indirect, uint32_t offset)
{
   return (struct kk_grid){
      .mode = KK_GRID_INDIRECT,
      .indirect = indirect,
      .offset = offset,
   };
}

static bool
kk_grid_is_indirect(struct kk_grid grid)
{
   return grid.mode == KK_GRID_INDIRECT;
}

void kk_dispatch_precomp(struct kk_cmd_buffer *cmd, struct kk_grid grid,
                         bool pre_gfx, enum libkk_program idx, void *data,
                         size_t data_size);

#define MESA_DISPATCH_PRECOMP kk_dispatch_precomp

void kk_cmd_write(struct kk_cmd_buffer *cmd, struct libkk_imm_write write);

#endif
