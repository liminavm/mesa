/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 *
 * Recording: encoders append to their command buffer's stream; render-pass and pipeline
 * descriptors are plain structs until the call that consumes them. Also the compiler, since
 * libraries and pipelines are created from descriptors the same way.
 *
 * Argument tables are host objects whose entries KK sets between draws (mtl_set_address). Metal
 * reads an encoder's tables at each draw or dispatch, so the entries changed since the table was
 * last written into this command buffer are written into the stream right before each one; the
 * first use in a recording writes them all, since the host's copy is whatever the last replayed
 * command buffer left in it.
 */
#include "lmn_bridge.h"

#include "kosmickrisp/bridge/vk_to_mtl_map.h"
#include "vulkan/vulkan_core.h"

#include <string.h>

#define LMN_KIND_RP_ATTACHMENT 100u
#define LMN_MAX_TABLE_BINDINGS 4

struct lmn_encoder {
   struct lmn_obj base;
   struct lmn_command_buffer *cb;
   bool render;
   /* A timestamp was written: the encoder updates the timestamp fence as it ends. */
   bool stamped;
   struct {
      struct lmn_argument_table *table;
      uint32_t stages;
   } tables[LMN_MAX_TABLE_BINDINGS];
   unsigned table_count;
};

static struct lmn_encoder *
E(void *encoder)
{
   return LMN_CAST(lmn_encoder, encoder, LMN_KIND_ENCODER);
}

static void
emit(struct lmn_encoder *e, uint16_t op, const void *payload, uint32_t len)
{
   struct lmn_iov iov = {payload, len};
   lmn_cb_append(e->cb, op, &iov, payload ? 1 : 0);
}

static void
flush_table(struct lmn_command_buffer *cb, struct lmn_argument_table *t)
{
   uint32_t mask = t->last_serial == cb->serial ? t->dirty : t->set_mask;
   t->dirty = 0;
   t->last_serial = cb->serial;
   if (!mask)
      return;
   struct lmn_address_binding b[32];
   uint32_t n = 0;
   while (mask) {
      uint32_t i = (uint32_t)__builtin_ctz(mask);
      mask &= mask - 1;
      b[n++] = (struct lmn_address_binding){.index = i, .address = t->addr[i]};
   }
   struct lmn_set_addresses c = {.table = t->base.handle, .count = n};
   struct lmn_iov iov[2] = {{&c, sizeof(c)}, {b, n * (uint32_t)sizeof(b[0])}};
   lmn_cb_append(cb, LMN_OP_SET_ADDRESSES, iov, 2);
}

/* Before a draw or dispatch: the bound tables' changed entries. */
static void
flush_tables(struct lmn_encoder *e)
{
   for (unsigned i = 0; i < e->table_count; i++)
      flush_table(e->cb, e->tables[i].table);
}

static void
bind_table(struct lmn_encoder *e, mtl_argument_table *table, uint32_t stages)
{
   struct lmn_argument_table *t = LMN_CAST(lmn_argument_table, table, LMN_KIND_ARGUMENT_TABLE);
   /* A compute encoder has one table; a render binding takes its stages from the others. */
   unsigned kept = 0;
   for (unsigned i = 0; e->render && i < e->table_count; i++) {
      uint32_t left = e->tables[i].stages & ~stages;
      if (!left)
         continue;
      e->tables[kept] = e->tables[i];
      e->tables[kept].stages = left;
      kept++;
   }
   e->table_count = kept;
   if (e->table_count == LMN_MAX_TABLE_BINDINGS) {
      fprintf(stderr, "[limina-kk] more than %u argument tables bound\n", LMN_MAX_TABLE_BINDINGS);
      abort();
   }
   e->tables[e->table_count].table = t;
   e->tables[e->table_count].stages = stages;
   e->table_count++;
   lmn_cb_ref(e->cb, t);
   flush_table(e->cb, t);
   struct lmn_set_argument_table c = {.table = t->base.handle, .stages = stages};
   emit(e, LMN_OP_SET_ARGUMENT_TABLE, &c, sizeof(c));
}

static struct lmn_encoder *
encoder_new(mtl_command_buffer *command_buffer, bool render)
{
   struct lmn_command_buffer *cb =
      LMN_CAST(lmn_command_buffer, command_buffer, LMN_KIND_COMMAND_BUFFER);
   struct lmn_encoder *e = lmn_obj_new(cb->base.dev, LMN_KIND_ENCODER, sizeof(*e), NULL);
   if (!e) {
      cb->failed = true;
      return NULL;
   }
   e->cb = cb;
   e->render = render;
   return e;
}

/* -- every encoder -- */

/* The device's timestamp fence, waited on (`wait`) or updated by `e` at every stage. */
static void
ts_fence_op(struct lmn_encoder *e, bool wait)
{
   struct lmn_fence_op c = {
      .fence = atomic_load(&e->base.dev->ts_fence)->handle,
      .stages = MTL_STAGE_ALL,
      .wait = wait,
   };
   emit(e, LMN_OP_FENCE, &c, sizeof(c));
}

/* After an encoder's BEGIN: one that follows a resolve reads after it. */
static void
encoder_begun(struct lmn_encoder *e)
{
   if (e->cb->resolved)
      ts_fence_op(e, true);
   e->cb->resolved = false;
}

void
mtl_end_encoding(void *encoder)
{
   struct lmn_encoder *e = E(encoder);
   if (e->stamped)
      ts_fence_op(e, false);
   emit(e, LMN_OP_END_ENCODING, NULL, 0);
}

static void
write_timestamp(struct lmn_encoder *e, uint32_t stage, mtl_counter_heap *heap, uint32_t index)
{
   struct lmn_obj *h = lmn_cast(heap, LMN_KIND_COUNTER_HEAP, __func__);
   lmn_cb_ref(e->cb, h);
   /* Relaxed, as the Metal bridge asks. */
   struct lmn_write_timestamp c = {.heap = h->handle, .index = index, .stage = stage};
   emit(e, LMN_OP_WRITE_TIMESTAMP, &c, sizeof(c));
   e->stamped = true;
}

void
mtl_encoder_note_released(void *encoder)
{
}

uint64_t
mtl_encoder_generation(void *encoder)
{
   return 1;
}

void
mtl_encoder_guard_stats(struct mtl_encoder_guard_stats *out)
{
   memset(out, 0, sizeof(*out));
   out->ctx_mode = "not applicable on limina's Metal transport";
}

void
mtl_encoder_note_allocator(void *encoder, void *allocator, uint32_t resets,
                           uint32_t ops_since_reset)
{
}

void
mtl_encoder_report_incarnation(void *encoder, const char *why)
{
}

static void
barrier(void *encoder, uint32_t kind, uint32_t after, uint32_t before)
{
   /* As the Metal bridge: resource-alias visibility, needed when occlusion queries are used. */
   struct lmn_barrier c = {.kind = kind, .after = after, .before = before, .visibility = 2};
   emit(E(encoder), LMN_OP_BARRIER, &c, sizeof(c));
}

void
mtl_barrier_after_stages(void *encoder, enum mtl_stages after_stages,
                         enum mtl_stages before_queue_stages)
{
   barrier(encoder, LMN_BARRIER_AFTER_STAGES, after_stages, before_queue_stages);
}

void
mtl_barrier_after_encoder_stages(void *encoder, enum mtl_stages after_stages,
                                 enum mtl_stages before_queue_stages)
{
   barrier(encoder, LMN_BARRIER_AFTER_ENCODER_STAGES, after_stages, before_queue_stages);
}

void
mtl_barrier_after_queue_stages(void *encoder, enum mtl_stages after_queue_stages,
                               enum mtl_stages before_stages)
{
   barrier(encoder, LMN_BARRIER_AFTER_QUEUE_STAGES, after_queue_stages, before_stages);
}

static void
fence_op(void *encoder, mtl_fence *fence, enum mtl_stages stages, bool wait)
{
   struct lmn_obj *f = lmn_cast(fence, LMN_KIND_FENCE, __func__);
   lmn_cb_ref(E(encoder)->cb, f);
   struct lmn_fence_op c = {.fence = f->handle, .stages = stages, .wait = wait};
   emit(E(encoder), LMN_OP_FENCE, &c, sizeof(c));
}

void
mtl_update_fence(void *encoder, mtl_fence *fence, enum mtl_stages after_stages)
{
   fence_op(encoder, fence, after_stages, false);
}

void
mtl_wait_for_fence(void *encoder, mtl_fence *fence, enum mtl_stages before_stages)
{
   fence_op(encoder, fence, before_stages, true);
}

void
mtl_encoder_set_label(void *encoder, const char *label)
{
}

void
mtl_encoder_insert_debug_signpost(void *encoder, const char *label)
{
}

void
mtl_encoder_push_debug_group(void *encoder, const char *label)
{
}

void
mtl_encoder_pop_debug_group(void *encoder)
{
}

/* -- compute -- */

mtl_compute_encoder *
mtl_new_compute_command_encoder(mtl_command_buffer *cmd_buffer)
{
   struct lmn_encoder *e = encoder_new(cmd_buffer, false);
   if (e) {
      emit(e, LMN_OP_BEGIN_COMPUTE, NULL, 0);
      encoder_begun(e);
   }
   return e;
}

static struct lmn_buf_ref
buf_ref(mtl_buffer *buffer, uint64_t offset)
{
   struct lmn_buffer *b = LMN_CAST(lmn_buffer, buffer, LMN_KIND_BUFFER);
   return (struct lmn_buf_ref){.arena = b->heap->arena,
                               .offset = b->heap->offset + b->offset + offset};
}

static uint32_t
texture_handle(mtl_texture *texture)
{
   return LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE)->base.handle;
}

/* The handle of a texture the stream names, held by the command buffer. */
static uint32_t
texture_ref(struct lmn_encoder *e, mtl_texture *texture)
{
   lmn_cb_ref(e->cb, texture);
   return texture_handle(texture);
}

void
mtl_copy_from_buffer_to_buffer(mtl_compute_encoder *encoder, mtl_buffer *src_buf,
                               size_t src_offset, mtl_buffer *dst_buf, size_t dst_offset,
                               size_t size)
{
   struct lmn_copy_buffer c = {
      .src = buf_ref(src_buf, src_offset),
      .dst = buf_ref(dst_buf, dst_offset),
      .length = size,
   };
   emit(E(encoder), LMN_OP_COPY_BUFFER, &c, sizeof(c));
}

static void
copy_buffer_texture(mtl_compute_encoder *encoder, struct mtl_buffer_image_copy *d,
                    bool to_texture)
{
   struct lmn_copy_buffer_texture c = {
      .texture = texture_ref(E(encoder), d->image),
      .to_texture = to_texture,
      .level = (uint32_t)d->image_level,
      .slice = (uint32_t)d->image_slice,
      .options = d->options,
      .origin = {(uint32_t)d->image_origin.x, (uint32_t)d->image_origin.y,
                 (uint32_t)d->image_origin.z},
      .size = {(uint32_t)d->image_size.x, (uint32_t)d->image_size.y,
               (uint32_t)d->image_size.z},
      .buffer = buf_ref(d->buffer, d->buffer_offset_B),
      .bytes_per_row = d->buffer_stride_B,
      .bytes_per_image = d->buffer_2d_image_size_B,
   };
   emit(E(encoder), LMN_OP_COPY_BUFFER_TEXTURE, &c, sizeof(c));
}

void
mtl_copy_from_buffer_to_texture(mtl_compute_encoder *encoder, struct mtl_buffer_image_copy *data)
{
   copy_buffer_texture(encoder, data, true);
}

void
mtl_copy_from_texture_to_buffer(mtl_compute_encoder *encoder, struct mtl_buffer_image_copy *data)
{
   copy_buffer_texture(encoder, data, false);
}

void
mtl_copy_from_texture_to_texture(mtl_compute_encoder *encoder, mtl_texture *src_tex_handle,
                                 size_t src_slice, size_t src_level, struct mtl_origin src_origin,
                                 struct mtl_size src_size, mtl_texture *dst_tex_handle,
                                 size_t dst_slice, size_t dst_level, struct mtl_origin dst_origin)
{
   struct lmn_copy_texture c = {
      .src = texture_ref(E(encoder), src_tex_handle),
      .src_slice = (uint32_t)src_slice,
      .src_level = (uint32_t)src_level,
      .dst = texture_ref(E(encoder), dst_tex_handle),
      .dst_slice = (uint32_t)dst_slice,
      .dst_level = (uint32_t)dst_level,
      .src_origin = {(uint32_t)src_origin.x, (uint32_t)src_origin.y, (uint32_t)src_origin.z},
      .src_size = {(uint32_t)src_size.x, (uint32_t)src_size.y, (uint32_t)src_size.z},
      .dst_origin = {(uint32_t)dst_origin.x, (uint32_t)dst_origin.y, (uint32_t)dst_origin.z},
   };
   emit(E(encoder), LMN_OP_COPY_TEXTURE, &c, sizeof(c));
}

void
mtl_compute_set_pipeline_state(mtl_compute_encoder *encoder,
                               mtl_compute_pipeline_state *state_handle)
{
   struct lmn_obj *p = lmn_cast(state_handle, LMN_KIND_COMPUTE_PIPELINE, __func__);
   lmn_cb_ref(E(encoder)->cb, p);
   struct lmn_set_compute_pipeline c = {.handle = p->handle};
   emit(E(encoder), LMN_OP_SET_COMPUTE_PIPELINE, &c, sizeof(c));
}

void
mtl_compute_set_argument_table(mtl_compute_encoder *encoder, mtl_argument_table *table)
{
   bind_table(E(encoder), table, 0);
}

static void
size3(uint32_t out[3], struct mtl_size s)
{
   out[0] = (uint32_t)s.x;
   out[1] = (uint32_t)s.y;
   out[2] = (uint32_t)s.z;
}

void
mtl_dispatch_threads(mtl_compute_encoder *encoder, struct mtl_size grid_size,
                     struct mtl_size local_size)
{
   struct lmn_encoder *e = E(encoder);
   flush_tables(e);
   struct lmn_dispatch_threads c;
   size3(c.grid, grid_size);
   size3(c.threads_per_group, local_size);
   emit(e, LMN_OP_DISPATCH_THREADS, &c, sizeof(c));
}

void
mtl_dispatch_threadgroups_with_indirect_buffer(mtl_compute_encoder *encoder, uint64_t addr,
                                               struct mtl_size local_size)
{
   struct lmn_encoder *e = E(encoder);
   flush_tables(e);
   struct lmn_dispatch_indirect c = {.address = addr};
   size3(c.threads_per_group, local_size);
   emit(e, LMN_OP_DISPATCH_INDIRECT, &c, sizeof(c));
}

void
mtl_compute_write_timestamp(mtl_compute_encoder *encoder, mtl_counter_heap *heap, uint32_t index)
{
   write_timestamp(E(encoder), 0, heap, index);
}

/* -- render pass descriptors -- */

struct lmn_rp_attachment {
   uint32_t kind;
   struct lmn_render_attachment *a;
   /* What a.texture and a.resolve_texture name, for the command buffer to hold. */
   void *texture, *resolve;
};

struct lmn_render_pass_desc {
   struct lmn_obj base;
   struct lmn_begin_render_pass p;
   struct lmn_rp_attachment colors[LMN_MAX_COLOR_ATTACHMENTS], depth, stencil;
};

static void
attachment_defaults(struct lmn_rp_attachment *w, struct lmn_render_attachment *a)
{
   w->kind = LMN_KIND_RP_ATTACHMENT;
   w->a = a;
   /* Metal's defaults. */
   a->load_action = MTL_LOAD_ACTION_DONT_CARE;
   a->store_action = MTL_STORE_ACTION_DONT_CARE;
}

mtl_render_pass_descriptor *
mtl_new_render_pass_descriptor(void)
{
   struct lmn_render_pass_desc *d =
      lmn_obj_new(NULL, LMN_KIND_RENDER_PASS_DESC, sizeof(*d), NULL);
   if (!d)
      return NULL;
   for (unsigned i = 0; i < LMN_MAX_COLOR_ATTACHMENTS; i++) {
      attachment_defaults(&d->colors[i], &d->p.colors[i]);
      d->p.colors[i].clear[3] = 1.0;
   }
   attachment_defaults(&d->depth, &d->p.depth);
   d->p.depth.clear[0] = 1.0;
   attachment_defaults(&d->stencil, &d->p.stencil);
   return d;
}

static struct lmn_render_pass_desc *
RP(mtl_render_pass_descriptor *d)
{
   return LMN_CAST(lmn_render_pass_desc, d, LMN_KIND_RENDER_PASS_DESC);
}

mtl_render_pass_attachment_descriptor *
mtl_render_pass_descriptor_get_color_attachment(mtl_render_pass_descriptor *descriptor,
                                                uint32_t index)
{
   return &RP(descriptor)->colors[index];
}

mtl_render_pass_attachment_descriptor *
mtl_render_pass_descriptor_get_depth_attachment(mtl_render_pass_descriptor *descriptor)
{
   return &RP(descriptor)->depth;
}

mtl_render_pass_attachment_descriptor *
mtl_render_pass_descriptor_get_stencil_attachment(mtl_render_pass_descriptor *descriptor)
{
   return &RP(descriptor)->stencil;
}

static struct lmn_render_attachment *
A(mtl_render_pass_attachment_descriptor *d)
{
   return ((struct lmn_rp_attachment *)lmn_cast(d, LMN_KIND_RP_ATTACHMENT, __func__))->a;
}

void
mtl_render_pass_attachment_descriptor_set_texture(mtl_render_pass_attachment_descriptor *d,
                                                  mtl_texture *texture)
{
   A(d)->texture = texture ? texture_handle(texture) : 0;
   ((struct lmn_rp_attachment *)d)->texture = texture;
}

void
mtl_render_pass_attachment_descriptor_set_level(mtl_render_pass_attachment_descriptor *d,
                                                uint32_t level)
{
   A(d)->level = level;
}

void
mtl_render_pass_attachment_descriptor_set_slice(mtl_render_pass_attachment_descriptor *d,
                                                uint32_t slice)
{
   A(d)->slice = slice;
}

void
mtl_render_pass_attachment_descriptor_set_load_action(mtl_render_pass_attachment_descriptor *d,
                                                      enum mtl_load_action action)
{
   A(d)->load_action = action;
}

void
mtl_render_pass_attachment_descriptor_set_store_action(mtl_render_pass_attachment_descriptor *d,
                                                       enum mtl_store_action action)
{
   A(d)->store_action = action;
}

void
mtl_render_pass_attachment_descriptor_set_clear_color(mtl_render_pass_attachment_descriptor *d,
                                                      struct mtl_clear_color clear_color)
{
   memcpy(A(d)->clear, clear_color.channel, sizeof(A(d)->clear));
}

void
mtl_render_pass_attachment_descriptor_set_clear_depth(mtl_render_pass_attachment_descriptor *d,
                                                      double depth)
{
   A(d)->clear[0] = depth;
}

void
mtl_render_pass_attachment_descriptor_set_clear_stencil(mtl_render_pass_attachment_descriptor *d,
                                                        uint32_t stencil)
{
   A(d)->clear_stencil = stencil;
}

void
mtl_render_pass_attachment_descriptor_set_depth_resolve_filter(
   mtl_render_pass_attachment_descriptor *d, enum mtl_depth_resolve_filter filter)
{
   A(d)->resolve_filter = filter;
}

void
mtl_render_pass_attachment_descriptor_set_stencil_resolve_filter(
   mtl_render_pass_attachment_descriptor *d, enum mtl_stencil_resolve_filter filter)
{
   A(d)->resolve_filter = filter;
}

void
mtl_render_pass_attachment_descriptor_set_resolve_texture(
   mtl_render_pass_attachment_descriptor *d, mtl_texture *resolve_texture)
{
   A(d)->resolve_texture = resolve_texture ? texture_handle(resolve_texture) : 0;
   ((struct lmn_rp_attachment *)d)->resolve = resolve_texture;
}

void
mtl_render_pass_attachment_descriptor_set_resolve_level(mtl_render_pass_attachment_descriptor *d,
                                                        uint32_t resolve_level)
{
   A(d)->resolve_level = resolve_level;
}

void
mtl_render_pass_attachment_descriptor_set_resolve_slice(mtl_render_pass_attachment_descriptor *d,
                                                        uint32_t resolve_slice)
{
   A(d)->resolve_slice = resolve_slice;
}

void
mtl_render_pass_attachment_descriptor_set_resolve_depth_plane(
   mtl_render_pass_attachment_descriptor *d, uint32_t resolve_depth_plane)
{
   A(d)->resolve_depth_plane = resolve_depth_plane;
}

void
mtl_render_pass_descriptor_set_render_target_array_length(mtl_render_pass_descriptor *d,
                                                          uint32_t length)
{
   RP(d)->p.array_length = length;
}

void
mtl_render_pass_descriptor_set_render_target_width(mtl_render_pass_descriptor *d, uint32_t width)
{
   RP(d)->p.width = width;
}

void
mtl_render_pass_descriptor_set_render_target_height(mtl_render_pass_descriptor *d,
                                                    uint32_t height)
{
   RP(d)->p.height = height;
}

void
mtl_render_pass_descriptor_set_default_raster_sample_count(mtl_render_pass_descriptor *d,
                                                           uint32_t sample_count)
{
   RP(d)->p.default_raster_sample_count = sample_count;
}

void
mtl_render_pass_descriptor_set_sample_positions(mtl_render_pass_descriptor *d,
                                                const struct mtl_sample_position *positions,
                                                uint32_t count)
{
   struct lmn_render_pass_desc *rp = RP(d);
   if (count > LMN_MAX_SAMPLE_POSITIONS)
      count = LMN_MAX_SAMPLE_POSITIONS;
   rp->p.sample_position_count = count;
   for (uint32_t i = 0; i < count; i++) {
      rp->p.sample_positions[2 * i] = positions[i].x;
      rp->p.sample_positions[2 * i + 1] = positions[i].y;
   }
}

void
mtl_render_pass_descriptor_set_visibility_buffer(mtl_render_pass_descriptor *d,
                                                 mtl_buffer *visibility_buffer)
{
   RP(d)->p.visibility =
      visibility_buffer ? buf_ref(visibility_buffer, 0) : (struct lmn_buf_ref){0};
}

/* -- render -- */

mtl_render_encoder *
mtl_new_render_command_encoder_with_descriptor(mtl_command_buffer *command_buffer,
                                               mtl_render_pass_descriptor *descriptor)
{
   struct lmn_render_pass_desc *rp = RP(descriptor);
   struct lmn_encoder *e = encoder_new(command_buffer, true);
   if (!e)
      return NULL;
   for (unsigned i = 0; i < LMN_MAX_COLOR_ATTACHMENTS; i++) {
      lmn_cb_ref(e->cb, rp->colors[i].texture);
      lmn_cb_ref(e->cb, rp->colors[i].resolve);
   }
   lmn_cb_ref(e->cb, rp->depth.texture);
   lmn_cb_ref(e->cb, rp->depth.resolve);
   lmn_cb_ref(e->cb, rp->stencil.texture);
   lmn_cb_ref(e->cb, rp->stencil.resolve);
   emit(e, LMN_OP_BEGIN_RENDER_PASS, &rp->p, sizeof(rp->p));
   encoder_begun(e);
   return e;
}

void
mtl_set_viewports(mtl_render_encoder *encoder, struct mtl_viewport *viewports, uint32_t count)
{
   struct lmn_viewport v[16];
   if (count > 16)
      count = 16;
   for (uint32_t i = 0; i < count; i++)
      v[i] = (struct lmn_viewport){
         .x = viewports[i].originX,
         .y = viewports[i].originY,
         .width = viewports[i].width,
         .height = viewports[i].height,
         .znear = viewports[i].znear,
         .zfar = viewports[i].zfar,
      };
   struct lmn_set_viewports c = {.count = count};
   struct lmn_iov iov[2] = {{&c, sizeof(c)}, {v, count * (uint32_t)sizeof(v[0])}};
   lmn_cb_append(E(encoder)->cb, LMN_OP_SET_VIEWPORTS, iov, 2);
}

void
mtl_set_scissor_rects(mtl_render_encoder *encoder, struct mtl_scissor_rect *scissor_rects,
                      uint32_t count)
{
   struct lmn_scissor s[16];
   if (count > 16)
      count = 16;
   for (uint32_t i = 0; i < count; i++)
      s[i] = (struct lmn_scissor){
         .x = (uint32_t)scissor_rects[i].x,
         .y = (uint32_t)scissor_rects[i].y,
         .width = (uint32_t)scissor_rects[i].width,
         .height = (uint32_t)scissor_rects[i].height,
      };
   struct lmn_set_scissors c = {.count = count};
   struct lmn_iov iov[2] = {{&c, sizeof(c)}, {s, count * (uint32_t)sizeof(s[0])}};
   lmn_cb_append(E(encoder)->cb, LMN_OP_SET_SCISSORS, iov, 2);
}

void
mtl_render_set_pipeline_state(mtl_render_encoder *encoder, mtl_render_pipeline_state *pipeline)
{
   struct lmn_obj *p = lmn_cast(pipeline, LMN_KIND_RENDER_PIPELINE, __func__);
   lmn_cb_ref(E(encoder)->cb, p);
   struct lmn_set_render_pipeline c = {.handle = p->handle};
   emit(E(encoder), LMN_OP_SET_RENDER_PIPELINE, &c, sizeof(c));
}

void
mtl_set_depth_stencil_state(mtl_render_encoder *encoder, mtl_depth_stencil_state *state)
{
   struct lmn_obj *ds = state ? lmn_cast(state, LMN_KIND_DEPTH_STENCIL, __func__) : NULL;
   lmn_cb_ref(E(encoder)->cb, ds);
   struct lmn_set_depth_stencil c = {.handle = ds ? ds->handle : 0};
   emit(E(encoder), LMN_OP_SET_DEPTH_STENCIL, &c, sizeof(c));
}

static void
render_state(mtl_render_encoder *encoder, struct lmn_render_state c)
{
   emit(E(encoder), LMN_OP_RENDER_STATE, &c, sizeof(c));
}

void
mtl_set_stencil_references(mtl_render_encoder *encoder, uint32_t front, uint32_t back)
{
   render_state(encoder,
                (struct lmn_render_state){.kind = LMN_RS_STENCIL_REFERENCE, .a = front, .b = back});
}

void
mtl_set_front_face_winding(mtl_render_encoder *encoder, enum mtl_winding winding)
{
   render_state(encoder, (struct lmn_render_state){.kind = LMN_RS_WINDING, .a = winding});
}

void
mtl_set_cull_mode(mtl_render_encoder *encoder, enum mtl_cull_mode mode)
{
   render_state(encoder, (struct lmn_render_state){.kind = LMN_RS_CULL_MODE, .a = mode});
}

void
mtl_set_visibility_result_mode(mtl_render_encoder *encoder, enum mtl_visibility_result_mode mode,
                               size_t offset)
{
   render_state(encoder, (struct lmn_render_state){
                            .kind = LMN_RS_VISIBILITY_RESULT, .a = mode, .offset = offset});
}

void
mtl_set_depth_bias(mtl_render_encoder *encoder, float depth_bias, float slope_scale, float clamp)
{
   render_state(encoder, (struct lmn_render_state){
                            .kind = LMN_RS_DEPTH_BIAS, .f = {depth_bias, slope_scale, clamp}});
}

void
mtl_set_depth_clip_mode(mtl_render_encoder *encoder, enum mtl_depth_clip_mode mode)
{
   render_state(encoder, (struct lmn_render_state){.kind = LMN_RS_DEPTH_CLIP_MODE, .a = mode});
}

void
mtl_set_depth_test_bounds(mtl_render_encoder *encoder, float min, float max)
{
   render_state(encoder,
                (struct lmn_render_state){.kind = LMN_RS_DEPTH_TEST_BOUNDS, .f = {min, max}});
}

void
mtl_set_vertex_amplification_count(mtl_render_encoder *encoder, uint32_t *layer_ids,
                                   uint32_t id_count)
{
   struct lmn_set_amplification c = {.count = id_count};
   struct lmn_iov iov[2] = {{&c, sizeof(c)}, {layer_ids, id_count * 4}};
   lmn_cb_append(E(encoder)->cb, LMN_OP_SET_AMPLIFICATION, iov, 2);
}

void
mtl_render_set_argument_table(mtl_render_encoder *encoder, mtl_argument_table *table,
                              enum mtl_render_stages stages)
{
   bind_table(E(encoder), table, stages);
}

void
mtl_draw_primitives(mtl_render_encoder *encoder, enum mtl_primitive_type primitve_type,
                    uint32_t vertexStart, uint32_t vertexCount, uint32_t instanceCount,
                    uint32_t baseInstance)
{
   struct lmn_encoder *e = E(encoder);
   flush_tables(e);
   struct lmn_draw c = {
      .primitive = primitve_type,
      .vertex_start = vertexStart,
      .vertex_count = vertexCount,
      .instance_count = instanceCount,
      .base_instance = baseInstance,
   };
   emit(e, LMN_OP_DRAW, &c, sizeof(c));
}

void
mtl_draw_indexed_primitives(mtl_render_encoder *encoder, enum mtl_primitive_type primitve_type,
                            uint32_t index_count, enum mtl_index_type index_type,
                            uint64_t index_addr, uint64_t index_buffer_length,
                            uint32_t instance_count, int32_t base_vertex, uint32_t base_instance)
{
   struct lmn_encoder *e = E(encoder);
   flush_tables(e);
   struct lmn_draw_indexed c = {
      .primitive = primitve_type,
      .index_count = index_count,
      .index_type = index_type,
      .instance_count = instance_count,
      .base_vertex = base_vertex,
      .base_instance = base_instance,
      .index_address = index_addr,
      .index_length = index_buffer_length,
   };
   emit(e, LMN_OP_DRAW_INDEXED, &c, sizeof(c));
}

void
mtl_draw_primitives_indirect(mtl_render_encoder *encoder, enum mtl_primitive_type primitve_type,
                             uint64_t addr)
{
   struct lmn_encoder *e = E(encoder);
   flush_tables(e);
   struct lmn_draw_indirect c = {.primitive = primitve_type, .indirect = addr};
   emit(e, LMN_OP_DRAW_INDIRECT, &c, sizeof(c));
}

void
mtl_draw_indexed_primitives_indirect(mtl_render_encoder *encoder,
                                     enum mtl_primitive_type primitve_type,
                                     enum mtl_index_type index_type, uint64_t index_addr,
                                     uint64_t index_buffer_length, uint64_t addr)
{
   struct lmn_encoder *e = E(encoder);
   flush_tables(e);
   struct lmn_draw_indirect c = {
      .primitive = primitve_type,
      .indexed = 1,
      .index_type = index_type,
      .index_address = index_addr,
      .index_length = index_buffer_length,
      .indirect = addr,
   };
   emit(e, LMN_OP_DRAW_INDIRECT, &c, sizeof(c));
}

void
mtl_render_write_timestamp(mtl_render_encoder *encoder, enum mtl_render_stages stage,
                           mtl_counter_heap *heap, uint32_t index)
{
   write_timestamp(E(encoder), stage, heap, index);
}

void
mtl_render_set_color_store_action(mtl_render_encoder *encoder, enum mtl_store_action action,
                                  uint32_t index)
{
   render_state(encoder, (struct lmn_render_state){
                            .kind = LMN_RS_COLOR_STORE_ACTION, .a = action, .b = index});
}

void
mtl_render_set_depth_store_action(mtl_render_encoder *encoder, enum mtl_store_action action)
{
   render_state(encoder,
                (struct lmn_render_state){.kind = LMN_RS_DEPTH_STORE_ACTION, .a = action});
}

void
mtl_render_set_stencil_store_action(mtl_render_encoder *encoder, enum mtl_store_action action)
{
   render_state(encoder,
                (struct lmn_render_state){.kind = LMN_RS_STENCIL_STORE_ACTION, .a = action});
}

/* -- depth-stencil state -- */

struct lmn_stencil_desc_obj {
   struct lmn_obj base;
   struct lmn_stencil_desc s;
};

struct lmn_depth_stencil_desc_obj {
   struct lmn_obj base;
   struct lmn_create_depth_stencil c;
};

mtl_stencil_descriptor *
mtl_new_stencil_descriptor(void)
{
   struct lmn_stencil_desc_obj *d =
      lmn_obj_new(NULL, LMN_KIND_STENCIL_DESC, sizeof(*d), NULL);
   if (d) {
      d->s.compare_function = MTL_COMPARE_FUNCTION_ALWAYS;
      d->s.read_mask = ~0u;
      d->s.write_mask = ~0u;
   }
   return d;
}

static struct lmn_stencil_desc *
SDESC(mtl_stencil_descriptor *d)
{
   return &LMN_CAST(lmn_stencil_desc_obj, d, LMN_KIND_STENCIL_DESC)->s;
}

/* VkStencilOp and MTLStencilOperation list the same operations in the same order. */
static uint32_t
stencil_op(VkStencilOp op)
{
   if ((uint32_t)op > VK_STENCIL_OP_DECREMENT_AND_WRAP) {
      fprintf(stderr, "[limina-kk] unsupported VkStencilOp %d\n", op);
      abort();
   }
   return (uint32_t)op;
}

void
mtl_stencil_descriptor_set_stencil_failure_operation(mtl_stencil_descriptor *d, VkStencilOp op)
{
   SDESC(d)->stencil_failure = stencil_op(op);
}

void
mtl_stencil_descriptor_set_depth_failure_operation(mtl_stencil_descriptor *d, VkStencilOp op)
{
   SDESC(d)->depth_failure = stencil_op(op);
}

void
mtl_stencil_descriptor_set_depth_stencil_pass_operation(mtl_stencil_descriptor *d,
                                                        VkStencilOp op)
{
   SDESC(d)->depth_stencil_pass = stencil_op(op);
}

void
mtl_stencil_descriptor_set_stencil_compare_function(mtl_stencil_descriptor *d, VkCompareOp op)
{
   SDESC(d)->compare_function = vk_compare_op_to_mtl_compare_function(op);
}

void
mtl_stencil_descriptor_set_read_mask(mtl_stencil_descriptor *d, uint32_t mask)
{
   SDESC(d)->read_mask = mask;
}

void
mtl_stencil_descriptor_set_write_mask(mtl_stencil_descriptor *d, uint32_t mask)
{
   SDESC(d)->write_mask = mask;
}

mtl_depth_stencil_descriptor *
mtl_new_depth_stencil_descriptor(void)
{
   struct lmn_depth_stencil_desc_obj *d =
      lmn_obj_new(NULL, LMN_KIND_DEPTH_STENCIL_DESC, sizeof(*d), NULL);
   if (d)
      d->c.depth_compare = MTL_COMPARE_FUNCTION_ALWAYS;
   return d;
}

static struct lmn_create_depth_stencil *
DSDESC(mtl_depth_stencil_descriptor *d)
{
   return &LMN_CAST(lmn_depth_stencil_desc_obj, d, LMN_KIND_DEPTH_STENCIL_DESC)->c;
}

void
mtl_depth_stencil_descriptor_set_depth_compare_function(mtl_depth_stencil_descriptor *d,
                                                        VkCompareOp op)
{
   DSDESC(d)->depth_compare = vk_compare_op_to_mtl_compare_function(op);
}

void
mtl_depth_stencil_descriptor_set_depth_write_enabled(mtl_depth_stencil_descriptor *d,
                                                     bool enable_write)
{
   DSDESC(d)->depth_write = enable_write;
}

void
mtl_depth_stencil_descriptor_set_back_face_stencil(mtl_depth_stencil_descriptor *d,
                                                   mtl_stencil_descriptor *stencil)
{
   DSDESC(d)->has_back = stencil != NULL;
   if (stencil)
      DSDESC(d)->back = *SDESC(stencil);
}

void
mtl_depth_stencil_descriptor_set_front_face_stencil(mtl_depth_stencil_descriptor *d,
                                                    mtl_stencil_descriptor *stencil)
{
   DSDESC(d)->has_front = stencil != NULL;
   if (stencil)
      DSDESC(d)->front = *SDESC(stencil);
}

mtl_depth_stencil_state *
mtl_new_depth_stencil_state(mtl_device *device, mtl_depth_stencil_descriptor *descriptor)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   struct lmn_obj *s =
      lmn_obj_new(dev, LMN_KIND_DEPTH_STENCIL, sizeof(*s), lmn_destroy_host_object);
   if (!s)
      return NULL;
   s->handle = lmn_new_handle(dev);
   struct lmn_create_depth_stencil c = *DSDESC(descriptor);
   c.handle = s->handle;
   lmn_push(dev, LMN_OP_CREATE_DEPTH_STENCIL, &c, sizeof(c));
   return s;
}

/* -- the compiler -- */

mtl_compiler *
mtl_new_compiler(mtl_device *device)
{
   return lmn_obj_new(LMN_CAST(lmn_device, device, LMN_KIND_DEVICE), LMN_KIND_COMPILER,
                      sizeof(struct lmn_obj), NULL);
}

mtl_library *
mtl_new_library(mtl_compiler *compiler, const char *src, enum mtl_language_version msl_version,
                enum mtl_math_mode math_mode, enum mtl_math_floating_point_functions math_fp_fns)
{
   struct lmn_device *dev = lmn_cast(compiler, LMN_KIND_COMPILER, __func__)->dev;
   struct lmn_obj *lib =
      lmn_obj_new(dev, LMN_KIND_LIBRARY, sizeof(*lib), lmn_destroy_host_object);
   struct lmn_slot slot;
   if (!lib)
      return NULL;
   if (!lmn_slot_alloc(dev, &slot)) {
      free(lib);
      return NULL;
   }
   lib->handle = lmn_new_handle(dev);
   size_t len = strlen(src);
   struct lmn_create_library c = {
      .handle = lib->handle,
      .source_len = (uint32_t)len,
      .language_version = msl_version,
      .math_mode = math_mode,
      .math_fp_functions = math_fp_fns,
      .reply = slot.ref,
   };
   struct lmn_iov iov[2] = {{&c, sizeof(c)}, {src, (uint32_t)len}};
   struct lmn_reply r = lmn_call(dev, LMN_OP_CREATE_LIBRARY, iov, 2, &slot);
   lmn_slot_free(dev, &slot);
   if (r.status != 0) {
      fprintf(stderr, "[limina-kk] the host could not compile a library (error %u)\n", r.status);
      free(lib);
      return NULL;
   }
   return lib;
}

struct lmn_function {
   struct lmn_obj base;
   struct lmn_obj *library;
   char *name;
};

static void
function_destroy(struct lmn_obj *obj)
{
   struct lmn_function *f = (struct lmn_function *)obj;
   mtl_release(f->library);
   free(f->name);
}

mtl_function_descriptor *
mtl_new_library_function_descriptor(mtl_library *library, const char *entry_point)
{
   struct lmn_obj *lib = lmn_cast(library, LMN_KIND_LIBRARY, __func__);
   struct lmn_function *f =
      lmn_obj_new(lib->dev, LMN_KIND_FUNCTION, sizeof(*f), function_destroy);
   if (!f)
      return NULL;
   f->library = mtl_retain(lib);
   f->name = strdup(entry_point);
   return f;
}

static struct lmn_function *
FN(mtl_function_descriptor *f)
{
   return LMN_CAST(lmn_function, f, LMN_KIND_FUNCTION);
}

/* Sends a pipeline create and waits for it; on failure frees `p`, which the host never made. */
static void *
pipeline_call(struct lmn_device *dev, struct lmn_obj *p, uint16_t op, struct lmn_iov *iov,
              unsigned n, struct lmn_slot *slot, const char *what)
{
   struct lmn_reply r = lmn_call(dev, op, iov, n, slot);
   lmn_slot_free(dev, slot);
   if (r.status != 0) {
      fprintf(stderr, "[limina-kk] the host could not create a %s (error %u)\n", what, r.status);
      free(p);
      return NULL;
   }
   return p;
}

mtl_compute_pipeline_state *
mtl_new_compute_pipeline_state(mtl_compiler *compiler, mtl_function_descriptor *function,
                               uint64_t max_total_threads_per_threadgroup)
{
   struct lmn_device *dev = lmn_cast(compiler, LMN_KIND_COMPILER, __func__)->dev;
   struct lmn_function *f = FN(function);
   struct lmn_obj *p =
      lmn_obj_new(dev, LMN_KIND_COMPUTE_PIPELINE, sizeof(*p), lmn_destroy_host_object);
   struct lmn_slot slot;
   if (!p)
      return NULL;
   if (!lmn_slot_alloc(dev, &slot)) {
      free(p);
      return NULL;
   }
   p->handle = lmn_new_handle(dev);
   uint32_t len = (uint32_t)strlen(f->name);
   struct lmn_create_compute_pipeline c = {
      .handle = p->handle,
      .library = f->library->handle,
      .name_len = len,
      .max_total_threads = max_total_threads_per_threadgroup,
      .reply = slot.ref,
   };
   struct lmn_iov iov[2] = {{&c, sizeof(c)}, {f->name, len}};
   return pipeline_call(dev, p, LMN_OP_CREATE_COMPUTE_PIPELINE, iov, 2, &slot,
                        "compute pipeline");
}

struct lmn_render_pipeline_desc {
   struct lmn_obj base;
   struct lmn_function *vertex, *fragment;
   struct lmn_create_render_pipeline c;
};

static void
render_pipeline_desc_destroy(struct lmn_obj *obj)
{
   struct lmn_render_pipeline_desc *d = (struct lmn_render_pipeline_desc *)obj;
   mtl_release(d->vertex);
   mtl_release(d->fragment);
}

mtl_render_pipeline_descriptor *
mtl_new_render_pipeline_descriptor(void)
{
   struct lmn_render_pipeline_desc *d = lmn_obj_new(
      NULL, LMN_KIND_RENDER_PIPELINE_DESC, sizeof(*d), render_pipeline_desc_destroy);
   if (d) {
      d->c.raster_sample_count = 1;
      d->c.rasterization_enabled = 1;
   }
   return d;
}

static struct lmn_render_pipeline_desc *
RPD(mtl_render_pipeline_descriptor *d)
{
   return LMN_CAST(lmn_render_pipeline_desc, d, LMN_KIND_RENDER_PIPELINE_DESC);
}

void
mtl_render_pipeline_descriptor_set_vertex_shader(mtl_render_pipeline_descriptor *d,
                                                 mtl_function_descriptor *shader)
{
   mtl_release(RPD(d)->vertex);
   RPD(d)->vertex = mtl_retain(FN(shader));
}

void
mtl_render_pipeline_descriptor_set_fragment_shader(mtl_render_pipeline_descriptor *d,
                                                   mtl_function_descriptor *shader)
{
   mtl_release(RPD(d)->fragment);
   RPD(d)->fragment = shader ? mtl_retain(FN(shader)) : NULL;
}

void
mtl_render_pipeline_descriptor_set_input_primitive_topology(
   mtl_render_pipeline_descriptor *d, enum mtl_primitive_topology_class topology_class)
{
   RPD(d)->c.topology_class = topology_class;
}

void
mtl_render_pipeline_descriptor_set_color_attachment_format(mtl_render_pipeline_descriptor *d,
                                                           uint8_t index,
                                                           enum mtl_pixel_format format)
{
   RPD(d)->c.color_formats[index] = format;
}

void
mtl_render_pipeline_descriptor_set_raster_sample_count(mtl_render_pipeline_descriptor *d,
                                                       uint32_t sample_count)
{
   RPD(d)->c.raster_sample_count = sample_count;
}

void
mtl_render_pipeline_descriptor_set_rasterization_enabled(mtl_render_pipeline_descriptor *d,
                                                         bool enabled)
{
   RPD(d)->c.rasterization_enabled = enabled;
}

void
mtl_render_pipeline_descriptor_set_max_vertex_amplification_count(
   mtl_render_pipeline_descriptor *d, uint32_t count)
{
   RPD(d)->c.max_vertex_amplification = count;
}

void
mtl_render_pipeline_descriptor_set_label(mtl_render_pipeline_descriptor *d, const char *label)
{
}

mtl_render_pipeline_state *
mtl_new_render_pipeline(mtl_compiler *compiler, mtl_render_pass_descriptor *descriptor)
{
   /* The parameter is a render pipeline descriptor, whatever its type says. */
   struct lmn_device *dev = lmn_cast(compiler, LMN_KIND_COMPILER, __func__)->dev;
   struct lmn_render_pipeline_desc *d = RPD(descriptor);
   if (!d->vertex) {
      fprintf(stderr, "[limina-kk] a render pipeline without a vertex function\n");
      return NULL;
   }
   struct lmn_obj *p =
      lmn_obj_new(dev, LMN_KIND_RENDER_PIPELINE, sizeof(*p), lmn_destroy_host_object);
   struct lmn_slot slot;
   if (!p)
      return NULL;
   if (!lmn_slot_alloc(dev, &slot)) {
      free(p);
      return NULL;
   }
   p->handle = lmn_new_handle(dev);
   struct lmn_create_render_pipeline c = d->c;
   uint32_t vlen = (uint32_t)strlen(d->vertex->name);
   uint32_t flen = d->fragment ? (uint32_t)strlen(d->fragment->name) : 0;
   c.handle = p->handle;
   c.vertex_library = d->vertex->library->handle;
   c.fragment_library = d->fragment ? d->fragment->library->handle : 0;
   c.vertex_name_len = vlen;
   c.fragment_name_len = flen;
   c.reply = slot.ref;
   struct lmn_iov iov[3] = {
      {&c, sizeof(c)},
      {d->vertex->name, vlen},
      {d->fragment ? d->fragment->name : "", flen},
   };
   return pipeline_call(dev, p, LMN_OP_CREATE_RENDER_PIPELINE, iov, 3, &slot, "render pipeline");
}
