/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_cmd_buffer.h"
#include "util/u_atomic.h"

#include <dlfcn.h>

#include "kk_buffer.h"
#include "kk_cmd_pool.h"
#include "kk_descriptor_set_layout.h"
#include "kk_entrypoints.h"
#include "kk_image_view.h"
#include "kk_limina_capture.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_command_buffer.h"
#include "kosmickrisp/bridge/mtl_device.h"
#include "kosmickrisp/bridge/mtl_encoder.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_alloc.h"
#include "vk_pipeline_layout.h"

static void
kk_descriptor_state_fini(struct kk_cmd_buffer *cmd,
                         struct kk_descriptor_state *desc)
{
   struct kk_cmd_pool *pool = kk_cmd_buffer_pool(cmd);

   for (unsigned i = 0; i < KK_MAX_SETS; i++) {
      vk_free(&pool->vk.alloc, desc->push[i]);
      desc->push[i] = NULL;
   }
}

static void
kk_cmd_release_resources(struct kk_device *dev, struct kk_cmd_buffer *cmd)
{
   struct kk_cmd_pool *pool = kk_cmd_buffer_pool(cmd);

   kk_cmd_release_dynamic_ds_state(cmd);
   kk_descriptor_state_fini(cmd, &cmd->state.gfx.descriptors);
   kk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);

   kk_cmd_pool_free_bo_list(pool, &cmd->uploader.bos);

   /* Release all command buffers used */
   util_dynarray_foreach(&cmd->submit_cmd_bufs, mtl_command_buffer *, cmd_buf) {
      mtl_release(*cmd_buf);
   }
   util_dynarray_clear(&cmd->submit_cmd_bufs);

   /* limina: any charge still outstanding belongs to a command buffer that was recorded but
    * never submitted (or whose submission already discharged and cleared this array). Discharge
    * it here so the allocator can leave DRAINING. */
   util_dynarray_foreach(&cmd->charged_allocs, kk_pooled_alloc_ptr, pap) {
      kk_alloc_pool_discharge(dev, *pap);
   }
   util_dynarray_clear(&cmd->charged_allocs);

   /* Release all BOs used as descriptor buffers for submissions */
   util_dynarray_foreach(&cmd->large_bos, struct kk_bo *, bo) {
      kk_destroy_bo(dev, *bo);
   }
   util_dynarray_clear(&cmd->large_bos);
}

static void
kk_destroy_encoder_state(struct kk_encoder_state *es)
{
   assert(es->encoder == NULL);
   assert(es->cmd_buf == NULL);
   /* limina: the allocator is a pool borrow, returned at kk_stop_encoder. Nothing to release. */
   assert(es->pa == NULL);

   util_dynarray_fini(&es->ts_resolves);
}

static void
kk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct kk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct kk_cmd_buffer, vk);
   struct kk_cmd_pool *pool = kk_cmd_buffer_pool(cmd);

   mtl_release(cmd->argument_table);
   kk_destroy_encoder_state(&cmd->cmp[0]);
   kk_destroy_encoder_state(&cmd->cmp[1]);
   kk_destroy_encoder_state(&cmd->gfx);

   vk_command_buffer_finish(&cmd->vk);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   kk_cmd_release_resources(dev, cmd);
   util_dynarray_fini(&cmd->submit_cmd_bufs);
   util_dynarray_fini(&cmd->charged_allocs);
   util_dynarray_fini(&cmd->large_bos);

   vk_free(&pool->vk.alloc, cmd);
}

static bool
kk_init_encoder_state(struct kk_encoder_state *es, mtl_device *handle)
{
   /* limina: allocators now come from the device pool at encoder start. */
   es->pa = NULL;
   es->allocator = NULL;
   es->ts_resolves = UTIL_DYNARRAY_INIT;
   return true;
}

static VkResult
kk_create_cmd_buffer(struct vk_command_pool *vk_pool,
                     VkCommandBufferLevel level,
                     struct vk_command_buffer **cmd_buffer_out)
{
   struct kk_cmd_pool *pool = container_of(vk_pool, struct kk_cmd_pool, vk);
   struct kk_device *dev = kk_cmd_pool_device(pool);
   struct kk_cmd_buffer *cmd;
   VkResult result;

   cmd = vk_zalloc(&pool->vk.alloc, sizeof(*cmd), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cmd == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = vk_command_buffer_init_with_params(
      &cmd->vk, &(struct vk_command_buffer_init_params){
                   .pool = &pool->vk,
                   .ops = &kk_cmd_buffer_ops,
                   .level = level,
                   .needs_cmd_queue = true,
                });
   if (result != VK_SUCCESS)
      goto alloc_fail;

   cmd->pre_gfx = &cmd->cmp[0];
   cmd->post_gfx = &cmd->cmp[1];
   if (!kk_init_encoder_state(cmd->pre_gfx, dev->mtl_handle))
      goto pre_gfx_allocator_fail;

   if (!kk_init_encoder_state(&cmd->gfx, dev->mtl_handle))
      goto gfx_allocator_fail;

   if (!kk_init_encoder_state(cmd->post_gfx, dev->mtl_handle))
      goto post_gfx_allocator_fail;

   {
      mtl_argument_table_descriptor *desc = mtl_new_argument_table_descriptor();
      /* Root at 0, samplers at 1 and per draw data at 2 */
      mtl_set_max_buffer_binding_count(desc, 3u);
      cmd->argument_table = mtl_new_argument_table(dev->mtl_handle, desc);
      mtl_set_address(cmd->argument_table, dev->samplers.table.bo->gpu, 1u);
      mtl_release(desc);
   }

   cmd->submit_cmd_bufs = UTIL_DYNARRAY_INIT;
   cmd->charged_allocs = UTIL_DYNARRAY_INIT;
   cmd->large_bos = UTIL_DYNARRAY_INIT;

   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;

   list_inithead(&cmd->uploader.bos);

   *cmd_buffer_out = &cmd->vk;

   return VK_SUCCESS;

post_gfx_allocator_fail:
   kk_destroy_encoder_state(&cmd->gfx);
gfx_allocator_fail:
   kk_destroy_encoder_state(cmd->pre_gfx);
pre_gfx_allocator_fail:
   vk_command_buffer_finish(&cmd->vk);
   result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
alloc_fail:
   vk_free(&pool->vk.alloc, cmd);
   return result;
}

/* limina: acquire a pooled allocator and begin `cb` on it, charging the borrow.
 *
 * The charge is taken HERE, at begin, not at commit. An allocator is returned to the pool at
 * kk_stop_encoder while its command buffers still sit uncommitted in submit_cmd_bufs; charging
 * at commit would leave a window in which pending == 0 yet the encoding memory must still
 * survive, and a reset landing there would discard commands the GPU has not been handed.
 */
static bool
kk_encoder_begin(struct kk_cmd_buffer *cmd, struct kk_encoder_state *es,
                 enum kk_alloc_class klass)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   es->pa = kk_alloc_pool_acquire(dev, klass);
   if (es->pa == NULL)
      return false;
   es->allocator = es->pa->handle;

   /* Grow the tracking slot BEFORE charging. The other order loses the charge entirely when the
    * grow fails: nothing would ever discharge it, the allocator would wedge in draining forever,
    * and under the destroy policy that silently erodes the pool below its floor. Failing here
    * instead costs only this encoder. The slot is NULLed first so a concurrent walk of the array
    * can never see an uninitialised entry (kk_alloc_pool_discharge tolerates NULL). */
   kk_pooled_alloc_ptr *slot =
      util_dynarray_grow(&cmd->charged_allocs, kk_pooled_alloc_ptr, 1);
   if (slot == NULL) {
      kk_alloc_pool_release(dev, es->pa, 0);
      es->pa = NULL;
      es->allocator = NULL;
      return false;
   }
   *slot = NULL;

   mtl_begin_command_buffer(es->cmd_buf, es->allocator);
   kk_alloc_pool_charge(dev, es->pa);
   *slot = es->pa;
   return true;
}

void
kk_reset_cmd_buffer_internal(struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   /* If the command buffer was not ended, we may have lingering encoders.
    * Call twice since post_gfx will be moved to pre_gfx but not ended. */
   cs_end(cmd);
   cs_end(cmd);
   kk_cmd_release_resources(dev, cmd);

   /* limina: no allocator resets here any more. Allocators are pool borrows returned at
    * kk_stop_encoder, and the pool resets them when they have drained. Resetting per
    * vkBeginCommandBuffer was the root of the ratchet: an epoch's every render pass piled onto
    * one allocator, so the high-water was set by the heaviest epoch and then kept for ever. */

   cmd->uploader.bo = NULL;
   cmd->uploader.offset = 0;

   memset(&cmd->state, 0, sizeof(cmd->state));
   cmd->uses_heap = false;
}

static void
kk_reset_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer,
                    UNUSED VkCommandBufferResetFlags flags)
{
   struct kk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct kk_cmd_buffer, vk);

   vk_command_buffer_reset(&cmd->vk);
   kk_reset_cmd_buffer_internal(cmd);
   cmd->submitted = false;
   cmd->one_time_submit = false;
}

const struct vk_command_buffer_ops kk_cmd_buffer_ops = {
   .create = kk_create_cmd_buffer,
   .reset = kk_reset_cmd_buffer,
   .destroy = kk_destroy_cmd_buffer,
};

VKAPI_ATTR VkResult VKAPI_CALL
kk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                      const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   kk_reset_cmd_buffer(&cmd->vk, 0u);
   vk_command_buffer_begin(&cmd->vk, pBeginInfo);
   cmd->one_time_submit =
      pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   /* Call twice since post_gfx will be moved to pre_gfx but not ended. */
   cs_end(cmd);
   cs_end(cmd);

   return vk_command_buffer_end(&cmd->vk);
}

static bool
kk_can_ignore_barrier(VkAccessFlags2 access, VkPipelineStageFlags2 stage)
{
   if (access == VK_ACCESS_2_NONE || stage == VK_PIPELINE_STAGE_2_NONE)
      return true;

   const VkAccessFlags2 ignore_access =
      VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
   const VkPipelineStageFlags2 ignore_stage = VK_PIPELINE_STAGE_2_HOST_BIT;
   return (!(access ^ ignore_access)) || (!(stage ^ ignore_stage));
}

/* LIMINA instrumentation: has this texture ever been a render target before?
 *
 * Ordering is exhausted as an explanation, so the question becomes what a pass ENCODES rather
 * than when it runs. A pass that begins on a texture already containing drawing, with a load
 * action of CLEAR or DONT_CARE, discards that drawing -- which is precisely the shape of a card
 * keeping its background and its later rows while losing its header and title.
 *
 * An open-addressing set of texture pointers, never emptied: it only ever answers "seen before",
 * and a stale entry for a freed texture is harmless here (it can only make the detector more
 * suspicious, and false hits are cheap to check by hand). */
#define KK_LIMINA_SEEN_BITS 14u
#define KK_LIMINA_SEEN_SIZE (1u << KK_LIMINA_SEEN_BITS)
static const void *kk_limina_seen[KK_LIMINA_SEEN_SIZE];

static bool
kk_limina_seen_texture(const void *tex)
{
   uintptr_t h = (uintptr_t)tex;
   h = (h >> 4) * 2654435761u;
   for (unsigned probe = 0; probe < 64u; ++probe) {
      unsigned i = (unsigned)((h + probe) & (KK_LIMINA_SEEN_SIZE - 1u));
      if (kk_limina_seen[i] == tex)
         return true;
      if (kk_limina_seen[i] == NULL) {
         kk_limina_seen[i] = tex;
         return false;
      }
   }
   /* Table full along this probe run: report "seen" so a miss is never invented. */
   return true;
}

void
cs_start_render(struct kk_cmd_buffer *cmd)
{
   p_atomic_inc(&kk_limina_counts.render_pass_starts);
   kk_limina_counts_tick("rp");

   {
      struct kk_rendering_state *r = &cmd->state.gfx.render;
      for (uint32_t i = 0; i < r->color_att_count; i++) {
         const struct kk_image_view *iview = r->color_att[i].iview;
         if (!iview || r->color_map[i] == MESA_VK_ATTACHMENT_UNUSED)
            continue;

         const void *tex = iview->planes[0].mtl_handle_render;
         enum mtl_load_action load = r->limina_load_action[i];
         bool seen = kk_limina_seen_texture(tex);

         /* limina: count this pass for the triggered GPU capture while the extent is in hand. */
         kk_limina_capture_note_pass(iview->vk.extent.width, iview->vk.extent.height, tex);

         if (!seen) {
            /* A label offscreen is created per card, so its first pass is ALWAYS a fresh
             * attachment -- which means the one bucket the missing text lives in was being
             * counted and then skipped. Split it by size so a new label or icon is separable
             * from a new full-surface target. */
            p_atomic_inc(&kk_limina_counts.start_fresh);
            if (iview->vk.extent.width <= 512u && iview->vk.extent.height <= 512u)
               p_atomic_inc(&kk_limina_counts.start_fresh_small);
            continue;
         }

         if (load == MTL_LOAD_ACTION_LOAD) {
            p_atomic_inc(&kk_limina_counts.start_seen_load);
            continue;
         }

         if (load == MTL_LOAD_ACTION_CLEAR)
            p_atomic_inc(&kk_limina_counts.start_seen_clear);
         else
            p_atomic_inc(&kk_limina_counts.start_seen_dontcare);

         p_atomic_inc(&kk_limina_counts.reload_hazard);

         /* LIMINA A/B lever, KK_LIMINA_FORCE_LOAD=1: begin the pass by loading the target
          * instead of clearing or discarding it. Detecting these says little on its own --
          * clearing a reused offscreen before redrawing it is ordinary -- so turn the detector
          * into an arm. If the corruption survives every attachment keeping its previous
          * contents, then nothing is being discarded that should have been kept, and content
          * loss at pass start is out. Not a fix: it leaves stale pixels wherever a clear was
          * genuinely intended. */
         static int force_load = -1;
         if (unlikely(force_load < 0)) {
            const char *env = getenv("KK_LIMINA_FORCE_LOAD");
            force_load = !env || !strcmp(env, "0") ? 0
                         : !strcmp(env, "small")   ? 2
                                                   : 1;
            fprintf(stderr, "[LIMINA] KK pass-start load %s\n",
                    force_load == 2 ? "FORCED to LOAD on small drawn targets "
                                      "(KK_LIMINA_FORCE_LOAD=small)"
                    : force_load ? "FORCED to LOAD on all drawn targets (KK_LIMINA_FORCE_LOAD)"
                                 : "as encoded (default)");
            fflush(stderr);
         }
         /* =1 on every drawn target is unusable as a measurement: card backgrounds stop being
          * cleared, previous frames pile up inside them, and the leftover ink reads as a healthy
          * header -- a clean 0/20 that pixel inspection shows is stale text drawn twice over. So
          * =small restricts the force to the offscreen size class a label or icon uses, where a
          * discarded target would actually explain the missing rows, and leaves the large
          * surfaces to clear as encoded. */
         bool small = iview->vk.extent.width <= 512u && iview->vk.extent.height <= 512u;
         if (force_load == 1 || (force_load == 2 && small)) {
            mtl_render_pass_attachment_descriptor_set_load_action(
               mtl_render_pass_descriptor_get_color_attachment(
                  cmd->state.gfx.render_pass_descriptor, r->color_map[i]),
               MTL_LOAD_ACTION_LOAD);
         }
         /* A card's label and icon offscreens are small; the scanout is not. Splitting them
          * keeps a full-surface clear -- which is normal and expected every frame -- from
          * drowning out the case that matters. */
         if (iview->vk.extent.width <= 512u && iview->vk.extent.height <= 512u)
            p_atomic_inc(&kk_limina_counts.reload_hazard_small);
      }
   }
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_graphics_state *state = &cmd->state.gfx;
   uint32_t view_mask = state->render.view_mask;
   assert(state->render_pass_descriptor);

   /* limina: the only place a triggered GPU capture may open. Metal records at the API layer and
    * only sees command buffers CREATED after startCapture, so the hook has to sit before the
    * creation below -- one line later and the capture would silently omit this command buffer. */
   kk_limina_capture_cmdbuf_begin(dev);
   cmd->gfx.cmd_buf = mtl_new_command_buffer(dev->mtl_handle);
   if (!kk_encoder_begin(cmd, &cmd->gfx, KK_ALLOC_CLASS_RENDER)) {
      mtl_release(cmd->gfx.cmd_buf);
      cmd->gfx.cmd_buf = NULL;
      vk_device_set_lost(&dev->vk, "out of command allocators");
      return;
   }
   cmd->gfx.encoder = mtl_new_render_command_encoder_with_descriptor(
      cmd->gfx.cmd_buf, state->render_pass_descriptor);
   cmd->gfx.ops = 0;

   /* limina: name the encoder when this pass is one the triggered capture is hunting, so the
    * working/failing pair is findable by name in Xcode rather than by scrubbing every encoder. */
   if (unlikely(kk_limina_capture_pending_label[0] != '\0')) {
      mtl_render_encoder_set_label(cmd->gfx.encoder, kk_limina_capture_pending_label);
      kk_limina_capture_pending_label[0] = '\0';
   }

   uint32_t layer_ids[KK_MAX_MULTIVIEW_VIEW_COUNT] = {};
   uint32_t count = 0u;
   u_foreach_bit(id, view_mask)
      layer_ids[count++] = id;
   if (view_mask == 0u) {
      layer_ids[count++] = 0;
   }
   mtl_set_vertex_amplification_count(cmd->gfx.encoder, layer_ids, count);

   /* Argument table won't ever change */
   mtl_render_set_argument_table(
      cmd->gfx.encoder, cmd->argument_table,
      MTL_RENDER_STAGE_VERTEX | MTL_RENDER_STAGE_FRAGMENT);

   kk_cmd_buffer_dirty_all_gfx(cmd);
}

mtl_render_encoder *
cs_get_render(struct kk_cmd_buffer *cmd)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;

   if (gfx->need_to_start_render_pass) {
      /* limina: Metal requires defaultRasterSampleCount >= 1; an attachment-less
       * render pass has no attachment to derive it from and a draw can reach
       * here before a pipeline set it, leaving 0 -> the render command encoder
       * comes back nil and the pass start crashes (observed pre-MTL4 as a
       * worker-killing assert; on the live-record path the nil propagates to
       * mtl_set_vertex_amplification_count). Clamp. */
      gfx->render.samples =
         gfx->pipeline_sample_count ? gfx->pipeline_sample_count : 1u;
      mtl_render_pass_descriptor_set_default_raster_sample_count(
         cmd->state.gfx.render_pass_descriptor, gfx->render.samples);
      gfx->need_to_start_render_pass = false;
      cs_start_render(cmd);
   }

   cmd->gfx.ops++;
   return cmd->gfx.encoder;
}

static void
kk_start_compute_encoder(struct kk_cmd_buffer *cmd, struct kk_encoder_state *es,
                         mtl_device *handle,
                         mtl_argument_table *argument_table)
{
   kk_limina_capture_cmdbuf_begin(kk_cmd_buffer_device(cmd));
   es->cmd_buf = mtl_new_command_buffer(handle);
   if (!kk_encoder_begin(cmd, es, KK_ALLOC_CLASS_COMPUTE)) {
      mtl_release(es->cmd_buf);
      es->cmd_buf = NULL;
      return;
   }
   es->encoder = mtl_new_compute_command_encoder(es->cmd_buf);
   es->ops = 0;

   /* Argument table won't ever change */
   mtl_compute_set_argument_table(es->encoder, argument_table);
}

mtl_compute_encoder *
cs_get_compute(struct kk_cmd_buffer *cmd, bool pre_gfx)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   mtl_compute_encoder *encoder;
   if (cmd->gfx.encoder) {
      /* pre_gfx while a pass is open is the dangerous route: cs_end submits pre_gfx BEFORE the
       * gfx command buffer, so this work overtakes draws recorded earlier in the same pass.
       * post_gfx is the safe-by-design route. Vulkan forbids copies and dispatches inside a
       * render pass, so conformant traffic should only ever reach post_gfx here. */
      if (pre_gfx) {
         p_atomic_inc(&kk_limina_counts.compute_during_pass_pregfx);
         kk_limina_note_midpass_caller(__builtin_return_address(1));
      }
      else
         p_atomic_inc(&kk_limina_counts.compute_during_pass_postgfx);
   }
   /* If we are not inside a render, we can just take pre_gfx. */
   if (!cmd->gfx.encoder || pre_gfx) {
      if (!cmd->pre_gfx->encoder) {
         kk_start_compute_encoder(cmd, cmd->pre_gfx, dev->mtl_handle,
                                  cmd->argument_table);
      }
      encoder = cmd->pre_gfx->encoder;
      cmd->pre_gfx->ops++;
   } else {
      if (!cmd->post_gfx->encoder) {
         kk_start_compute_encoder(cmd, cmd->post_gfx, dev->mtl_handle,
                                  cmd->argument_table);
      }
      encoder = cmd->post_gfx->encoder;
      cmd->post_gfx->ops++;
   }

   return encoder;
}

static void
kk_stop_encoder(struct kk_cmd_buffer *cmd, struct kk_encoder_state *es)
{
   /* TODO_KOSMICKRISP This is probably overkill */
   mtl_barrier_after_stages(es->encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_end_encoding(es->encoder);
   mtl_release(es->encoder);
   es->encoder = NULL;

   /* Fold the pending timestamp counter-heap resolves into `cmd_buf` */
   util_dynarray_foreach(&es->ts_resolves, struct kk_ts_resolve, r) {
      mtl_command_resolve_counter_heap(es->cmd_buf, r->heap, r->index, 1u,
                                       r->dst_addr);
   }

   util_dynarray_clear(&es->ts_resolves);

   mtl_end_command_buffer(es->cmd_buf);

   /* limina: reuse is legal the moment the command buffer ends (Apple: "You can safely reuse
    * command allocators after ending the command buffer"); only *reset* needs GPU completion,
    * which the pool gates on the charge taken at begin. So the borrow goes back now. */
   kk_alloc_pool_release(kk_cmd_buffer_device(cmd), es->pa, es->ops);
   es->pa = NULL;
   es->allocator = NULL;

   util_dynarray_append(&cmd->submit_cmd_bufs, es->cmd_buf);
   es->cmd_buf = NULL;
}

void
cs_end(struct kk_cmd_buffer *cmd)
{
   assert(cmd);

   if (cmd->pre_gfx->encoder) {
      /* Submit pre_gfx now that its encoder is closed. Command buffers are
       * appended here (rather than at creation) so submit_cmd_bufs stays in
       * encode order: pre_gfx first, then gfx below. post_gfx is promoted into
       * the pre_gfx slot with its encoder still open, so it is submitted by a
       * later cs_end() and therefore always ends up after gfx. This is why
       * every flush site calls cs_end() twice. */
      kk_stop_encoder(cmd, cmd->pre_gfx);

      SWAP(cmd->pre_gfx, cmd->post_gfx);
   } else if (cmd->post_gfx->encoder) {
      /* No pre_gfx, but a post_gfx exists (e.g. compute issued during a render
       * pass). Promote it so a later cs_end() closes and submits it after the
       * gfx command buffer appended below. */
      SWAP(cmd->pre_gfx, cmd->post_gfx);
   }

   if (cmd->gfx.encoder) {
      kk_stop_encoder(cmd, &cmd->gfx);
   }
}

void
kk_cmd_bind_root_to_argument_table(struct kk_cmd_buffer *cmd, uint64_t addr)
{
   mtl_set_address(cmd->argument_table, addr, 0u);
   cmd->state.root_addr = addr;
}

/* LIMINA A/B lever for kk_CmdPipelineBarrier2, read once from KK_LIMINA_BARRIER. Both arms are
 * deliberately off by default: `norestart` is not generally correct (input attachments read as
 * textures need the pass break) and `widen` is a blunt over-synchronisation. They exist to split
 * "KK delivers the ordering zink asks for" from "KK drops or mis-scopes it". */
enum kk_limina_barrier_mode
kk_limina_barrier_mode(void)
{
   static int mode = -1;
   if (unlikely(mode < 0)) {
      const char *env = getenv("KK_LIMINA_BARRIER");
      if (env && !strcmp(env, "norestart"))
         mode = KK_LIMINA_BARRIER_NORESTART;
      else if (env && !strcmp(env, "widen"))
         mode = KK_LIMINA_BARRIER_WIDEN;
      else
         mode = KK_LIMINA_BARRIER_DEFAULT;
      /* Self-evidencing: an arm whose engagement cannot be observed in the log is worse than no
       * arm at all -- a silent no-op reads exactly like a clean exoneration. */
      fprintf(stderr, "[LIMINA] KK barrier mode = %s\n",
              mode == KK_LIMINA_BARRIER_NORESTART ? "norestart (pass restart SUPPRESSED)"
              : mode == KK_LIMINA_BARRIER_WIDEN   ? "widen (pre_gfx barrier scope = ALL)"
                                                  : "default");
      fflush(stderr);
   }
   return mode;
}

/* LIMINA instrumentation: the GPU-side bump allocator behind geometry unrolling. There is one
 * 128 MiB heap per device, its `bottom` is a plain uint32 in a CPU-mapped BO, and it is reset by
 * zeroing that word. The reset is issued through kk_cmd_write, which routes to post_gfx while a
 * render pass is open -- so the reset is submitted AFTER the draws, while the allocations that
 * feed them are hoisted to pre_gfx and submitted BEFORE. Reading the watermark says whether that
 * ever actually runs the heap dry, rather than leaving it to argument. */
uint32_t kk_limina_heap_size;
uint32_t kk_limina_heap_hiwater;

/* LIMINA instrumentation: attribute the compute work that is issued with pre_gfx=true while a
 * render pass is open. That route is the interesting one -- cs_end submits pre_gfx BEFORE the gfx
 * command buffer, so the work overtakes draws already recorded in the same pass. Vulkan forbids
 * copies and dispatches inside a render pass, so a large count here must come from KK's own
 * draw-time helpers rather than from client traffic; dladdr says which. */
#define KK_LIMINA_CALLERS 12
static struct {
   const void *addr;
   const char *name;
   uint64_t count;
} kk_limina_midpass[KK_LIMINA_CALLERS];

void
kk_limina_note_midpass_caller(void *ret_addr)
{
   for (unsigned i = 0; i < KK_LIMINA_CALLERS; ++i) {
      if (kk_limina_midpass[i].addr == ret_addr) {
         kk_limina_midpass[i].count++;
         return;
      }
      if (kk_limina_midpass[i].addr == NULL) {
         Dl_info info;
         kk_limina_midpass[i].addr = ret_addr;
         kk_limina_midpass[i].name =
            dladdr(ret_addr, &info) && info.dli_sname ? info.dli_sname : "?";
         kk_limina_midpass[i].count = 1u;
         return;
      }
   }
}

static void
kk_limina_print_midpass_callers(void)
{
   for (unsigned i = 0; i < KK_LIMINA_CALLERS; ++i) {
      if (kk_limina_midpass[i].addr == NULL)
         break;
      fprintf(stderr, "[LIMINA]   midpass pre_gfx caller: %-44s %llu\n",
              kk_limina_midpass[i].name,
              (unsigned long long)kk_limina_midpass[i].count);
   }
}

/* LIMINA instrumentation: which KK paths a workload actually exercises. Printed periodically
 * rather than per event -- the point is the ratio, not a trace. */
struct kk_limina_counts kk_limina_counts;

void
kk_limina_counts_tick(const char *why)
{
   static uint64_t last;
   uint64_t n = p_atomic_inc_return(&kk_limina_counts.ticks);
   if (n > 3u && n - last < 2000u)
      return;
   last = n;
   fprintf(stderr,
           "[LIMINA] KK counts: barriers=%llu (breaks_pass=%llu pregfx=%llu noop=%llu) "
           "render_pass_starts=%llu restarts=%llu/%llu(flush) "
           "compute_during_pass=%llu/%llu(pregfx) color_map_nonidentity=%llu/%llu (%s)\n",
           (unsigned long long)kk_limina_counts.barriers,
           (unsigned long long)kk_limina_counts.barrier_breaks_pass,
           (unsigned long long)kk_limina_counts.barrier_pregfx,
           (unsigned long long)kk_limina_counts.barrier_noop,
           (unsigned long long)kk_limina_counts.render_pass_starts,
           (unsigned long long)kk_limina_counts.render_pass_restarts,
           (unsigned long long)kk_limina_counts.render_pass_restarts_flush,
           (unsigned long long)kk_limina_counts.compute_during_pass_postgfx,
           (unsigned long long)kk_limina_counts.compute_during_pass_pregfx,
           (unsigned long long)kk_limina_counts.color_map_nonidentity,
           (unsigned long long)kk_limina_counts.color_att_seen, why);
   /* Only the accumulated high-water mark: this site has no device, and the bump pointer at an
    * arbitrary tick is not the number of interest anyway. kk_heap() samples it per use. */
   if (kk_limina_heap_size) {
      fprintf(stderr, "[LIMINA]   poly heap: hiwater=%u size=%u (%.2f%% of heap)\n",
              kk_limina_heap_hiwater, kk_limina_heap_size,
              100.0 * kk_limina_heap_hiwater / kk_limina_heap_size);
   }
   fprintf(stderr,
           "[LIMINA]   unroll triggers: fan=%llu promote=%llu robust=%llu restart=%llu\n",
           (unsigned long long)kk_limina_counts.unroll_trig_fan,
           (unsigned long long)kk_limina_counts.unroll_trig_promote,
           (unsigned long long)kk_limina_counts.unroll_trig_robust,
           (unsigned long long)kk_limina_counts.unroll_trig_restart);
   fprintf(stderr, "[LIMINA]   unroll_geometry calls=%llu (fan=%llu strip=%llu other=%llu)\n",
           (unsigned long long)kk_limina_counts.unroll_calls,
           (unsigned long long)kk_limina_counts.unroll_fan,
           (unsigned long long)kk_limina_counts.unroll_strip,
           (unsigned long long)kk_limina_counts.unroll_other);
   fprintf(stderr,
           "[LIMINA]   pass starts: fresh=%llu seen+LOAD=%llu seen+CLEAR=%llu "
           "seen+DONTCARE=%llu (fresh small=%llu) | reload_hazard=%llu (small=%llu)"
           " | pass ends DONTCARE=%llu (small=%llu)\n",
           (unsigned long long)kk_limina_counts.start_fresh,
           (unsigned long long)kk_limina_counts.start_seen_load,
           (unsigned long long)kk_limina_counts.start_seen_clear,
           (unsigned long long)kk_limina_counts.start_seen_dontcare,
           (unsigned long long)kk_limina_counts.start_fresh_small,
           (unsigned long long)kk_limina_counts.reload_hazard,
           (unsigned long long)kk_limina_counts.reload_hazard_small,
           (unsigned long long)kk_limina_counts.store_dontcare,
           (unsigned long long)kk_limina_counts.store_dontcare_small);
   kk_limina_print_midpass_callers();
   fflush(stderr);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                       const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   p_atomic_inc(&kk_limina_counts.barriers);
   kk_limina_counts_tick("barrier");

   /* TODO_KOSMICKRISP Don't break the render pass and add a single encoder
    * barrier. This requires to read directly from the framebuffer which
    * requires not reading input attachments as textures.
    */
   if (cmd->gfx.encoder) {
      /* LIMINA A/B lever, KK_LIMINA_BARRIER=norestart: implement the TODO above -- keep the
       * render pass open and issue an in-encoder barrier instead of tearing the pass down and
       * restarting it. This is NOT generally correct (input attachments read as textures need
       * the break), so it is off by default. It exists to answer one question: is the
       * store/end/restart-with-LOAD round trip losing everything drawn before the barrier?
       * That is the shape of the gnome-shell card corruption -- earlier-painted content (a
       * notification's header row and title) gone, later-painted content (icon, body) intact. */
      p_atomic_inc(&kk_limina_counts.barrier_breaks_pass);
      if (kk_limina_barrier_mode() == KK_LIMINA_BARRIER_NORESTART) {
         mtl_barrier_after_stages(cmd->gfx.encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
         return;
      }
      kk_apply_attachment_store_ops(cmd, true);
      cs_end(cmd);
      p_atomic_inc(&kk_limina_counts.render_pass_restarts);
      cs_start_render(cmd);
   } else if (cmd->pre_gfx->encoder) {
      /* We chain encoders, so an intra-encoder barrier is enough here:
       * no need to tear down and recreate the encoder.
       */
      p_atomic_inc(&kk_limina_counts.barrier_pregfx);
      /* LIMINA A/B lever, KK_LIMINA_BARRIER=widen. The default scope below covers only
       * DISPATCH|BLIT on both sides, so a barrier whose consumer is a RENDER stage -- which is
       * every "upload a glyph, then sample it" edge, since KK encodes copies as compute -- is
       * scoped away to nothing. `widen` takes both sides to MTL_STAGE_ALL. */
      enum mtl_stages stages = kk_limina_barrier_mode() == KK_LIMINA_BARRIER_WIDEN
                                  ? MTL_STAGE_ALL
                                  : (MTL_STAGE_DISPATCH | MTL_STAGE_BLIT);
      mtl_barrier_after_encoder_stages(cmd->pre_gfx->encoder, stages, stages);
   } else {
      /* Neither encoder is open, so there is nothing to barrier against and the dependency is
       * simply dropped. Counted because "the barrier did nothing" and "the barrier was scoped
       * wrong" are different faults with the same symptom. */
      p_atomic_inc(&kk_limina_counts.barrier_noop);
   }
}

static void
kk_bind_descriptor_sets(struct kk_descriptor_state *desc,
                        const VkBindDescriptorSetsInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   /* From the Vulkan 1.3.275 spec:
    *
    *    "When binding a descriptor set (see Descriptor Set Binding) to
    *    set number N...
    *
    *    If, additionally, the previously bound descriptor set for set
    *    N was bound using a pipeline layout not compatible for set N,
    *    then all bindings in sets numbered greater than N are
    *    disturbed."
    *
    * This means that, if some earlier set gets bound in such a way that
    * it changes set_dynamic_buffer_start[s], this binding is implicitly
    * invalidated.
    */
   uint8_t dyn_buffer_start =
      pipeline_layout->dynamic_descriptor_offset[info->firstSet];

   uint32_t next_dyn_offset = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      unsigned s = i + info->firstSet;
      VK_FROM_HANDLE(kk_descriptor_set, set, info->pDescriptorSets[i]);

      if (desc->sets[s] != set) {
         if (set != NULL) {
            desc->root.sets[s] = set->addr;
            desc->set_sizes[s] = set->size;
         } else {
            desc->root.sets[s] = 0;
            desc->set_sizes[s] = 0;
         }
         desc->sets[s] = set;

         /* Binding descriptors invalidates push descriptors */
         desc->push_dirty &= ~BITFIELD_BIT(s);
      }

      if (pipeline_layout->set_layouts[s] != NULL) {
         const struct kk_descriptor_set_layout *set_layout =
            vk_to_kk_descriptor_set_layout(pipeline_layout->set_layouts[s]);

         if (set != NULL && set_layout->vk.dynamic_descriptor_count > 0) {
            for (uint32_t j = 0; j < set_layout->vk.dynamic_descriptor_count;
                 j++) {
               struct kk_buffer_address addr = set->dynamic_buffers[j];
               addr.base_addr += info->pDynamicOffsets[next_dyn_offset + j];
               desc->root.dynamic_buffers[dyn_buffer_start + j] = addr;
            }
            next_dyn_offset += set->layout->vk.dynamic_descriptor_count;
         }

         dyn_buffer_start += set_layout->vk.dynamic_descriptor_count;
      } else {
         assert(set == NULL);
      }
   }
   assert(dyn_buffer_start <= KK_MAX_DYNAMIC_BUFFERS);
   assert(next_dyn_offset <= info->dynamicOffsetCount);

   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBindDescriptorSets2KHR(
   VkCommandBuffer commandBuffer,
   const VkBindDescriptorSetsInfoKHR *pBindDescriptorSetsInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      kk_bind_descriptor_sets(&cmd->state.gfx.descriptors,
                              pBindDescriptorSetsInfo);
   }

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      kk_bind_descriptor_sets(&cmd->state.cs.descriptors,
                              pBindDescriptorSetsInfo);
   }
}

static struct kk_push_descriptor_set *
kk_cmd_push_descriptors(struct kk_cmd_buffer *cmd,
                        struct kk_descriptor_state *desc,
                        struct kk_descriptor_set_layout *set_layout,
                        uint32_t set)
{
   assert(set < KK_MAX_SETS);
   if (unlikely(desc->push[set] == NULL)) {
      size_t size = sizeof(*desc->push[set]);
      desc->push[set] = vk_zalloc(&cmd->vk.pool->alloc, size, 8,
                                  VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (unlikely(desc->push[set] == NULL)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
   }

   /* Pushing descriptors replaces whatever sets are bound */
   desc->push[set]->layout = set_layout;
   desc->push[set]->limina_used_size =
      MAX2(desc->push[set]->limina_used_size,
           set_layout->non_variable_descriptor_buffer_size);
   desc->sets[set] = NULL;
   desc->push_dirty |= BITFIELD_BIT(set);

   return desc->push[set];
}

static void
kk_push_descriptor_set(struct kk_cmd_buffer *cmd,
                       struct kk_descriptor_state *desc,
                       const VkPushDescriptorSetInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   struct kk_descriptor_set_layout *set_layout =
      vk_to_kk_descriptor_set_layout(pipeline_layout->set_layouts[info->set]);

   struct kk_push_descriptor_set *push_set =
      kk_cmd_push_descriptors(cmd, desc, set_layout, info->set);
   if (unlikely(push_set == NULL))
      return;

   kk_push_descriptor_set_update(push_set, info->descriptorWriteCount,
                                 info->pDescriptorWrites);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPushDescriptorSet2KHR(
   VkCommandBuffer commandBuffer,
   const VkPushDescriptorSetInfoKHR *pPushDescriptorSetInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      kk_push_descriptor_set(cmd, &cmd->state.gfx.descriptors,
                             pPushDescriptorSetInfo);
   }

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      kk_push_descriptor_set(cmd, &cmd->state.cs.descriptors,
                             pPushDescriptorSetInfo);
   }
}

static void
kk_push_constants(UNUSED struct kk_cmd_buffer *cmd,
                  struct kk_descriptor_state *desc,
                  const VkPushConstantsInfoKHR *info)
{
   memcpy(desc->root.push + info->offset, info->pValues, info->size);
   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPushConstants2KHR(VkCommandBuffer commandBuffer,
                        const VkPushConstantsInfoKHR *pPushConstantsInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS)
      kk_push_constants(cmd, &cmd->state.gfx.descriptors, pPushConstantsInfo);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      kk_push_constants(cmd, &cmd->state.cs.descriptors, pPushConstantsInfo);
}

void
kk_cmd_release_dynamic_ds_state(struct kk_cmd_buffer *cmd)
{
   if (cmd->state.gfx.is_depth_stencil_dynamic &&
       cmd->state.gfx.depth_stencil_state)
      mtl_release(cmd->state.gfx.depth_stencil_state);
   cmd->state.gfx.depth_stencil_state = NULL;
}

static VkResult
kk_cmd_buffer_alloc_bo(struct kk_cmd_buffer *cmd, struct kk_cmd_bo **bo_out)
{
   VkResult result = kk_cmd_pool_alloc_bo(kk_cmd_buffer_pool(cmd), bo_out);
   if (result != VK_SUCCESS)
      return result;

   list_addtail(&(*bo_out)->link, &cmd->uploader.bos);
   return VK_SUCCESS;
}

struct kk_ptr
kk_pool_alloc(struct kk_cmd_buffer *cmd, uint32_t size_B, uint32_t alignment_B)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_uploader *uploader = &cmd->uploader;

   /* Specially handle large allocations owned by the command buffer, e.g. used
    * for statically allocated vertex output buffers with geometry shaders.
    */
   if (size_B > KK_CMD_BO_SIZE) {
      struct kk_bo *buffer = NULL;
      const VkResult result =
         kk_alloc_bo(dev, &cmd->vk.base, size_B, alignment_B, &buffer);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return (struct kk_ptr){0};
      }
      util_dynarray_append(&cmd->large_bos, buffer);

      return (struct kk_ptr){
         .gpu = buffer->gpu,
         .cpu = buffer->cpu,

         .buffer = buffer->map,
         .offset = 0u,
      };
   }

   assert(size_B <= KK_CMD_BO_SIZE);
   assert(alignment_B > 0);

   const uint32_t offset = align(uploader->offset, alignment_B);

   assert(offset <= KK_CMD_BO_SIZE);
   if (uploader->bo != NULL && size_B <= KK_CMD_BO_SIZE - offset) {
      uploader->offset = offset + size_B;

      return (struct kk_ptr){
         .gpu = uploader->bo->gpu + offset,
         .cpu = uploader->bo->cpu + offset,

         .buffer = uploader->bo->map,
         .offset = offset,
      };
   }

   struct kk_cmd_bo *bo;
   const VkResult result = kk_cmd_buffer_alloc_bo(cmd, &bo);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct kk_ptr){0};
   }

   /* Pick whichever of the current upload BO and the new BO will have more
    * room left to be the BO for the next upload.  If our upload size is
    * bigger than the old offset, we're better off burning the whole new
    * upload BO on this one allocation and continuing on the current upload
    * BO.
    */
   if (uploader->bo == NULL || size_B < uploader->offset) {
      uploader->bo = bo->bo;
      uploader->offset = size_B;
   }

   return (struct kk_ptr){
      .gpu = bo->bo->gpu,
      .cpu = bo->bo->cpu,

      .buffer = bo->bo->map,
      .offset = 0u,
   };
}

struct kk_ptr
kk_pool_upload(struct kk_cmd_buffer *cmd, const void *data, uint32_t size,
               uint32_t alignment)
{
   struct kk_ptr T = kk_pool_alloc(cmd, size, alignment);
   if (unlikely(T.cpu == NULL))
      return (struct kk_ptr){0};

   memcpy(T.cpu, data, size);
   return T;
}

uint64_t
kk_upload_descriptor_root(struct kk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct kk_descriptor_state *desc = kk_get_descriptors_state(cmd, bind_point);
   struct kk_root_descriptor_table *root = &desc->root;
   struct kk_ptr root_ptr = kk_pool_alloc(cmd, sizeof(*root), 8u);
   if (unlikely(!root_ptr.gpu))
      return 0u;

   root->addr = root_ptr.gpu;

   memcpy(root_ptr.cpu, root, sizeof(*root));
   desc->root_dirty = false;

   return root_ptr.gpu;
}

void
kk_cmd_buffer_flush_push_descriptors(struct kk_cmd_buffer *cmd,
                                     struct kk_descriptor_state *desc)
{
   u_foreach_bit(set_idx, desc->push_dirty) {
      struct kk_push_descriptor_set *push_set = desc->push[set_idx];

      /* limina: upload only the bytes the set actually uses instead of the full
       * KK_PUSH_DESCRIPTOR_SET_SIZE (2 KiB). Zink pushes descriptors per draw,
       * so this directly scales the upload-pool burn rate (and BO/residency
       * churn). Matches regular descriptor sets, which already size by the
       * layout. Was LIMINA_KK_SLIMPUSH, default-ON since round 19 and
       * unconditional since 2026-08-14.
       *
       * Size by the per-set high-water mark, NOT the latest push's layout:
       * retained bindings from earlier (larger-layout) pushes must stay in the
       * upload. Latest-layout sizing truncated them — visible as
       * flickering/transparent frames in glmark2 texture/shading/effect2d
       * (round 19 regression, caught by eyeball). */
      const uint32_t latest_layout_size =
         push_set->layout->non_variable_descriptor_buffer_size;
      static int limina_stats = -1;
      if (limina_stats < 0)
         limina_stats = getenv("LIMINA_KK_STATS") != NULL;
      if (limina_stats && push_set->limina_used_size > latest_layout_size) {
         static int truncs = 0;
         if (truncs++ % 1000 == 0)
            fprintf(stderr,
                    "[LIMINA-KK-SLIMPUSH] retained-binding carry: used=%u > "
                    "latest-layout=%u (x%d) — latest-layout sizing would "
                    "truncate\n",
                    push_set->limina_used_size, latest_layout_size, truncs);
      }
      uint32_t size = align(push_set->limina_used_size, KK_MIN_UBO_ALIGNMENT);
      size = CLAMP(size, KK_MIN_UBO_ALIGNMENT, sizeof(push_set->data));
      struct kk_ptr push_gpu =
         kk_pool_upload(cmd, push_set->data, size, KK_MIN_UBO_ALIGNMENT);
      if (unlikely(!push_gpu.gpu))
         return;

      desc->root.sets[set_idx] = push_gpu.gpu;
      desc->set_sizes[set_idx] = size;
   }

   desc->root_dirty = true;
   desc->push_dirty = 0;
}

void
kk_dispatch_precomp(struct kk_cmd_buffer *cmd, struct kk_grid grid,
                    bool pre_gfx, enum libkk_program idx, void *data,
                    size_t data_size)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_precompiled_shader *prog = &dev->precompiled_cache.shaders[idx];

   mtl_compute_encoder *encoder = cs_get_compute(cmd, pre_gfx);
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);

   struct kk_ptr data_gpu = kk_pool_upload(cmd, data, data_size, 8u);
   if (unlikely(!data_gpu.gpu))
      return;

   mtl_set_address(cmd->argument_table, data_gpu.gpu, 0u);
   mtl_compute_set_pipeline_state(encoder, prog->pipeline);

   struct mtl_size local_size = {
      .x = prog->info.workgroup_size[0],
      .y = prog->info.workgroup_size[1],
      .z = prog->info.workgroup_size[2],
   };

   if (grid.mode == KK_GRID_DIRECT)
      mtl_dispatch_threads(encoder, grid.size, local_size);
   else
      mtl_dispatch_threadgroups_with_indirect_buffer(encoder, grid.addr,
                                                     local_size);
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);

   /* Rebind the exiting root. */
   mtl_set_address(cmd->argument_table, cmd->state.root_addr, 0u);
}

void
kk_cmd_write(struct kk_cmd_buffer *cmd, struct libkk_imm_write write)
{
   /* If we are mid render, it must go to post_gfx */
   libkk_write_u32(cmd, kk_grid_1d(1), !cmd->gfx.encoder, write.address,
                   write.value);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPushDescriptorSetWithTemplate2KHR(
   VkCommandBuffer commandBuffer, const VkPushDescriptorSetWithTemplateInfoKHR
                                     *pPushDescriptorSetWithTemplateInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  pPushDescriptorSetWithTemplateInfo->descriptorUpdateTemplate);
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout,
                  pPushDescriptorSetWithTemplateInfo->layout);

   struct kk_descriptor_state *desc =
      kk_get_descriptors_state(cmd, template->bind_point);
   struct kk_descriptor_set_layout *set_layout = vk_to_kk_descriptor_set_layout(
      pipeline_layout->set_layouts[pPushDescriptorSetWithTemplateInfo->set]);
   struct kk_push_descriptor_set *push_set = kk_cmd_push_descriptors(
      cmd, desc, set_layout, pPushDescriptorSetWithTemplateInfo->set);
   if (unlikely(push_set == NULL))
      return;

   kk_push_descriptor_set_update_template(
      push_set, set_layout, template,
      pPushDescriptorSetWithTemplateInfo->pData);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBeginConditionalRendering2EXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfo2EXT *pConditionalRenderingBegin)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   cmd->state.cond_render.address =
      pConditionalRenderingBegin->addressRange.address;
   cmd->state.cond_render.inverted = pConditionalRenderingBegin->flags &
                                     VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;
   cmd->state.cond_render.enabled = true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   cmd->state.cond_render.enabled = false;
}

void kk_apply_attachment_store_ops(struct kk_cmd_buffer *cmd, bool force_store)
{
   if (!cmd->gfx.encoder)
      return;

   struct kk_rendering_state *render = &cmd->state.gfx.render;
   mtl_render_encoder *encoder = cs_get_render(cmd);

   force_store |= render->force_attachment_store;

   /* LIMINA A/B lever, KK_LIMINA_FORCE_STORE=1: end every pass by storing its colour attachments
    * rather than discarding them. The mirror of KK_LIMINA_FORCE_LOAD, which was measured and did
    * not cure -- so loss at pass START is out, and loss at pass END is what is left. A cure here
    * names a store; no cure retires the store side as cleanly as the load side was retired.
    * Announces itself, because a lever whose engagement cannot be observed is worse than none. */
   {
      static int limina_force_store = -1;

      if (unlikely(limina_force_store < 0)) {
         const char *env = getenv("KK_LIMINA_FORCE_STORE");
         limina_force_store = env && strcmp(env, "0") ? 1 : 0;
         fprintf(stderr, "[LIMINA] KK pass-end store %s\n",
                 limina_force_store ? "FORCED to STORE on all attachments "
                                      "(KK_LIMINA_FORCE_STORE)"
                                    : "as encoded (default)");
         fflush(stderr);
      }
      force_store |= (bool)limina_force_store;
   }

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      uint32_t logical_index = cmd->state.gfx.render.color_map[i];

      if (render->color_att[i].iview && logical_index != MESA_VK_ATTACHMENT_UNUSED) {
         bool resolve = render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE;
         bool retain = (render->color_att[i].load_op == VK_ATTACHMENT_LOAD_OP_LOAD
            || render->color_att[i].load_op == VK_ATTACHMENT_LOAD_OP_NONE)
            && render->color_att[i].store_op == VK_ATTACHMENT_STORE_OP_NONE;

         enum mtl_store_action store_action = force_store
            || resolve
            || retain
            ? MTL_STORE_ACTION_STORE
            : vk_attachment_store_op_to_mtl_store_action(render->color_att[i].store_op);
         /* Count what the pass WOULD have done, not what the arm made it do -- counting the
          * forced result makes the number read 0 by construction whenever the arm is on, which
          * is exactly when someone is looking at it. */
         if (vk_attachment_store_op_to_mtl_store_action(render->color_att[i].store_op)
             == MTL_STORE_ACTION_DONT_CARE && !resolve && !retain) {
            const struct kk_image_view *siview = render->color_att[i].iview;

            p_atomic_inc(&kk_limina_counts.store_dontcare);
            if (siview->vk.extent.width <= 512u && siview->vk.extent.height <= 512u)
               p_atomic_inc(&kk_limina_counts.store_dontcare_small);
         }
         mtl_render_set_color_store_action(encoder, store_action, logical_index);
      }
   }
   if (render->depth_att.iview) {
      bool resolve = render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE;
      bool retain = (render->depth_att.load_op == VK_ATTACHMENT_LOAD_OP_LOAD ||
                     render->depth_att.load_op == VK_ATTACHMENT_LOAD_OP_NONE) &&
                    render->depth_att.store_op == VK_ATTACHMENT_STORE_OP_NONE;

      enum mtl_store_action store_action = force_store
            || resolve
            || retain
            ? MTL_STORE_ACTION_STORE
            : vk_attachment_store_op_to_mtl_store_action(render->depth_att.store_op);
      mtl_render_set_depth_store_action(encoder, store_action);
   }
   if (render->stencil_att.iview) {
      bool resolve = render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE;
      bool retain = (render->stencil_att.load_op == VK_ATTACHMENT_LOAD_OP_LOAD
         || render->stencil_att.load_op == VK_ATTACHMENT_LOAD_OP_NONE)
         && render->stencil_att.store_op == VK_ATTACHMENT_STORE_OP_NONE;

      enum mtl_store_action store_action = force_store
            || resolve
            || retain
            ? MTL_STORE_ACTION_STORE
            : vk_attachment_store_op_to_mtl_store_action(render->stencil_att.store_op);
      mtl_render_set_stencil_store_action(encoder, store_action);
   }
}
