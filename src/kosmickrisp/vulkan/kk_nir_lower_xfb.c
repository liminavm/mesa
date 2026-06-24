/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * Transform feedback lowering: capture is implemented as vertex-shader
 * global stores into the bound transform feedback buffers. The per-draw
 * write base (bound address + current append offset - firstVertex * stride)
 * and an active mask live in the root descriptor table; the shader indexes
 * with the raw Metal vertex_id (which includes vertexStart).
 *
 * This is exact for the surface GLES3 permits while transform feedback is
 * active: non-indexed draws with list topologies (the API rejects indexed
 * draws and requires the draw mode to match primitiveMode). Desktop-GL-only
 * cases (strips/fans, indexed draws, instancing) capture one entry per VS
 * invocation instead of one per primitive vertex — best effort until a
 * compute-prepass path exists.
 */

#include "kk_cmd_buffer.h"
#include "kk_shader.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "util/bitset.h"

static void
lower_xfb_group(nir_builder *b, nir_intrinsic_instr *intr, unsigned start_comp,
                unsigned num_comps, unsigned buffer, unsigned offset_dw)
{
   nir_def *value = intr->src[0].ssa;
   unsigned base_comp = nir_intrinsic_component(intr);

   assert(start_comp >= base_comp);
   assert(value->bit_size == 32 &&
          "16-bit xfb capture not supported (mediump handled by upconvert)");

   /* Gather the captured channels from the stored value. */
   nir_def *chans[4];
   for (unsigned k = 0; k < num_comps; k++)
      chans[k] = nir_channel(b, value, start_comp - base_comp + k);
   nir_def *data = nir_vec(b, chans, num_comps);

   nir_def *argbuf = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);

   nir_def *mask = nir_load_global_constant(
      b, 1, 32,
      nir_iadd_imm(b, argbuf,
                   offsetof(struct kk_root_descriptor_table,
                            draw.xfb_active_mask)));
   nir_def *active =
      nir_ine_imm(b, nir_iand_imm(b, mask, BITFIELD_BIT(buffer)), 0);

   nir_push_if(b, active);
   {
      nir_def *base = nir_load_global_constant(
         b, 1, 64,
         nir_iadd_imm(b, argbuf,
                      offsetof(struct kk_root_descriptor_table,
                               draw.xfb_base[buffer])));

      /* slot = (instance_id - first_instance) * verts_per_instance +
       * vertex_id; the CPU pre-folds firstVertex into xfb_base. Metal's
       * instance_id includes base_instance. */
      nir_def *vpi = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, argbuf,
                      offsetof(struct kk_root_descriptor_table,
                               draw.xfb_verts_per_instance)));
      nir_def *first_inst = nir_load_global_constant(
         b, 1, 32,
         nir_iadd_imm(b, argbuf,
                      offsetof(struct kk_root_descriptor_table,
                               draw.xfb_first_instance)));
      nir_def *rel_inst =
         nir_isub(b, nir_load_instance_id(b), first_inst);
      nir_def *slot = nir_iadd(b, nir_imul(b, rel_inst, vpi),
                               nir_load_vertex_id(b));

      unsigned stride_B = b->shader->info.xfb_stride[buffer] * 4;
      nir_def *addr =
         nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, slot), stride_B));
      addr = nir_iadd_imm(b, addr, offset_dw * 4);

      nir_store_global(b, data, addr, .align_mul = 4);
   }
   nir_pop_if(b, NULL);

   BITSET_SET(b->shader->info.system_values_read, SYSTEM_VALUE_VERTEX_ID);
   BITSET_SET(b->shader->info.system_values_read, SYSTEM_VALUE_INSTANCE_ID);
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       !nir_intrinsic_has_io_xfb(intr))
      return false;

   /* The getter returns the full struct; out[] is indexed by the absolute
    * start component (io_xfb2 is just the storage for the second half). */
   nir_io_xfb xfb = nir_intrinsic_io_xfb(intr);

   bool progress = false;
   b->cursor = nir_before_instr(&intr->instr);

   for (unsigned c = 0; c < 4; c++) {
      unsigned num = xfb.out[c].num_components;
      if (!num)
         continue;

      lower_xfb_group(b, intr, c, num, xfb.out[c].buffer, xfb.out[c].offset);
      progress = true;
   }

   return progress;
}

bool
kk_nir_lower_xfb(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_VERTEX);

   return nir_shader_intrinsics_pass(nir, lower, nir_metadata_none, NULL);
}
