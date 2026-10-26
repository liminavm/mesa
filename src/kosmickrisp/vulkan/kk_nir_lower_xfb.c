/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 *
 * Transform feedback lowering: capture is implemented as vertex-shader
 * global stores into the bound transform feedback buffers. The per-draw
 * write base (bound address + current append offset) and an active mask live
 * in the root descriptor table.
 *
 * Transform feedback records one entry per primitive vertex, in the order
 * the topology defines, but an indexed or strip/fan/loop draw runs the vertex
 * shader once per index or per vertex. So while capture is on, the draw is
 * issued as a non-indexed list of primitive vertices from vertex 0
 * (kk_xfb_draw), and the shader maps its invocation back to the vertex it
 * stands for: the primitive and corner, the corner's position in the original
 * topology, and, for an indexed draw, the index stored there. Every
 * load_vertex_id (the vertex fetch included) and load_first_vertex then sees
 * the original draw's values. The rasterised corner order is rotated so the
 * provoking vertex comes first, which is the only convention Metal has, while
 * the capture slot keeps the API order. No compute pass is involved.
 */

#include "kk_cmd_buffer.h"
#include "kk_shader.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "util/bitset.h"

struct remap {
   nir_def *vertex_id;   /* the original draw's vertex index */
   nir_def *first_vertex; /* the original draw's firstVertex/vertexOffset */
   nir_def *slot;        /* the capture slot within one instance */
};

static nir_def *
load_root(nir_builder *b, unsigned bit_size, size_t offset)
{
   nir_def *argbuf = nir_load_buffer_ptr_kk(b, 1, 64, .binding = 0);
   return nir_load_global_constant(b, 1, bit_size,
                                   nir_iadd_imm(b, argbuf, offset));
}

#define ROOT(b, bits, field)                                                   \
   load_root(b, bits, offsetof(struct kk_root_descriptor_table, draw.field))

/* Position, in the original draw, of corner c of primitive t. The orders are
 * Vulkan's, with VK_EXT_provoking_vertex's last-vertex variants for strips
 * and fans; they match GL's transform feedback order. */
static nir_def *
topology_position(nir_builder *b, nir_def *prim, nir_def *last, nir_def *t,
                  nir_def *c, nir_def *vpp, nir_def *count)
{
   nir_def *odd = nir_iand_imm(b, t, 1);
   nir_def *list = nir_iadd(b, nir_imul(b, t, vpp), c);
   nir_def *strip = nir_iadd(b, t, c);
   nir_def *loop = nir_umod(b, strip, nir_umax(b, count, nir_imm_int(b, 1)));

   /* Triangle strip, first vertex provoking: (t, t+1+odd, t+2-odd). Last
    * vertex provoking: (t+odd, t+1-odd, t+2). */
   nir_def *c0 = nir_ieq_imm(b, c, 0), *c1 = nir_ieq_imm(b, c, 1);
   nir_def *tri_first =
      nir_bcsel(b, c0, t,
                nir_bcsel(b, c1, nir_iadd(b, nir_iadd_imm(b, t, 1), odd),
                          nir_isub(b, nir_iadd_imm(b, t, 2), odd)));
   nir_def *tri_last =
      nir_bcsel(b, c0, nir_iadd(b, t, odd),
                nir_bcsel(b, c1, nir_isub(b, nir_iadd_imm(b, t, 1), odd),
                          nir_iadd_imm(b, t, 2)));
   nir_def *tri_strip = nir_bcsel(b, last, tri_last, tri_first);

   /* Triangle fan, first vertex provoking: (t+1, t+2, 0). Last: (0, t+1, t+2). */
   nir_def *fan_first =
      nir_bcsel(b, nir_ieq_imm(b, c, 2), nir_imm_int(b, 0), nir_iadd(b, t, nir_iadd_imm(b, c, 1)));
   nir_def *fan_last = nir_bcsel(b, c0, nir_imm_int(b, 0), nir_iadd(b, t, c));
   nir_def *fan = nir_bcsel(b, last, fan_last, fan_first);

   nir_def *pos = list;
   pos = nir_bcsel(b, nir_ieq_imm(b, prim, MESA_PRIM_LINE_STRIP), strip, pos);
   pos = nir_bcsel(b, nir_ieq_imm(b, prim, MESA_PRIM_LINE_LOOP), loop, pos);
   pos = nir_bcsel(b, nir_ieq_imm(b, prim, MESA_PRIM_TRIANGLE_STRIP), tri_strip, pos);
   pos = nir_bcsel(b, nir_ieq_imm(b, prim, MESA_PRIM_TRIANGLE_FAN), fan, pos);
   return pos;
}

/* Index buffer entry i, read as aligned words so 8- and 16-bit indices need
 * no narrow loads. */
static nir_def *
load_index(nir_builder *b, nir_def *index_addr, nir_def *index_size,
           nir_def *i)
{
   nir_def *addr =
      nir_iadd(b, index_addr, nir_u2u64(b, nir_imul(b, i, index_size)));
   nir_def *word_addr = nir_iand_imm(b, addr, ~(uint64_t)3);
   nir_def *word = nir_load_global_constant(b, 1, 32, word_addr);
   nir_def *shift = nir_ishl_imm(b, nir_u2u32(b, nir_iand_imm(b, addr, 3)), 3);
   nir_def *value = nir_ushr(b, word, shift);
   nir_def *mask = nir_bcsel(b, nir_ieq_imm(b, index_size, 1),
                             nir_imm_int(b, 0xff),
                             nir_bcsel(b, nir_ieq_imm(b, index_size, 2),
                                       nir_imm_int(b, 0xffff),
                                       nir_imm_int(b, ~0)));
   return nir_iand(b, value, mask);
}

static struct remap
build_remap(nir_builder *b, nir_instr **raw_instr, nir_instr **raw_first_instr)
{
   nir_def *raw = nir_load_vertex_id(b);
   nir_def *raw_first = nir_load_first_vertex(b);
   *raw_instr = nir_def_instr(raw);
   *raw_first_instr = nir_def_instr(raw_first);
   nir_def *mode = ROOT(b, 32, xfb_remap);

   nir_push_if(b, nir_ine_imm(b, mode, 0));
   nir_def *vid, *first, *slot;
   {
      nir_def *prim = nir_iand_imm(b, mode, KK_XFB_REMAP_PRIM_MASK);
      nir_def *index_size =
         nir_ubitfield_extract_imm(b, mode, KK_XFB_REMAP_INDEX_SHIFT, 3);
      nir_def *last = nir_ine_imm(b, nir_iand_imm(b, mode, KK_XFB_REMAP_PROVOKE_LAST), 0);
      nir_def *count = ROOT(b, 32, xfb_remap_count);
      first = ROOT(b, 32, xfb_remap_base);

      nir_def *is_point = nir_ieq_imm(b, prim, MESA_PRIM_POINTS);
      nir_def *is_line = nir_ior(b, nir_ieq_imm(b, prim, MESA_PRIM_LINES),
                                 nir_ior(b, nir_ieq_imm(b, prim, MESA_PRIM_LINE_STRIP),
                                         nir_ieq_imm(b, prim, MESA_PRIM_LINE_LOOP)));
      nir_def *vpp = nir_bcsel(b, is_point, nir_imm_int(b, 1),
                               nir_bcsel(b, is_line, nir_imm_int(b, 2),
                                         nir_imm_int(b, 3)));

      /* raw is the invocation's place in the issued list: primitive t,
       * rasterised corner k. Metal provokes from corner 0, so with the last
       * vertex provoking, corner k draws API corner (k + vpp - 1) % vpp: a
       * rotation, which keeps the winding. */
      nir_def *t = nir_udiv(b, raw, vpp);
      nir_def *k = nir_isub(b, raw, nir_imul(b, t, vpp));
      nir_def *c = nir_bcsel(b, last,
                             nir_umod(b, nir_iadd(b, k, nir_iadd_imm(b, vpp, -1)), vpp),
                             k);
      slot = nir_iadd(b, nir_imul(b, t, vpp), c);

      nir_def *pos = topology_position(b, prim, last, t, c, vpp, count);

      nir_push_if(b, nir_ine_imm(b, index_size, 0));
      nir_def *indexed;
      {
         nir_def *index_addr = ROOT(b, 64, xfb_remap_index_addr);
         indexed = nir_iadd(b, load_index(b, index_addr, index_size, pos), first);
      }
      nir_push_else(b, NULL);
      nir_def *direct = nir_iadd(b, pos, first);
      nir_pop_if(b, NULL);
      vid = nir_if_phi(b, indexed, direct);
   }
   nir_push_else(b, NULL);
   nir_pop_if(b, NULL);

   struct remap r = {
      .vertex_id = nir_if_phi(b, vid, raw),
      .first_vertex = nir_if_phi(b, first, raw_first),
      .slot = nir_if_phi(b, slot, raw),
   };
   return r;
}

static void
lower_xfb_group(nir_builder *b, nir_intrinsic_instr *intr, unsigned start_comp,
                unsigned num_comps, unsigned buffer, unsigned offset_dw,
                nir_def *slot_in_instance)
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

   /* slot = (instance_id - first_instance) * verts_per_instance + the
    * vertex's slot in its instance. Metal's instance_id includes
    * base_instance. */
   nir_def *vpi = ROOT(b, 32, xfb_verts_per_instance);
   nir_def *first_inst = ROOT(b, 32, xfb_first_instance);
   nir_def *rel_inst = nir_isub(b, nir_load_instance_id(b), first_inst);
   nir_def *slot = nir_iadd(b, nir_imul(b, rel_inst, vpi), slot_in_instance);

   nir_def *mask = ROOT(b, 32, xfb_active_mask);
   nir_def *active =
      nir_iand(b, nir_ine_imm(b, nir_iand_imm(b, mask, BITFIELD_BIT(buffer)), 0),
               nir_ult(b, slot, ROOT(b, 32, xfb_slot_limit)));

   nir_push_if(b, active);
   {
      nir_def *base = load_root(
         b, 64,
         offsetof(struct kk_root_descriptor_table, draw.xfb_base) +
            buffer * sizeof(uint64_t));

      unsigned stride_B = b->shader->info.xfb_stride[buffer] * 4;
      nir_def *addr =
         nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, slot), stride_B));
      addr = nir_iadd_imm(b, addr, offset_dw * 4);

      nir_store_global(b, data, addr, .align_mul = 4);
   }
   nir_pop_if(b, NULL);
}

struct lower_state {
   struct remap remap;
   nir_instr *raw_vertex_id;
   nir_instr *raw_first_vertex;
   uint64_t dropped; /* slots that had an xfb-only store removed */
};

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct lower_state *state = data;

   if ((intr->intrinsic == nir_intrinsic_load_vertex_id &&
        &intr->instr != state->raw_vertex_id) ||
       (intr->intrinsic == nir_intrinsic_load_first_vertex &&
        &intr->instr != state->raw_first_vertex)) {
      /* Every load but the raw pair build_remap reads from. */
      nir_def *repl = intr->intrinsic == nir_intrinsic_load_vertex_id
                         ? state->remap.vertex_id
                         : state->remap.first_vertex;
      nir_def_replace(&intr->def, repl);
      return true;
   }

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

      lower_xfb_group(b, intr, c, num, xfb.out[c].buffer, xfb.out[c].offset,
                      state->remap.slot);
      progress = true;
   }

   /* Capture is done; an output only transform feedback consumed is no
    * varying at all. Left in, nir_opt_varyings may have packed it into the
    * free components of a slot the fragment shader reads with another type,
    * and Metal refuses to link that pair. */
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.no_varying && sem.location >= VARYING_SLOT_VAR0) {
      state->dropped |= BITFIELD64_BIT(sem.location);
      nir_instr_remove(&intr->instr);
      progress = true;
   }

   return progress;
}

static bool
gather_written(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_store_output)
      *(uint64_t *)data |=
         BITFIELD64_BIT(nir_intrinsic_io_semantics(intr).location);
   return false;
}

bool
kk_nir_lower_xfb(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_VERTEX);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   struct lower_state state = {0};
   state.remap = build_remap(&b, &state.raw_vertex_id, &state.raw_first_vertex);

   nir_shader_intrinsics_pass(nir, lower, nir_metadata_none, &state);

   /* A slot that lost its last store is no longer an output. */
   if (state.dropped) {
      uint64_t written = 0;
      nir_shader_intrinsics_pass(nir, gather_written, nir_metadata_all, &written);
      nir->info.outputs_written &= ~(state.dropped & ~written);
   }

   BITSET_SET(nir->info.system_values_read, SYSTEM_VALUE_VERTEX_ID);
   BITSET_SET(nir->info.system_values_read, SYSTEM_VALUE_FIRST_VERTEX);
   BITSET_SET(nir->info.system_values_read, SYSTEM_VALUE_INSTANCE_ID);
   return true;
}
