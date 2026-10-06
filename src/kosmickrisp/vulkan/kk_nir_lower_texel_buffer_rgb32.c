/*
 * Copyright 2026 LunarG, Inc.
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: MIT
 */
#include "kk_private.h"

#include "kk_descriptor_types.h"
#include "kk_shader.h"

#include "nir.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"

#include "util/u_dynarray.h"

#include <stdbool.h>

/* A uniform texel buffer of R32G32B32_{SFLOAT,UINT,SINT} is a Metal texture
 * buffer of the channel format, three texels to an element, and its descriptor
 * says so with KK_TEXEL_BUFFER_RGB32 (see kk_CreateBufferView). The format is
 * a property of the view, not of the shader, so every buffer fetch and size
 * query asks the descriptor and takes the three-texel path when the flag is
 * set. Runs before descriptor lowering, which resolves the flags query. */

static nir_def *
rgb32_flag(nir_builder *b, nir_tex_instr *tex)
{
   nir_def *flags =
      nir_build_texture_query(b, tex, nir_texop_texel_buffer_flags_kk, 1,
                              nir_type_uint32, false, false);
   return nir_test_mask(b, flags, KK_TEXEL_BUFFER_RGB32);
}

static void
lower_txs(nir_builder *b, nir_tex_instr *tex)
{
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *rgb32 = rgb32_flag(b, tex);

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *size = &tex->def;
   nir_def *elements = nir_bcsel(b, rgb32, nir_udiv_imm(b, size, 3), size);
   nir_def_rewrite_uses_after(size, elements);
}

static nir_def *
fetch_channel(nir_builder *b, nir_tex_instr *tex, nir_def *index,
              unsigned channel)
{
   nir_def *texel = nir_iadd_imm(b, nir_imul_imm(b, index, 3), channel);

   /* Inserted before the rewrite: a clone's sources join their use lists
    * only on insertion. */
   nir_tex_instr *fetch =
      nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &fetch->instr);
   int coord = nir_tex_instr_src_index(fetch, nir_tex_src_coord);
   nir_src_rewrite(&fetch->src[coord].src, texel);
   return &fetch->def;
}

static void
lower_txf(nir_builder *b, nir_tex_instr *tex)
{
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *rgb32 = rgb32_flag(b, tex);
   nir_def *index = nir_get_tex_src(tex, nir_tex_src_coord);

   nir_def *emulated;
   nir_push_if(b, rgb32);
   {
      nir_def *texels[3];
      for (unsigned c = 0; c < 3; c++)
         texels[c] = fetch_channel(b, tex, index, c);

      /* A one-channel fetch returns (x, 0, 0, 1), so the first one also
       * carries the alpha of a three-channel format, in the fetch's type. */
      nir_def *comps[4];
      for (unsigned c = 0; c < tex->def.num_components; c++)
         comps[c] =
            c < 3 ? nir_channel(b, texels[c], 0) : nir_channel(b, texels[0], 3);
      emulated = nir_vec(b, comps, tex->def.num_components);
   }
   nir_push_else(b, NULL);
   {
      nir_instr_remove(&tex->instr);
      nir_builder_instr_insert(b, &tex->instr);
   }
   nir_pop_if(b, NULL);

   /* Every use but the phi's own: nir_def_rewrite_uses_after judges "after"
    * within a block, and the phi is in another one, so it would make the phi
    * its own else value and leave the fetch dead. */
   nir_def *texel = nir_if_phi(b, emulated, &tex->def);
   nir_foreach_use_including_if_safe(use, &tex->def) {
      if (!nir_src_is_if(use) && nir_src_use_instr(use) == nir_def_instr(texel))
         continue;
      nir_src_rewrite(use, texel);
   }
}

static bool
is_buffer_fetch_or_size(nir_instr *instr)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   return tex->sampler_dim == GLSL_SAMPLER_DIM_BUF &&
          (tex->op == nir_texop_txf || tex->op == nir_texop_txs) &&
          nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) >= 0;
}

bool
kk_nir_lower_texel_buffer_rgb32(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      /* Collected first: the fetch path adds control flow around the
       * instruction being lowered, which a walk of the blocks would trip on. */
      struct util_dynarray texs;
      util_dynarray_init(&texs, NULL);

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (is_buffer_fetch_or_size(instr))
               util_dynarray_append(&texs, nir_instr_as_tex(instr));
         }
      }

      nir_builder b = nir_builder_create(impl);
      util_dynarray_foreach(&texs, nir_tex_instr *, tex) {
         if ((*tex)->op == nir_texop_txs)
            lower_txs(&b, *tex);
         else
            lower_txf(&b, *tex);
      }

      bool impl_progress = util_dynarray_num_elements(&texs, nir_tex_instr *);
      util_dynarray_fini(&texs);
      progress |= nir_progress(impl_progress, impl, nir_metadata_none);
   }

   return progress;
}
