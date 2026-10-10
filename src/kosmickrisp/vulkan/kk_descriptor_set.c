/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_descriptor_set.h"

#include "kk_bo.h"
#include "kk_buffer.h"
#include "kk_buffer_view.h"
#include "kk_descriptor_set_layout.h"
#include "kk_device.h"
#include "kk_entrypoints.h"
#include "kk_image_view.h"
#include "kk_physical_device.h"
#include "kk_sampler.h"

#include "kosmickrisp/bridge/mtl_bridge.h"

#include "util/format/u_format.h"
#include "util/half_float.h"

static inline uint32_t
align_u32(uint32_t v, uint32_t a)
{
   assert(a != 0 && a == (a & -a));
   return (v + a - 1) & ~(a - 1);
}

/* The slot a descriptor lives in, or NULL with *size_out 0 when (binding, elem)
 * leaves the set. A binding past the layout would index set->layout->binding[]
 * out of bounds; an element whose byte offset reaches set->size would land past
 * the buffer; a dynamic-only set has no buffer at all (size 0, mapped_ptr
 * NULL). Under venus a guest drives the write, so this forms no out-of-range
 * pointer and every caller skips a NULL. *size_out is the bytes left to the end
 * of the set, which the caller checks its write against. */
static inline void *
desc_ubo_data(struct kk_descriptor_set *set, uint32_t binding, uint32_t elem,
              uint32_t *size_out)
{
   if (size_out != NULL)
      *size_out = 0;

   if (binding >= set->layout->binding_count || set->mapped_ptr == NULL)
      return NULL;

   const struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   uint64_t offset = (uint64_t)binding_layout->offset +
                     (uint64_t)elem * binding_layout->stride;
   if (offset >= set->size)
      return NULL;

   if (size_out != NULL)
      *size_out = set->size - (uint32_t)offset;

   return (char *)set->mapped_ptr + offset;
}

static void
write_desc(struct kk_descriptor_set *set, uint32_t binding, uint32_t elem,
           const void *desc_data, size_t desc_size)
{
   uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   if (dst == NULL || desc_size > dst_size) {
      mesa_loge("kk: refusing a %zu-byte descriptor write to binding %u "
                "element %u, outside the set's buffer",
                desc_size, binding, elem);
      return;
   }
   memcpy(dst, desc_data, desc_size);
}

/* Whether an update of `count` descriptors of `type` at (binding, elem) stays
 * inside what the set's layout and buffer hold. desc_ubo_data() indexes
 * set->layout->binding[binding] and offsets into the set's buffer by
 * elem*stride, and the dynamic-buffer writer indexes
 * set->dynamic_buffers[dynamic_buffer_index + elem]; a binding, element or
 * count past those overruns. The write's type must match the binding's (or a
 * mutable binding whose stride holds it), or a write would size a
 * differently-typed descriptor by the binding's stride -- including a write
 * onto a stride-0 dynamic binding whose buffer may be absent. For an inline
 * uniform block elem and count are a byte offset and size. Under venus a guest
 * drives writes, copies and templates unchecked, so each entry runs this
 * first. dyn_capacity is the dynamic-buffer array's length: the layout's
 * dynamic_descriptor_count for a pool set, 0 for a push set whose on-stack
 * dynamic_buffers[] is empty. `type` must be one the entry points admit, so the
 * mutable stride lookup never reaches an unknown type. */
static bool
kk_descriptor_update_in_bounds(const struct kk_descriptor_set *set,
                               uint32_t dyn_capacity, VkDescriptorType type,
                               uint32_t binding, uint32_t elem, uint32_t count)
{
   const struct kk_descriptor_set_layout *layout = set->layout;
   if (binding >= layout->binding_count)
      return false;
   const struct kk_descriptor_set_binding_layout *bl =
      &layout->binding[binding];

   if ((uint64_t)elem + count > bl->array_size)
      return false;

   const bool is_inline = type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
   const bool is_dynamic = type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                           type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;

   if (type != bl->type) {
      /* A mutable binding holds any admitted non-inline, non-dynamic type whose
       * descriptor fits its stride; KK keeps no type list, only the stride. */
      if (bl->type != VK_DESCRIPTOR_TYPE_MUTABLE_EXT || is_inline || is_dynamic)
         return false;
      uint32_t stride = 0, align = 0;
      kk_descriptor_stride_align_for_type(type, NULL, &stride, &align);
      if (stride == 0 || stride > bl->stride)
         return false;
   }

   if (is_dynamic)
      return (uint64_t)bl->dynamic_buffer_index + elem + count <= dyn_capacity;

   /* inline has stride 1, so one byte-range formula covers both. */
   return (uint64_t)bl->offset + ((uint64_t)elem + count) * bl->stride <=
          set->size;
}

/* A VkWriteDescriptorSet's slots, bounded; count is descriptorCount, or the
 * inline block's dataSize (both in the units kk_descriptor_update_in_bounds
 * expects). */
static bool
kk_write_in_bounds(const struct kk_descriptor_set *set, uint32_t dyn_capacity,
                   const VkWriteDescriptorSet *write, uint32_t count)
{
   if (!kk_descriptor_update_in_bounds(set, dyn_capacity, write->descriptorType,
                                       write->dstBinding,
                                       write->dstArrayElement, count)) {
      mesa_loge(
         "kk: refusing a descriptor write of %u to binding %u element %u",
         count, write->dstBinding, write->dstArrayElement);
      return false;
   }
   return true;
}

/* Whether `count` slots of a set's own binding `binding` starting at `elem`
 * stay inside it. Like kk_descriptor_update_in_bounds, but for a copy, which
 * carries no incoming type -- each binding keeps its own, so there is no type
 * to match. A dynamic binding is bounded by the dynamic-buffer array, any other
 * by the descriptor buffer. */
static bool
kk_descriptor_slots_in_bounds(const struct kk_descriptor_set *set,
                              uint32_t binding, uint32_t elem, uint32_t count)
{
   if (binding >= set->layout->binding_count)
      return false;
   const struct kk_descriptor_set_binding_layout *bl =
      &set->layout->binding[binding];
   if ((uint64_t)elem + count > bl->array_size)
      return false;
   if (bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
       bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
      return (uint64_t)bl->dynamic_buffer_index + elem + count <=
             set->layout->vk.dynamic_descriptor_count;
   return (uint64_t)bl->offset + ((uint64_t)elem + count) * bl->stride <=
          set->size;
}

/* limina: LIMINA_KK_DESCLOG -- report each distinct descriptor SLOT, not each
 * distinct type tuple. Metal's validator says a 2D-array texture reached a shader
 * that declared a 2D one; deduping by (descriptor type, view type, Metal type) can
 * only say whether some write was internally consistent, which every write is. The
 * question is which slot a given resource id landed in, so the key is the slot plus
 * the texture's identity, and the line carries the id the shader will dereference. */
static void
kk_limina_desclog(const struct kk_descriptor_set *set, uint32_t binding,
                  uint32_t elem, VkDescriptorType type,
                  const struct kk_image_view *view, uint8_t plane,
                  bool is_input_attachment, uint64_t res_id)
{
   static int on = -1;
   if (on < 0)
      on = getenv("LIMINA_KK_DESCLOG") != NULL;
   if (!on)
      return;

   mtl_texture *tex = is_input_attachment ? view->planes[plane].mtl_handle_input
                                          : view->planes[plane].mtl_handle_sampled;
   if (tex == NULL)
      return;

   struct mtl_texture_props props = {0};
   mtl_texture_get_props(tex, &props);

   uint64_t key = (uint64_t)binding * 0x9e3779b97f4a7c15ull;
   key ^= (uint64_t)elem * 0xc2b2ae3d27d4eb4full;
   key ^= (uint64_t)type << 3;
   key ^= (uint64_t)view->vk.view_type << 11;
   key ^= (uint64_t)props.texture_type << 17;
   key ^= (uint64_t)props.sample_count << 23;
   key ^= (uint64_t)props.array_length << 29;
   key ^= (uint64_t)props.width << 37;
   key ^= (uint64_t)props.height << 45;
   key ^= (uint64_t)is_input_attachment << 61;

   static uint64_t seen[1024];
   static unsigned seen_n;
   for (unsigned i = 0; i < seen_n; i++)
      if (seen[i] == key)
         return;
   if (seen_n < ARRAY_SIZE(seen))
      seen[seen_n++] = key;

   fprintf(stderr,
           "[LIMINA-KK-DESC] set=%p binding=%u elem=%u desc_type=%u input=%d "
           "view=%p vk_view_type=%u layers=%u-%u -> id=0x%llx mtl_type=%u "
           "samples=%u tex_layers=%u %llux%llu\n",
           (void *)set, binding, elem, (unsigned)type, (int)is_input_attachment,
           (void *)view, (unsigned)view->vk.view_type,
           (unsigned)view->vk.base_array_layer, (unsigned)view->vk.layer_count,
           (unsigned long long)res_id, props.texture_type, props.sample_count,
           props.array_length, (unsigned long long)props.width,
           (unsigned long long)props.height);
   fflush(stderr);
}

static void
get_sampled_image_view_desc(const struct kk_descriptor_set *set,
                            uint32_t binding, uint32_t elem,
                            VkDescriptorType descriptor_type,
                            const VkDescriptorImageInfo *const info, void *dst,
                            size_t dst_size, bool is_input_attachment)
{
   struct kk_sampled_image_descriptor desc[3] = {};
   uint8_t plane_count = 1;

   if (descriptor_type != VK_DESCRIPTOR_TYPE_SAMPLER && info &&
       info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_image_view, view, info->imageView);

      plane_count = view->plane_count;
      for (uint8_t plane = 0; plane < plane_count; plane++) {
         if (is_input_attachment) {
            assert(view->planes[plane].sampled_gpu_resource_id);
            desc[plane].image_gpu_resource_id =
               view->planes[plane].input_gpu_resource_id;
         } else {
            assert(view->planes[plane].sampled_gpu_resource_id);
            desc[plane].image_gpu_resource_id =
               view->planes[plane].sampled_gpu_resource_id;
         }

         float min_lod = MAX2(view->vk.min_lod - view->vk.base_mip_level, 0.0);
         desc[plane].image_min_lod_fp16 = _mesa_float_to_half(min_lod);
         desc[plane].image_min_lod_uint16 = min_lod;
         kk_limina_desclog(set, binding, elem, descriptor_type, view, plane,
                           is_input_attachment,
                           desc[plane].image_gpu_resource_id);
      }
   }

   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      VK_FROM_HANDLE(kk_sampler, sampler, info->sampler);
      if (sampler == NULL) {
         mesa_loge("kk: refusing a sampler descriptor write with no sampler to "
                   "binding %u element %u",
                   binding, elem);
         return;
      }

      if (sampler->has_border)
         assert(plane_count == 1);
      else
         plane_count = MAX2(plane_count, sampler->plane_count);

      for (uint8_t plane = 0; plane < plane_count; plane++) {
         /* We need to replicate the last sampler plane out to all image
          * planes due to sampler table entry limitations. See
          * nvk_CreateSampler in nvk_sampler.c for more details.
          */
         uint8_t sampler_plane = MIN2(plane, sampler->plane_count - 1u);
         assert(sampler->planes[sampler_plane].hw->handle);
         desc[plane].sampler_index = sampler->planes[sampler_plane].hw->index;
         desc[plane].sampler_lod_bias_fp16 = sampler->lod_bias_fp16;
         desc[plane].sampler_lod_min_fp16 = sampler->lod_min_fp16;
         desc[plane].sampler_lod_max_fp16 = sampler->lod_max_fp16;
         desc[plane].clamp_0_sampler_index_or_negative = -1;
      }

      if (sampler->has_border) {
         assert(sampler->plane_count == 2);
         desc[0].clamp_0_sampler_index_or_negative =
            sampler->planes[1].hw->index;

         assert(desc[0].clamp_0_sampler_index_or_negative >= 0 &&
                "we have a border colour");

         static_assert(sizeof(desc[0].border) == sizeof(sampler->custom_border),
                       "fixed format");

         memcpy(desc[0].border, sampler->custom_border.uint32,
                sizeof(sampler->custom_border));
      }
   }

   if (dst == NULL || sizeof(desc[0]) * plane_count > dst_size) {
      mesa_loge("kk: refusing a %u-plane image descriptor write to binding %u "
                "element %u, outside the set's buffer",
                plane_count, binding, elem);
      return;
   }
   memcpy(dst, desc, sizeof(desc[0]) * plane_count);
}

static void
write_sampled_image_view_desc(struct kk_descriptor_set *set,
                              const VkDescriptorImageInfo *const _info,
                              uint32_t binding, uint32_t elem,
                              VkDescriptorType descriptor_type)
{
   VkDescriptorImageInfo info = *_info;

   struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];
   if (descriptor_type == VK_DESCRIPTOR_TYPE_SAMPLER ||
       descriptor_type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
      if (binding_layout->immutable_samplers != NULL) {
         info.sampler =
            kk_sampler_to_handle(binding_layout->immutable_samplers[elem]);
      }
   }

   uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   get_sampled_image_view_desc(
      set, binding, elem, descriptor_type, &info, dst, dst_size,
      descriptor_type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);
}

static void
get_storage_image_view_desc(
   struct kk_descriptor_set_binding_layout *binding_layout,
   const VkDescriptorImageInfo *const info, void *dst, size_t dst_size)
{
   struct kk_storage_image_descriptor desc = {};

   if (info && info->imageView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_image_view, view, info->imageView);

      /* Storage images are always single plane */
      assert(view->plane_count == 1);
      uint8_t plane = 0;

      assert(view->planes[plane].storage_gpu_resource_id);
      desc.image_gpu_resource_id = view->planes[plane].storage_gpu_resource_id;
   }

   if (dst == NULL || sizeof(desc) > dst_size) {
      mesa_loge("kk: refusing a storage image descriptor write, outside the "
                "set's buffer");
      return;
   }
   memcpy(dst, &desc, sizeof(desc));
}

static void
write_storage_image_view_desc(struct kk_descriptor_set *set,
                              const VkDescriptorImageInfo *const info,
                              uint32_t binding, uint32_t elem)
{
   uint32_t dst_size;
   void *dst = desc_ubo_data(set, binding, elem, &dst_size);
   struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];
   get_storage_image_view_desc(binding_layout, info, dst, dst_size);
}

static void
write_buffer_desc(struct kk_descriptor_set *set,
                  const VkDescriptorBufferInfo *const info, uint32_t binding,
                  uint32_t elem)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);

   const struct kk_addr_range addr_range =
      kk_buffer_addr_range(buffer, info->offset, info->range);
   assert(addr_range.range <= UINT32_MAX);

   const struct kk_buffer_address desc = {
      .base_addr = addr_range.addr,
      .size = addr_range.range,
   };
   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static void
write_dynamic_buffer_desc(struct kk_descriptor_set *set,
                          const VkDescriptorBufferInfo *const info,
                          uint32_t binding, uint32_t elem)
{
   VK_FROM_HANDLE(kk_buffer, buffer, info->buffer);
   const struct kk_descriptor_set_binding_layout *binding_layout =
      &set->layout->binding[binding];

   const struct kk_addr_range addr_range =
      kk_buffer_addr_range(buffer, info->offset, info->range);
   assert(addr_range.range <= UINT32_MAX);

   struct kk_buffer_address *desc =
      &set->dynamic_buffers[binding_layout->dynamic_buffer_index + elem];
   *desc = (struct kk_buffer_address){
      .base_addr = addr_range.addr,
      .size = addr_range.range,
   };
}

static void
write_buffer_view_desc(struct kk_descriptor_set *set,
                       const VkBufferView bufferView, uint32_t binding,
                       uint32_t elem)
{
   struct kk_texel_buffer_descriptor desc = {};
   if (bufferView != VK_NULL_HANDLE) {
      VK_FROM_HANDLE(kk_buffer_view, view, bufferView);

      assert(view->mtl_texel_buffer_handle);
      assert(view->texel_buffer_gpu_id);

      desc.image_gpu_resource_id = view->texel_buffer_gpu_id;
      desc.flags = view->texel_buffer_flags;
   }
   write_desc(set, binding, elem, &desc, sizeof(desc));
}

static void
write_inline_uniform_data(struct kk_descriptor_set *set,
                          const VkWriteDescriptorSetInlineUniformBlock *info,
                          uint32_t binding, uint32_t offset)
{
   write_desc(set, binding, offset, info->pData, info->dataSize);
}

VKAPI_ATTR void VKAPI_CALL
kk_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                        const VkWriteDescriptorSet *pDescriptorWrites,
                        uint32_t descriptorCopyCount,
                        const VkCopyDescriptorSet *pDescriptorCopies)
{
   for (uint32_t w = 0; w < descriptorWriteCount; w++) {
      const VkWriteDescriptorSet *write = &pDescriptorWrites[w];
      VK_FROM_HANDLE(kk_descriptor_set, set, write->dstSet);
      if (set == NULL)
         continue;
      const uint32_t dyn_capacity = set->layout->vk.dynamic_descriptor_count;

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         if (!kk_write_in_bounds(set, dyn_capacity, write,
                                 write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         if (!kk_write_in_bounds(set, dyn_capacity, write,
                                 write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         if (!kk_write_in_bounds(set, dyn_capacity, write,
                                 write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         if (!kk_write_in_bounds(set, dyn_capacity, write,
                                 write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(set, write->pBufferInfo + j, write->dstBinding,
                              write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         if (!kk_write_in_bounds(set, dyn_capacity, write,
                                 write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_dynamic_buffer_desc(set, write->pBufferInfo + j,
                                      write->dstBinding,
                                      write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
         const VkWriteDescriptorSetInlineUniformBlock *write_inline =
            vk_find_struct_const(write->pNext,
                                 WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
         if (write_inline == NULL)
            break;
         if (!kk_write_in_bounds(set, dyn_capacity, write,
                                 write_inline->dataSize))
            break;
         write_inline_uniform_data(set, write_inline, write->dstBinding,
                                   write->dstArrayElement);
         break;
      }

      default:
         break;
      }
   }

   for (uint32_t i = 0; i < descriptorCopyCount; i++) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      VK_FROM_HANDLE(kk_descriptor_set, src, copy->srcSet);
      VK_FROM_HANDLE(kk_descriptor_set, dst, copy->dstSet);
      if (src == NULL || dst == NULL)
         continue;

      if (!kk_descriptor_slots_in_bounds(src, copy->srcBinding,
                                         copy->srcArrayElement,
                                         copy->descriptorCount) ||
          !kk_descriptor_slots_in_bounds(dst, copy->dstBinding,
                                         copy->dstArrayElement,
                                         copy->descriptorCount)) {
         mesa_loge("kk: refusing a descriptor copy of %u from binding %u "
                   "element %u to binding %u element %u, outside a set",
                   copy->descriptorCount, copy->srcBinding,
                   copy->srcArrayElement, copy->dstBinding,
                   copy->dstArrayElement);
         continue;
      }

      const struct kk_descriptor_set_binding_layout *src_binding_layout =
         &src->layout->binding[copy->srcBinding];
      const struct kk_descriptor_set_binding_layout *dst_binding_layout =
         &dst->layout->binding[copy->dstBinding];

      if (dst_binding_layout->stride > 0 && src_binding_layout->stride > 0) {
         for (uint32_t j = 0; j < copy->descriptorCount; j++) {
            uint32_t dst_max_size, src_max_size;
            void *dst_map = desc_ubo_data(
               dst, copy->dstBinding, copy->dstArrayElement + j, &dst_max_size);
            const void *src_map = desc_ubo_data(
               src, copy->srcBinding, copy->srcArrayElement + j, &src_max_size);
            const uint32_t copy_size =
               MIN2(dst_binding_layout->stride, src_binding_layout->stride);
            if (dst_map == NULL || src_map == NULL ||
                copy_size > dst_max_size || copy_size > src_max_size)
               continue;
            memcpy(dst_map, src_map, copy_size);
         }
      }

      /* Dynamic buffers live in a separate array; copy them only when both
       * bindings are dynamic -- the slot check above bounded each within its
       * own dynamic_descriptor_count. */
      const bool src_dynamic =
         src_binding_layout->type ==
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
         src_binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      const bool dst_dynamic =
         dst_binding_layout->type ==
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
         dst_binding_layout->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      if (src_dynamic && dst_dynamic) {
         const uint32_t dst_dyn_start =
            dst_binding_layout->dynamic_buffer_index + copy->dstArrayElement;
         const uint32_t src_dyn_start =
            src_binding_layout->dynamic_buffer_index + copy->srcArrayElement;
         typed_memcpy(&dst->dynamic_buffers[dst_dyn_start],
                      &src->dynamic_buffers[src_dyn_start],
                      copy->descriptorCount);
      }
   }
}

void
kk_push_descriptor_set_update(struct kk_push_descriptor_set *push_set,
                              uint32_t write_count,
                              const VkWriteDescriptorSet *writes)
{
   /* The push set's buffer is a fixed sizeof(push_set->data); a layout larger
    * than it, or an element past it, is bounded per write below (dyn_capacity 0
    * -- a push set has no dynamic-buffer array). */
   struct kk_descriptor_set set = {
      .layout = push_set->layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };

   for (uint32_t w = 0; w < write_count; w++) {
      const VkWriteDescriptorSet *write = &writes[w];

      switch (write->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         if (!kk_write_in_bounds(&set, 0, write, write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_sampled_image_view_desc(
               &set, write->pImageInfo + j, write->dstBinding,
               write->dstArrayElement + j, write->descriptorType);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         if (!kk_write_in_bounds(&set, 0, write, write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_storage_image_view_desc(&set, write->pImageInfo + j,
                                          write->dstBinding,
                                          write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         if (!kk_write_in_bounds(&set, 0, write, write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_view_desc(&set, write->pTexelBufferView[j],
                                   write->dstBinding,
                                   write->dstArrayElement + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         if (!kk_write_in_bounds(&set, 0, write, write->descriptorCount))
            break;
         for (uint32_t j = 0; j < write->descriptorCount; j++) {
            write_buffer_desc(&set, write->pBufferInfo + j, write->dstBinding,
                              write->dstArrayElement + j);
         }
         break;

      default:
         break;
      }
   }
}

static void kk_descriptor_pool_free(struct kk_descriptor_pool *pool,
                                    uint64_t addr, uint64_t size);

void
kk_descriptor_pool_set_label(struct kk_descriptor_pool *pool, const char *label)
{
   if (pool->bo)
      kk_bo_set_label(pool->bo, label);
}

static void
kk_descriptor_set_destroy(struct kk_device *dev,
                          struct kk_descriptor_pool *pool,
                          struct kk_descriptor_set *set)
{
   list_del(&set->link);
   if (set->size > 0)
      kk_descriptor_pool_free(pool, set->addr, set->size);
   vk_descriptor_set_layout_unref(&dev->vk, &set->layout->vk);

   vk_object_free(&dev->vk, NULL, set);
}

static void
kk_destroy_descriptor_pool(struct kk_device *dev,
                           const VkAllocationCallbacks *pAllocator,
                           struct kk_descriptor_pool *pool)
{
   list_for_each_entry_safe(struct kk_descriptor_set, set, &pool->sets, link)
      kk_descriptor_set_destroy(dev, pool, set);

   util_vma_heap_finish(&pool->heap);

   if (pool->bo != NULL)
      kk_destroy_bo(dev, pool->bo);

   vk_object_free(&dev->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_CreateDescriptorPool(VkDevice _device,
                        const VkDescriptorPoolCreateInfo *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(kk_device, dev, _device);
   struct kk_descriptor_pool *pool;
   VkResult result = VK_SUCCESS;

   pool = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   list_inithead(&pool->sets);

   /* Initialise the heap empty up front so every failure path below can run
    * kk_destroy_descriptor_pool (which calls util_vma_heap_finish) safely. The
    * pool came from vk_object_zalloc, so without this the heap's hole list is
    * NULL and finish walks it into a SIGSEGV when kk_alloc_bo fails. */
   util_vma_heap_init(&pool->heap, 0, 0);

   const VkMutableDescriptorTypeCreateInfoEXT *mutable_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT);

   uint32_t max_align = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      kk_descriptor_stride_align_for_type(pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      max_align = MAX2(max_align, alignment);
   }

   uint64_t mem_size = 0;
   for (unsigned i = 0; i < pCreateInfo->poolSizeCount; ++i) {
      const VkMutableDescriptorTypeListEXT *type_list = NULL;
      if (pCreateInfo->pPoolSizes[i].type == VK_DESCRIPTOR_TYPE_MUTABLE_EXT &&
          mutable_info && i < mutable_info->mutableDescriptorTypeListCount)
         type_list = &mutable_info->pMutableDescriptorTypeLists[i];

      uint32_t stride, alignment;
      kk_descriptor_stride_align_for_type(pCreateInfo->pPoolSizes[i].type,
                                          type_list, &stride, &alignment);
      mem_size += (uint64_t)MAX2(stride, max_align) *
                  pCreateInfo->pPoolSizes[i].descriptorCount;
   }

   /* Individual descriptor sets are aligned to the min UBO alignment to
    * ensure that we don't end up with unaligned data access in any shaders.
    * This means that each descriptor buffer allocated may burn up to 16B of
    * extra space to get the right alignment.  (Technically, it's at most 28B
    * because we're always going to start at least 4B aligned but we're being
    * conservative here.)  Allocate enough extra space that we can chop it
    * into maxSets pieces and align each one of them to 32B.
    */
   mem_size += (uint64_t)kk_min_cbuf_alignment() * pCreateInfo->maxSets;

   if (mem_size) {
      result = kk_alloc_bo(dev, &dev->vk.base, mem_size, 0u, &pool->bo);
      if (result != VK_SUCCESS) {
         kk_destroy_descriptor_pool(dev, pAllocator, pool);
         return result;
      }

      /* The BO may be larger thanks to GPU page alignment.  We may as well
       * make that extra space available to the client.  The empty heap set up
       * above is re-initialised here with the real range; it holds no holes
       * yet, so nothing leaks.
       */
      assert(pool->bo->size_B >= mem_size);
      util_vma_heap_init(&pool->heap, pool->bo->gpu, pool->bo->size_B);
   }

   *pDescriptorPool = kk_descriptor_pool_to_handle(pool);
   return result;
}

static VkResult
kk_descriptor_pool_alloc(struct kk_descriptor_pool *pool, uint64_t size,
                         uint64_t alignment, uint64_t *addr_out, void **map_out)
{
   assert(size > 0);
   assert(size % alignment == 0);

   if (size > pool->heap.free_size)
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   uint64_t addr = util_vma_heap_alloc(&pool->heap, size, alignment);
   if (addr == 0)
      return VK_ERROR_FRAGMENTED_POOL;

   assert(addr >= pool->bo->gpu);
   assert(addr + size <= pool->bo->gpu + pool->bo->size_B);
   uint64_t offset = addr - pool->bo->gpu;

   *addr_out = addr;
   *map_out = pool->bo->cpu + offset;

   return VK_SUCCESS;
}

static void
kk_descriptor_pool_free(struct kk_descriptor_pool *pool, uint64_t addr,
                        uint64_t size)
{
   assert(size > 0);
   assert(addr >= pool->bo->gpu);
   assert(addr + size <= pool->bo->gpu + pool->bo->size_B);
   util_vma_heap_free(&pool->heap, addr, size);
}

static VkResult
kk_descriptor_set_create(struct kk_device *dev, struct kk_descriptor_pool *pool,
                         struct kk_descriptor_set_layout *layout,
                         uint32_t variable_count,
                         struct kk_descriptor_set **out_set)
{
   struct kk_descriptor_set *set;
   VkResult result = VK_SUCCESS;

   uint32_t mem_size =
      sizeof(struct kk_descriptor_set) +
      layout->vk.dynamic_descriptor_count * sizeof(struct kk_buffer_address);
   set =
      vk_object_zalloc(&dev->vk, NULL, mem_size, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   set->size = layout->non_variable_descriptor_buffer_size;

   if (layout->binding_count > 0 &&
       (layout->binding[layout->binding_count - 1].flags &
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)) {
      uint32_t stride = layout->binding[layout->binding_count - 1].stride;
      set->size += stride * variable_count;
   }

   uint32_t alignment = kk_min_cbuf_alignment();
   set->size = align64(set->size, alignment);

   if (set->size > 0) {
      result = kk_descriptor_pool_alloc(pool, set->size, alignment, &set->addr,
                                        &set->mapped_ptr);
      if (result != VK_SUCCESS) {
         vk_object_free(&dev->vk, NULL, set);
         return result;
      }
   }

   vk_descriptor_set_layout_ref(&layout->vk);
   set->layout = layout;

   for (uint32_t b = 0; b < layout->binding_count; b++) {
      if (layout->binding[b].type != VK_DESCRIPTOR_TYPE_SAMPLER &&
          layout->binding[b].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         continue;

      if (layout->binding[b].immutable_samplers == NULL)
         continue;

      uint32_t array_size = layout->binding[b].array_size;
      if (layout->binding[b].flags &
          VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT)
         array_size = variable_count;

      const VkDescriptorImageInfo empty = {};
      for (uint32_t j = 0; j < array_size; j++) {
         write_sampled_image_view_desc(set, &empty, b, j,
                                       layout->binding[b].type);
      }
   }

   list_addtail(&set->link, &pool->sets);
   *out_set = set;

   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_AllocateDescriptorSets(VkDevice device,
                          const VkDescriptorSetAllocateInfo *pAllocateInfo,
                          VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, pAllocateInfo->descriptorPool);

   VkResult result = VK_SUCCESS;
   uint32_t i;

   struct kk_descriptor_set *set = NULL;

   const VkDescriptorSetVariableDescriptorCountAllocateInfo *var_desc_count =
      vk_find_struct_const(
         pAllocateInfo->pNext,
         DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

   /* allocate a set of buffers for each shader to contain descriptors */
   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(kk_descriptor_set_layout, layout,
                     pAllocateInfo->pSetLayouts[i]);
      /* If descriptorSetCount is zero or this structure is not included in
       * the pNext chain, then the variable lengths are considered to be zero.
       */
      const struct kk_descriptor_set_binding_layout *last =
         layout->binding_count > 0 ? &layout->binding[layout->binding_count - 1]
                                   : NULL;
      uint32_t variable_count = 0;
      if (last &&
          (last->flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) &&
          var_desc_count && var_desc_count->descriptorSetCount > 0) {
         /* A count past the binding's descriptorCount, or one missing from a
          * short array, would size the set beyond what the layout bounds. */
         if (var_desc_count->descriptorSetCount !=
                pAllocateInfo->descriptorSetCount ||
             var_desc_count->pDescriptorCounts[i] > last->array_size) {
            mesa_loge("kk: refusing a variable descriptor count past its "
                      "binding's %u, or missing",
                      last->array_size);
            result = vk_error(dev, VK_ERROR_OUT_OF_POOL_MEMORY);
            break;
         }
         variable_count = var_desc_count->pDescriptorCounts[i];
      }

      result =
         kk_descriptor_set_create(dev, pool, layout, variable_count, &set);
      if (result != VK_SUCCESS)
         break;

      pDescriptorSets[i] = kk_descriptor_set_to_handle(set);
   }

   if (result != VK_SUCCESS) {
      kk_FreeDescriptorSets(device, pAllocateInfo->descriptorPool, i,
                            pDescriptorSets);
      for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
         pDescriptorSets[i] = VK_NULL_HANDLE;
      }
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_FreeDescriptorSets(VkDevice device, VkDescriptorPool descriptorPool,
                      uint32_t descriptorSetCount,
                      const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, descriptorPool);

   for (uint32_t i = 0; i < descriptorSetCount; i++) {
      VK_FROM_HANDLE(kk_descriptor_set, set, pDescriptorSets[i]);

      if (set)
         kk_descriptor_set_destroy(dev, pool, set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
kk_DestroyDescriptorPool(VkDevice device, VkDescriptorPool _pool,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, _pool);

   if (!_pool)
      return;

   kk_destroy_descriptor_pool(dev, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_ResetDescriptorPool(VkDevice device, VkDescriptorPool descriptorPool,
                       VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(kk_device, dev, device);
   VK_FROM_HANDLE(kk_descriptor_pool, pool, descriptorPool);

   list_for_each_entry_safe(struct kk_descriptor_set, set, &pool->sets, link)
      kk_descriptor_set_destroy(dev, pool, set);

   return VK_SUCCESS;
}

/* A template entry's slots, bounded. array_count is the descriptor count, or
 * the inline block's byte size (the units kk_descriptor_update_in_bounds
 * expects for each type). */
static bool
kk_template_entry_in_bounds(const struct kk_descriptor_set *set,
                            uint32_t dyn_capacity,
                            const struct vk_descriptor_template_entry *entry)
{
   if (!kk_descriptor_update_in_bounds(set, dyn_capacity, entry->type,
                                       entry->binding, entry->array_element,
                                       entry->array_count)) {
      mesa_loge("kk: refusing a template update of %u to binding %u element %u",
                entry->array_count, entry->binding, entry->array_element);
      return false;
   }
   return true;
}

static void
kk_descriptor_set_write_template(
   struct kk_descriptor_set *set, uint32_t dyn_capacity,
   const struct vk_descriptor_update_template *template, const void *data)
{
   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct vk_descriptor_template_entry *entry = &template->entries[i];

      switch (entry->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         if (!kk_template_entry_in_bounds(set, dyn_capacity, entry))
            break;
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_sampled_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j,
                                          entry->type);
         }
         break;

      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         if (!kk_template_entry_in_bounds(set, dyn_capacity, entry))
            break;
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorImageInfo *info =
               data + entry->offset + j * entry->stride;

            write_storage_image_view_desc(set, info, entry->binding,
                                          entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         if (!kk_template_entry_in_bounds(set, dyn_capacity, entry))
            break;
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkBufferView *bview =
               data + entry->offset + j * entry->stride;

            write_buffer_view_desc(set, *bview, entry->binding,
                                   entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         if (!kk_template_entry_in_bounds(set, dyn_capacity, entry))
            break;
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_buffer_desc(set, info, entry->binding,
                              entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         if (!kk_template_entry_in_bounds(set, dyn_capacity, entry))
            break;
         for (uint32_t j = 0; j < entry->array_count; j++) {
            const VkDescriptorBufferInfo *info =
               data + entry->offset + j * entry->stride;

            write_dynamic_buffer_desc(set, info, entry->binding,
                                      entry->array_element + j);
         }
         break;

      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
         if (!kk_template_entry_in_bounds(set, dyn_capacity, entry))
            break;
         write_desc(set, entry->binding, entry->array_element,
                    data + entry->offset, entry->array_count);
         break;

      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
kk_UpdateDescriptorSetWithTemplate(
   VkDevice device, VkDescriptorSet descriptorSet,
   VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void *pData)
{
   VK_FROM_HANDLE(kk_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  descriptorUpdateTemplate);
   if (set == NULL)
      return;

   kk_descriptor_set_write_template(
      set, set->layout->vk.dynamic_descriptor_count, template, pData);
}

void
kk_push_descriptor_set_update_template(
   struct kk_push_descriptor_set *push_set,
   struct kk_descriptor_set_layout *layout,
   const struct vk_descriptor_update_template *template, const void *data)
{
   struct kk_descriptor_set tmp_set = {
      .layout = layout,
      .size = sizeof(push_set->data),
      .mapped_ptr = push_set->data,
   };
   /* dyn_capacity 0: the on-stack set's dynamic_buffers[] is empty. */
   kk_descriptor_set_write_template(&tmp_set, 0, template, data);
}
