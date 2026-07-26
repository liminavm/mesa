/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_QUERY_POOL_H
#define KK_QUERY_POOL_H 1

#include "kk_private.h"

#include "vulkan/runtime/vk_query_pool.h"

struct kk_query_pool {
   struct vk_query_pool vk;

   struct kk_bo *bo;

   uint32_t query_start;
   uint32_t query_stride;

   unsigned oq_queries;

   /* Timestamp pools only: the highest kk_timestamp_sync sequence number whose
    * report write targets this pool. Anything that observes a report has to be
    * ordered against it, because that write comes from a command-buffer
    * completion handler rather than from GPU command order. 0 = nothing
    * pending. See kk_encoder_write_timestamp. */
   uint64_t ts_pending_seq;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(kk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

uint16_t *kk_pool_oq_index_ptr(const struct kk_query_pool *pool);

#endif /* KK_QUERY_POOL_H */
