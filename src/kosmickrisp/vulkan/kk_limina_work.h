/*
 * Copyright 2026 limina
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_LIMINA_WORK_H
#define KK_LIMINA_WORK_H 1

#include <stdint.h>
#include <stdio.h>

#include "util/macros.h"

/* limina: the last encoders handed to the GPU, so a device loss can name the work it
 * killed rather than only its Metal error code.
 *
 * Metal's MTL4 error says "Caused GPU Address Fault Error" and nothing else -- no encoder, no
 * label, no resource -- so on its own it cannot distinguish a fault in a WebGL multisample
 * resolve from one in the compositor's own blit. Each encoder records a one-line description as
 * it is closed; a failing commit knows the sequence range its command buffer recorded, so the
 * report can mark exactly which of those lines were the GPU's last work.
 *
 * The ring is small and the record is one snprintf per encoder close, not per draw. */
uint64_t kk_limina_work_record(const char *fmt, ...) PRINTFLIKE(1, 2);

/* Most recent `max` entries, oldest first, marking those in [lo, hi). */
void kk_limina_work_dump(FILE *f, unsigned max, uint64_t lo, uint64_t hi);

#endif /* KK_LIMINA_WORK_H */
