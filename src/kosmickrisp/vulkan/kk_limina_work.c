/*
 * Copyright 2026 limina
 * SPDX-License-Identifier: MIT
 */

#include "kk_limina_work.h"

#include "util/simple_mtx.h"

#include <stdarg.h>

/* limina: see kk_limina_work.h. Guarded by its own lock: encoders close on whatever thread the
 * guest's rings run on, and the dump happens on Metal's feedback thread. */
#define KK_LIMINA_WORK_RING 64u

static simple_mtx_t kk_limina_work_lock = SIMPLE_MTX_INITIALIZER;
static struct {
   uint64_t seq;
   char what[72];
} kk_limina_work_ring[KK_LIMINA_WORK_RING];
static uint64_t kk_limina_work_next = 1u;

uint64_t
kk_limina_work_record(const char *fmt, ...)
{
   va_list ap;
   simple_mtx_lock(&kk_limina_work_lock);
   uint64_t seq = kk_limina_work_next++;
   unsigned slot = (unsigned)(seq % KK_LIMINA_WORK_RING);
   kk_limina_work_ring[slot].seq = seq;
   va_start(ap, fmt);
   vsnprintf(kk_limina_work_ring[slot].what, sizeof(kk_limina_work_ring[slot].what), fmt, ap);
   va_end(ap);
   simple_mtx_unlock(&kk_limina_work_lock);
   return seq;
}

void
kk_limina_work_dump(FILE *f, unsigned max, uint64_t lo, uint64_t hi)
{
   if (max > KK_LIMINA_WORK_RING)
      max = KK_LIMINA_WORK_RING;

   simple_mtx_lock(&kk_limina_work_lock);
   uint64_t newest = kk_limina_work_next;
   uint64_t oldest = newest > max ? newest - max : 1u;
   fprintf(f, "  last %u encoders (* = in the failing commit, seq %llu..%llu):\n",
           (unsigned)(newest - oldest), (unsigned long long)lo, (unsigned long long)hi);
   for (uint64_t seq = oldest; seq < newest; ++seq) {
      unsigned slot = (unsigned)(seq % KK_LIMINA_WORK_RING);
      /* The ring may have wrapped past this sequence while we were called. */
      if (kk_limina_work_ring[slot].seq != seq)
         continue;
      fprintf(f, "   %c %6llu  %s\n", (seq >= lo && seq < hi) ? '*' : ' ',
              (unsigned long long)seq, kk_limina_work_ring[slot].what);
   }
   simple_mtx_unlock(&kk_limina_work_lock);
   fflush(f);
}
