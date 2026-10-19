/*
 * Copyright 2026 limina
 * SPDX-License-Identifier: MIT
 */

#include "kk_limina_work.h"

#include "util/simple_mtx.h"

#include <stdarg.h>
#include <stdlib.h>

/* limina: see kk_limina_work.h. Guarded by its own lock: encoders close on whatever thread the
 * guest's rings run on, and the dump happens on Metal's feedback thread. */
#define KK_LIMINA_WORK_RING 128u

static simple_mtx_t kk_limina_work_lock = SIMPLE_MTX_INITIALIZER;
static struct {
   uint64_t seq;
   char what[120];
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

uint64_t
kk_limina_work_seq(void)
{
   simple_mtx_lock(&kk_limina_work_lock);
   uint64_t seq = kk_limina_work_next;
   simple_mtx_unlock(&kk_limina_work_lock);
   return seq;
}

bool
kk_limina_addr_log_enabled(void)
{
   static int on = -1;
   if (on < 0) {
      const char *e = getenv("LIMINA_KK_ADDR_LOG");
      on = e && e[0] && e[0] != '0';
      if (on)
         fprintf(stderr, "[LIMINA] KK GPU-address log ON (LIMINA_KK_ADDR_LOG)\n");
   }
   return on != 0;
}

void
kk_limina_addr_log(const char *fmt, ...)
{
   if (!kk_limina_addr_log_enabled())
      return;

   char line[192];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(line, sizeof(line), fmt, ap);
   va_end(ap);

   /* The work sequence ties an allocation event to the encoder timeline the device-loss report
    * prints, so "freed while seq 912..915 was in flight" is readable without correlating
    * timestamps. */
   fprintf(stderr, "[LIMINA-ADDR] seq=%llu %s\n",
           (unsigned long long)kk_limina_work_seq(), line);
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
