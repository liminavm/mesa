/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_encoder.h"

/* limina: per-class allocation census (limina_mtl_note_new). */
#include "mtl_bridge.h"

#include <Metal/MTL4CommandBuffer.h>
#include <Metal/MTL4ComputeCommandEncoder.h>
#include <Metal/MTL4Counters.h>
#include <Metal/MTL4RenderCommandEncoder.h>
/* limina: the full MTL4RenderPassDescriptor interface (MTL4CommandBuffer.h only forward-declares
 * it), needed by the LIMINA_KK_RPLOG dump below. */
#include <Metal/MTL4RenderPass.h>
#include <Metal/MTLRenderPass.h>

/* limina: atomics for the attachment-less clamp warning counter, execinfo for
 * its one-shot backtrace. */
#include <execinfo.h>
/* limina: dladdr, to name the frames the encoder guard records. */
#include <dlfcn.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

/* limina: LIMINA_KK_STATS=1 — once-per-second aggregate counters to stderr.
 * Measures render-pass split rate (renc + Load-action reloads) vs draw rate. */
#include <stdatomic.h>
#include <time.h>
/* limina: pthread_threadid_np for the LIMINA_KK_RPLOG thread tag. */
#include <pthread.h>
/* limina: the encoder's ivars and its driver image's UUID, for the AGX context check. */
#include <mach-o/loader.h>
#include <objc/runtime.h>

/* limina: RTLOG knob, cached — a getenv here sat on the per-draw path (round 24:
 * ~7% of the hot ring core in __findenv_locked). */
static inline bool
limina_kk_rtlog_cached(void)
{
   static int v = -1;
   if (v < 0)
      v = getenv("LIMINA_KK_RTLOG") != NULL;
   return v;
}
static _Atomic uint64_t limina_st_renc, limina_st_benc, limina_st_cenc,
   limina_st_draw, limina_st_loadc, limina_st_loadds;
static bool
limina_stats_on(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = getenv("LIMINA_KK_STATS") != NULL;
   return enabled;
}
static void
limina_stats_bump(_Atomic uint64_t *ctr)
{
   if (!limina_stats_on())
      return;
   atomic_fetch_add_explicit(ctr, 1, memory_order_relaxed);
   /* clock_gettime per bump sat on the per-draw path (round 25: ~3% of the
    * hot ring thread); only poll the clock every 1024 bumps — at >1k events/s
    * the once-per-second print cadence is unaffected. */
   static _Atomic uint64_t bumps;
   if (atomic_fetch_add_explicit(&bumps, 1, memory_order_relaxed) & 1023)
      return;
   static _Atomic long last_sec;
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   long prev = atomic_load_explicit(&last_sec, memory_order_relaxed);
   if (ts.tv_sec != prev &&
       atomic_compare_exchange_strong(&last_sec, &prev, ts.tv_sec)) {
      fprintf(stderr,
              "[LIMINA-KK-STATS] renc=%llu loadC=%llu loadDS=%llu benc=%llu "
              "cenc=%llu draw=%llu (last interval)\n",
              (unsigned long long)atomic_exchange(&limina_st_renc, 0),
              (unsigned long long)atomic_exchange(&limina_st_loadc, 0),
              (unsigned long long)atomic_exchange(&limina_st_loadds, 0),
              (unsigned long long)atomic_exchange(&limina_st_benc, 0),
              (unsigned long long)atomic_exchange(&limina_st_cenc, 0),
              (unsigned long long)atomic_exchange(&limina_st_draw, 0));
   }
}

/*
 * limina: compute-encoder liveness, keyed by pointer.
 *
 * The dogfood SIGSEGV inside AGX `prepareForEnqueue` is a store through
 * `ComputeContext+0x918`, whose only writer in AGXMetalG16X is `beginComputePass` -- which AGX
 * runs from `-[AGXG16XFamilyComputeContext_mtlnext initWithCommandBuffer:allocator:...]`, i.e. at
 * `[cmd_buf computeCommandEncoder]`. So every encoder KK is handed has been begun, and a NULL
 * there means the object lost its pass state after we got it. The dispatch ring showed the
 * faulting pointer had already served 58 copies, which leaves exactly two readings: the same
 * object was reset under us, or the address was recycled under a pointer we kept too long.
 *
 * A generation stamped per pointer separates them, and checking at the *use* site turns a
 * segfault inside Apple's driver into a named report in code we own -- which is the whole reason
 * this is at the bridge and not at the nine `cs_get_compute` callers: a future call site cannot
 * forget to be covered.
 *
 * Cost is a lock-free probe of at most LIMINA_ENC_PROBE slots on each recorded compute op. The
 * mutex is taken only when an encoder is created or marked, which is rare (35 encoders per 4096
 * dispatches in the measured dogfood ring).
 */
#define LIMINA_ENC_SLOTS 8192u
#define LIMINA_ENC_MASK (LIMINA_ENC_SLOTS - 1u)
#define LIMINA_ENC_PROBE 16u

enum limina_enc_state {
   LIMINA_ENC_UNKNOWN = 0, /* not in the table: evicted, or never a compute encoder */
   LIMINA_ENC_LIVE = 1,
   LIMINA_ENC_ENDED = 2,   /* endEncoding called */
   LIMINA_ENC_RELEASED = 3, /* our retain dropped; the address may be recycled at any moment */
   LIMINA_ENC_BROKEN = 4,   /* live, but its AGX context is not fit to record into: ops skipped */
};

/* Enough frames to name the caller and its caller; the bridge frame itself is skipped. */
#define LIMINA_ENC_FRAMES 4u

struct limina_enc_slot {
   _Atomic uintptr_t ptr;
   _Atomic uint64_t gen;
   _Atomic uint32_t state;
   _Atomic uint32_t pad;
   _Atomic uint64_t uses;
   _Atomic uint64_t thread; /* the thread that created this incarnation */
   /* Who created this incarnation, and who ended it. Not atomic: both are written under
    * limina_enc_lock, and a report racing a rewrite gets a stale frame, not a wild pointer.
    * These are the whole point of the table now that the guard prevents the crash -- a refusal
    * happens while both parties are still alive, so it can name the code that took the address
    * as well as the code still holding it, which no post-mortem crash report ever could. */
   void *born[LIMINA_ENC_FRAMES];
   void *ended[LIMINA_ENC_FRAMES];
   int born_n;
   int ended_n;
   uint64_t ended_thread;
   /* The AGX context behind this incarnation, as the encoder's ivars read at birth (see the
    * context check below). Written under the lock before `ptr` is published, read lock-free at
    * each op. `ours` = the encoder is of the class whose ivar offsets we looked up. */
   _Atomic uintptr_t impl;
   _Atomic uintptr_t alloc;
   uintptr_t cmdbuf;
   uintptr_t canary_birth;
   bool ours;
   uint32_t trip; /* enum limina_ctx_trip; set once, together with LIMINA_ENC_BROKEN */
   /* What KK says it began the command buffer on (mtl_encoder_note_allocator). */
   void *kk_alloc;
   uint32_t kk_resets;
   uint32_t kk_ops;
   /* The previous tenant of this address, kept across a reincarnation, so a trip can say
    * whether the new encoder inherited its predecessor's context. */
   uintptr_t prev_impl;
   void *prev_ended[LIMINA_ENC_FRAMES];
   int prev_ended_n;
};

static struct limina_enc_slot limina_enc_tab[LIMINA_ENC_SLOTS];
static pthread_mutex_t limina_enc_lock = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t limina_enc_gen_next = 1;
static _Atomic uint64_t limina_enc_bad[5];
static _Atomic uint64_t limina_enc_checks;   /* every guarded compute op */
static _Atomic uint64_t limina_enc_untracked; /* ... of which the table knew nothing */

static inline uint32_t
limina_enc_hash(const void *p)
{
   /* Encoders are malloc'd objects, so the low bits are dead; take from bit 6 up. */
   return (uint32_t)(((uintptr_t)p >> 6) & LIMINA_ENC_MASK);
}

/* Capture the caller's frames, skipping this function and its caller inside the bridge. */
static int
limina_enc_capture(void **out)
{
   void *raw[LIMINA_ENC_FRAMES + 2u];
   int n = backtrace(raw, (int)(LIMINA_ENC_FRAMES + 2u));
   int keep = 0;
   for (int i = 2; i < n && keep < (int)LIMINA_ENC_FRAMES; i++)
      out[keep++] = raw[i];
   return keep;
}

static void
limina_enc_print_frames(const char *what, void *const *frames, int n)
{
   if (n <= 0)
      return;
   fprintf(stderr, "[LIMINA-ENC]   %s:", what);
   for (int i = 0; i < n; i++) {
      Dl_info info;
      if (dladdr(frames[i], &info) && info.dli_sname)
         fprintf(stderr, " %s", info.dli_sname);
      else
         fprintf(stderr, " %p", frames[i]);
   }
   fprintf(stderr, "\n");
}

static const char *
limina_enc_state_name(uint32_t st)
{
   switch (st) {
   case LIMINA_ENC_LIVE: return "live";
   case LIMINA_ENC_ENDED: return "ENDED";
   case LIMINA_ENC_RELEASED: return "RELEASED";
   case LIMINA_ENC_BROKEN: return "BROKEN";
   default: return "UNKNOWN";
   }
}

/* Find the slot currently describing `enc`, or NULL. Lock-free: the worst a concurrent insert can
 * do is make us miss, which reports UNKNOWN rather than a lie. */
static struct limina_enc_slot *
limina_enc_find(const void *enc)
{
   const uint32_t h = limina_enc_hash(enc);
   for (uint32_t i = 0; i < LIMINA_ENC_PROBE; i++) {
      struct limina_enc_slot *s = &limina_enc_tab[(h + i) & LIMINA_ENC_MASK];
      if (atomic_load_explicit(&s->ptr, memory_order_acquire) == (uintptr_t)enc)
         return s;
   }
   return NULL;
}

static inline bool
limina_enc_guard_aborts(void)
{
   static int v = -1;
   if (v < 0) {
      const char *e = getenv("LIMINA_KK_ENC_GUARD");
      v = (e && !strcmp(e, "abort")) ? 1 : 0;
   }
   return v == 1;
}

/*
 * limina: the AGX context behind the encoder.
 *
 * The guard above was live through the 2026-09-11 dogfood fault and silent: the KK encoder AGX
 * faulted on was live and its own. So the bad object is not the one KK holds but the C++
 * ComputeContext the encoder reaches through its `_impl` ivar, a separate, page-aligned
 * allocation. Its pass state (`ctx+0x918` on G16X, `+0x8f0` on G13X) is written only by
 * beginComputePass, as `[ctx+off-8] = cmd; [ctx+off] = cmd + 0xc0`, and nothing clears it, so NULL
 * there means the context was zeroed or re-initialised wholesale -- and the next
 * prepareForEnqueue loads through it.
 *
 * So each incarnation records the ivars AGX gave it, and every op re-reads them and the canary.
 * WHERE it trips is the point: already NULL at birth means AGX handed out a broken context; set
 * at birth and NULL later means something ran in between. Either way the op is skipped and named
 * instead of faulting inside the driver.
 *
 * The canary is private AGX layout, so it is armed only on driver builds whose image UUID has
 * been disassembled, and only once the +0xc0 relation has been seen on a live encoder -- which
 * is what proves `_impl` is the object AGX faults on and the offset is right. The ivar
 * comparisons need no such proof and run whenever the ivars exist.
 * LIMINA_KK_CTX_CANARY=0 turns the canary off; =0x<hex> forces an offset for an unlisted build.
 */
enum limina_ctx_trip {
   LIMINA_CTX_OK = 0,
   LIMINA_CTX_BORN_BROKEN,
   LIMINA_CTX_DIED,
   LIMINA_CTX_SWAPPED_IMPL,
   LIMINA_CTX_SWAPPED_ALLOC,
};

/* Samples of the +0xc0 relation taken before the canary is trusted for good; any one failing
 * turns it off. */
#define LIMINA_CTX_VALIDATE 16u

static struct {
   bool inited;
   Class cls;
   ptrdiff_t off_impl, off_alloc, off_cmdbuf;
   uint32_t canary_off;
   _Atomic int canary_state; /* 0 = unproven, 1 = armed, -1 = off */
   uint32_t samples;
   char mode_unproven[64], mode_armed[64], mode_off[64];
} limina_ctx = {
   .off_impl = -1, .off_alloc = -1, .off_cmdbuf = -1,
   .mode_unproven = "canary=not-yet-seen", .mode_off = "canary=off",
};

static _Atomic uint64_t limina_ctx_born_broken, limina_ctx_died, limina_ctx_swapped,
   limina_ctx_alloc_mismatch, limina_ctx_skipped, limina_ctx_trips;

/* Driver builds whose beginComputePass has been read, with the offset it stores the pass state
 * at. The relation check still has to pass before the offset is used. */
static const struct {
   uint8_t uuid[16];
   const char *chip;
   uint32_t canary_off;
} limina_ctx_known[] = {
   /* AGXMetalG16X arm64e 84D26FE7-A779-3AA5-8D39-A20CBBCA3A96 (the dogfood faults) */
   { { 0x84, 0xd2, 0x6f, 0xe7, 0xa7, 0x79, 0x3a, 0xa5,
       0x8d, 0x39, 0xa2, 0x0c, 0xbb, 0xca, 0x3a, 0x96 }, "G16X", 0x918 },
   /* AGXMetalG13X arm64e E368C402-9A93-35F5-AE47-B1E561825A06 (M1 Max dev host) */
   { { 0xe3, 0x68, 0xc4, 0x02, 0x9a, 0x93, 0x35, 0xf5,
       0xae, 0x47, 0xb1, 0xe5, 0x61, 0x82, 0x5a, 0x06 }, "G13X", 0x8f0 },
};

static bool
limina_ctx_image_uuid(const void *addr, uint8_t out[16], const char **path)
{
   Dl_info info;
   if (!dladdr(addr, &info) || !info.dli_fbase)
      return false;
   *path = info.dli_fname;
   const struct mach_header_64 *mh = info.dli_fbase;
   if (mh->magic != MH_MAGIC_64)
      return false;
   const struct load_command *lc = (const void *)(mh + 1);
   for (uint32_t i = 0; i < mh->ncmds; i++) {
      if (lc->cmd == LC_UUID) {
         memcpy(out, ((const struct uuid_command *)lc)->uuid, 16);
         return true;
      }
      lc = (const void *)((const char *)lc + lc->cmdsize);
   }
   return false;
}

static ptrdiff_t
limina_ctx_ivar_off(Class cls, const char *name)
{
   Ivar iv = class_getInstanceVariable(cls, name);
   return iv ? ivar_getOffset(iv) : -1;
}

static inline uintptr_t
limina_ctx_ivar(const void *obj, ptrdiff_t off)
{
   return off < 0 ? 0 : *(const volatile uintptr_t *)((const char *)obj + off);
}

static inline uintptr_t
limina_ctx_word(uintptr_t impl, uint32_t off)
{
   return *(const volatile uintptr_t *)(impl + off);
}

/* Called once, under limina_enc_lock, with the first compute encoder AGX hands out. */
static void
limina_ctx_init(void *enc)
{
   limina_ctx.inited = true;
   Class cls = object_getClass((id)enc);
   limina_ctx.cls = cls;
   limina_ctx.off_impl = limina_ctx_ivar_off(cls, "_impl");
   limina_ctx.off_alloc = limina_ctx_ivar_off(cls, "_allocator");
   limina_ctx.off_cmdbuf = limina_ctx_ivar_off(cls, "_command_buffer");

   uint8_t uuid[16] = { 0 };
   const char *img = NULL;
   const bool have_uuid = limina_ctx_image_uuid((const void *)cls, uuid, &img);
   const char *chip = NULL;
   uint32_t off = 0;
   for (unsigned i = 0; have_uuid && i < sizeof(limina_ctx_known) / sizeof(limina_ctx_known[0]); i++) {
      if (!memcmp(uuid, limina_ctx_known[i].uuid, 16)) {
         chip = limina_ctx_known[i].chip;
         off = limina_ctx_known[i].canary_off;
      }
   }
   const char *e = getenv("LIMINA_KK_CTX_CANARY");
   if (e && e[0]) {
      off = (uint32_t)strtoul(e, NULL, 0);
      chip = off ? "forced" : "disabled";
   }
   if (limina_ctx.off_impl < 0)
      off = 0;
   limina_ctx.canary_off = off;
   atomic_store(&limina_ctx.canary_state, off ? 0 : -1);
   snprintf(limina_ctx.mode_armed, sizeof(limina_ctx.mode_armed), "canary=%s+0x%x",
            chip ? chip : "?", off);

   char us[40];
   snprintf(us, sizeof(us),
            "%02X%02X%02X%02X-%02X%02X-%02X%02X-%02X%02X-%02X%02X%02X%02X%02X%02X", uuid[0],
            uuid[1], uuid[2], uuid[3], uuid[4], uuid[5], uuid[6], uuid[7], uuid[8], uuid[9],
            uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]);
   fprintf(stderr,
           "[LIMINA-CTX] compute encoder class %s from %s (UUID %s): _impl@%td _allocator@%td "
           "_command_buffer@%td; pass-state canary %s",
           class_getName(cls), img ? img : "?", have_uuid ? us : "?", limina_ctx.off_impl,
           limina_ctx.off_alloc, limina_ctx.off_cmdbuf, off ? "at " : "OFF");
   if (off)
      fprintf(stderr, "+0x%x (%s), armed once the +0xc0 relation is seen\n", off, chip);
   else
      fprintf(stderr, " (%s)\n", limina_ctx.off_impl < 0 ? "no _impl ivar"
                                 : chip                  ? chip
                                                         : "driver build not listed");
   fflush(stderr);
}

/* Feed one birth sample to the relation check. Under limina_enc_lock. */
static void
limina_ctx_validate(uintptr_t canary, uintptr_t cmd)
{
   if (canary == 0 || limina_ctx.samples >= LIMINA_CTX_VALIDATE ||
       atomic_load(&limina_ctx.canary_state) < 0)
      return;
   limina_ctx.samples++;
   if (canary == cmd + 0xc0) {
      if (atomic_load(&limina_ctx.canary_state) == 0) {
         atomic_store(&limina_ctx.canary_state, 1);
         fprintf(stderr, "[LIMINA-CTX] pass-state canary armed (%s): [+0x%x]=0x%llx is "
                         "[+0x%x]+0xc0 as beginComputePass writes it\n",
                 limina_ctx.mode_armed, limina_ctx.canary_off, (unsigned long long)canary,
                 limina_ctx.canary_off - 8u);
         fflush(stderr);
      }
   } else {
      atomic_store(&limina_ctx.canary_state, -1);
      fprintf(stderr, "[LIMINA-CTX] pass-state canary turned OFF: [+0x%x]=0x%llx but "
                      "[+0x%x]=0x%llx, not the +0xc0 beginComputePass writes -- wrong offset or "
                      "_impl is not the ComputeContext on this build\n",
              limina_ctx.canary_off, (unsigned long long)canary, limina_ctx.canary_off - 8u,
              (unsigned long long)cmd);
      fflush(stderr);
   }
}

/* Name a trip. Under limina_enc_lock; the table walk is paid only here. */
static void
limina_ctx_report(struct limina_enc_slot *s, const void *enc, const char *site, uint32_t why,
                  uintptr_t impl_now, uintptr_t alloc_now, uintptr_t canary_now, bool canary_read)
{
   static const char *const what[] = {
      [LIMINA_CTX_OK] = "?",
      [LIMINA_CTX_BORN_BROKEN] = "AGX handed out a compute encoder whose context has no pass state",
      [LIMINA_CTX_DIED] = "the context of a live compute encoder lost its pass state",
      [LIMINA_CTX_SWAPPED_IMPL] = "a live compute encoder's _impl (its context) changed",
      [LIMINA_CTX_SWAPPED_ALLOC] = "a live compute encoder's _allocator changed",
   };
   const uint64_t n = atomic_fetch_add(&limina_ctx_trips, 1);
   if (n >= 16 && (n & 4095u) != 0)
      return;

   const uintptr_t impl_birth = atomic_load(&s->impl);
   fprintf(stderr,
           "[LIMINA-CTX] %s -- encoder %p at %s, gen %llu, %llu uses, created on thread 0x%llx, "
           "this thread 0x%llx (%llu so far)\n",
           what[why], enc, site, (unsigned long long)atomic_load(&s->gen),
           (unsigned long long)atomic_load(&s->uses), (unsigned long long)atomic_load(&s->thread),
           (unsigned long long)(uintptr_t)pthread_self(), (unsigned long long)n + 1);
   fprintf(stderr, "[LIMINA-CTX]   _impl: birth 0x%llx now 0x%llx; pass state [+0x%x]: birth "
                   "0x%llx now ",
           (unsigned long long)impl_birth, (unsigned long long)impl_now, limina_ctx.canary_off,
           (unsigned long long)s->canary_birth);
   if (canary_read)
      fprintf(stderr, "0x%llx\n", (unsigned long long)canary_now);
   else
      fprintf(stderr, "(not read)\n");
   fprintf(stderr, "[LIMINA-CTX]   _allocator: birth 0x%llx now 0x%llx; KK began the command "
                   "buffer on %p (resets %u, ops since its last reset %u)%s\n",
           (unsigned long long)atomic_load(&s->alloc), (unsigned long long)alloc_now, s->kk_alloc,
           s->kk_resets, s->kk_ops, s->kk_alloc ? "" : " -- not told yet, see the next line");
   fprintf(stderr, "[LIMINA-CTX]   _command_buffer: birth 0x%llx now 0x%llx\n",
           (unsigned long long)s->cmdbuf,
           (unsigned long long)limina_ctx_ivar(enc, limina_ctx.off_cmdbuf));

   /* Does anyone else think they own this context? AGX hands successive encoders on one
    * allocator the SAME ComputeContext (measured on G13X: one page-aligned context across every
    * encoder of a vkcube run), so a shared context with the previous tenant is normal -- but two
    * encoders LIVE on one context at once is the re-issue story in one line. */
   uint32_t sharers = 0;
   const struct limina_enc_slot *first = NULL;
   for (uint32_t i = 0; impl_birth && i < LIMINA_ENC_SLOTS; i++) {
      const struct limina_enc_slot *t = &limina_enc_tab[i];
      if (t == s || !atomic_load(&t->ptr))
         continue;
      const uint32_t tst = atomic_load(&t->state);
      if ((tst == LIMINA_ENC_LIVE || tst == LIMINA_ENC_BROKEN) &&
          (atomic_load(&t->impl) == impl_birth || (impl_now && atomic_load(&t->impl) == impl_now))) {
         if (!first)
            first = t;
         sharers++;
      }
   }
   if (first)
      fprintf(stderr, "[LIMINA-CTX]   %u other live encoder(s) on this context, first %p gen %llu "
                      "created on thread 0x%llx\n",
              sharers, (void *)atomic_load(&first->ptr), (unsigned long long)atomic_load(&first->gen),
              (unsigned long long)atomic_load(&first->thread));
   else
      fprintf(stderr, "[LIMINA-CTX]   no other live encoder on this context\n");

   limina_enc_print_frames("created by", s->born, s->born_n);
   if (s->prev_impl || s->prev_ended_n) {
      fprintf(stderr, "[LIMINA-CTX]   this address's previous tenant had _impl 0x%llx (%s)\n",
              (unsigned long long)s->prev_impl,
              s->prev_impl == impl_birth ? "the SAME context" : "a different context");
      limina_enc_print_frames("previous tenant ended by", s->prev_ended, s->prev_ended_n);
   }
   fprintf(stderr, "[LIMINA-CTX]   dropping this and every later op on this encoder rather than "
                   "recording into it (LIMINA_KK_ENC_GUARD=abort takes a core instead)\n");
   fflush(stderr);
}

static void
limina_enc_note_new(void *enc)
{
   if (!enc)
      return;
   const uint32_t h = limina_enc_hash(enc);
   pthread_mutex_lock(&limina_enc_lock);
   struct limina_enc_slot *pick = NULL;
   struct limina_enc_slot *oldest = NULL;
   bool same = false;
   for (uint32_t i = 0; i < LIMINA_ENC_PROBE; i++) {
      struct limina_enc_slot *s = &limina_enc_tab[(h + i) & LIMINA_ENC_MASK];
      uintptr_t sp = atomic_load(&s->ptr);
      if (sp == (uintptr_t)enc) { pick = s; same = true; break; } /* this address, reincarnated */
      if (sp == 0) { pick = pick ? pick : s; continue; }   /* free */
      if (!pick && atomic_load(&s->state) == LIMINA_ENC_RELEASED) { pick = s; continue; }
      if (!oldest || atomic_load(&s->gen) < atomic_load(&oldest->gen))
         oldest = s;
   }
   if (!pick)
      pick = oldest ? oldest : &limina_enc_tab[h];

   /* The context AGX attached, read before anyone records into the encoder -- this sample is
    * what can say "born broken". */
   if (!limina_ctx.inited)
      limina_ctx_init(enc);
   const bool ours = object_getClass((id)enc) == limina_ctx.cls;
   uintptr_t impl = 0, alloc = 0, cmdbuf = 0, canary = 0;
   bool born_broken = false;
   if (ours) {
      impl = limina_ctx_ivar(enc, limina_ctx.off_impl);
      alloc = limina_ctx_ivar(enc, limina_ctx.off_alloc);
      cmdbuf = limina_ctx_ivar(enc, limina_ctx.off_cmdbuf);
      if (impl && limina_ctx.canary_off && atomic_load(&limina_ctx.canary_state) >= 0) {
         canary = limina_ctx_word(impl, limina_ctx.canary_off);
         limina_ctx_validate(canary, limina_ctx_word(impl, limina_ctx.canary_off - 8u));
      }
      born_broken = atomic_load(&limina_ctx.canary_state) == 1 && (!impl || !canary);
   }

   if (same) {
      pick->prev_impl = atomic_load(&pick->impl);
      pick->prev_ended_n = pick->ended_n;
      memcpy(pick->prev_ended, pick->ended, sizeof(pick->ended));
   } else {
      pick->prev_impl = 0;
      pick->prev_ended_n = 0;
   }
   atomic_store(&pick->gen, atomic_fetch_add(&limina_enc_gen_next, 1));
   atomic_store(&pick->uses, 0);
   atomic_store(&pick->thread, (uint64_t)(uintptr_t)pthread_self());
   pick->born_n = limina_enc_capture(pick->born);
   pick->ended_n = 0;
   pick->ended_thread = 0;
   atomic_store(&pick->impl, impl);
   atomic_store(&pick->alloc, alloc);
   pick->cmdbuf = cmdbuf;
   pick->canary_birth = canary;
   pick->ours = ours;
   pick->trip = born_broken ? LIMINA_CTX_BORN_BROKEN : LIMINA_CTX_OK;
   pick->kk_alloc = NULL;
   pick->kk_resets = 0;
   pick->kk_ops = 0;
   atomic_store(&pick->state, born_broken ? LIMINA_ENC_BROKEN : LIMINA_ENC_LIVE);
   atomic_store_explicit(&pick->ptr, (uintptr_t)enc, memory_order_release);
   if (born_broken) {
      atomic_fetch_add(&limina_ctx_born_broken, 1);
      limina_ctx_report(pick, enc, "birth", LIMINA_CTX_BORN_BROKEN, impl, alloc, canary, true);
   }
   pthread_mutex_unlock(&limina_enc_lock);
   if (born_broken && limina_enc_guard_aborts())
      abort();
}

/* A live encoder's context no longer matches what it was born with. Marks it BROKEN once and
 * names it; every later op on it is skipped. */
static uint32_t
limina_ctx_trip(struct limina_enc_slot *s, void *enc, const char *site, uint32_t why,
                uintptr_t impl, uintptr_t alloc, uintptr_t canary, bool canary_read)
{
   pthread_mutex_lock(&limina_enc_lock);
   if (atomic_load(&s->ptr) == (uintptr_t)enc && atomic_load(&s->state) == LIMINA_ENC_LIVE) {
      atomic_store(&s->state, LIMINA_ENC_BROKEN);
      s->trip = why;
      atomic_fetch_add(why == LIMINA_CTX_DIED ? &limina_ctx_died : &limina_ctx_swapped, 1);
      limina_ctx_report(s, enc, site, why, impl, alloc, canary, canary_read);
   }
   pthread_mutex_unlock(&limina_enc_lock);
   if (limina_enc_guard_aborts())
      abort();
   return LIMINA_ENC_BROKEN;
}

/* The per-op half: two ivar loads and, when armed, one canary load. */
static inline uint32_t
limina_ctx_check(struct limina_enc_slot *s, void *enc, const char *site)
{
   if (!s->ours)
      return LIMINA_ENC_LIVE;
   const uintptr_t impl = limina_ctx_ivar(enc, limina_ctx.off_impl);
   const uintptr_t alloc = limina_ctx_ivar(enc, limina_ctx.off_alloc);
   if (__builtin_expect(impl != atomic_load_explicit(&s->impl, memory_order_relaxed), 0))
      return limina_ctx_trip(s, enc, site, LIMINA_CTX_SWAPPED_IMPL, impl, alloc, 0, false);
   if (__builtin_expect(alloc != atomic_load_explicit(&s->alloc, memory_order_relaxed), 0))
      return limina_ctx_trip(s, enc, site, LIMINA_CTX_SWAPPED_ALLOC, impl, alloc, 0, false);
   if (impl && atomic_load_explicit(&limina_ctx.canary_state, memory_order_relaxed) == 1) {
      const uintptr_t canary = limina_ctx_word(impl, limina_ctx.canary_off);
      if (__builtin_expect(canary == 0, 0))
         return limina_ctx_trip(s, enc, site, LIMINA_CTX_DIED, impl, alloc, canary, true);   }
   return LIMINA_ENC_LIVE;
}

void
mtl_encoder_note_allocator(void *encoder, void *allocator, uint32_t resets,
                           uint32_t ops_since_reset)
{
   if (!encoder)
      return;
   pthread_mutex_lock(&limina_enc_lock);
   struct limina_enc_slot *s = limina_enc_find(encoder);
   if (s) {
      s->kk_alloc = allocator;
      s->kk_resets = resets;
      s->kk_ops = ops_since_reset;
      const uintptr_t mine = atomic_load(&s->alloc);
      if (s->ours && limina_ctx.off_alloc >= 0 && mine != (uintptr_t)allocator) {
         const uint64_t n = atomic_fetch_add(&limina_ctx_alloc_mismatch, 1);
         if (n < 8 || (n & 4095u) == 0) {
            fprintf(stderr, "[LIMINA-CTX] compute encoder %p (gen %llu) carries _allocator 0x%llx "
                            "but KK began its command buffer on %p (%llu so far)\n",
                    encoder, (unsigned long long)atomic_load(&s->gen), (unsigned long long)mine,
                    allocator, (unsigned long long)n + 1);
            fflush(stderr);
         }
      }
      if (s->trip == LIMINA_CTX_BORN_BROKEN && atomic_load(&limina_ctx_trips) <= 16) {
         fprintf(stderr, "[LIMINA-CTX]   born-broken encoder %p: KK began its command buffer on "
                         "allocator %p, resets %u, ops since its last reset %u\n",
                 encoder, allocator, resets, ops_since_reset);
         fflush(stderr);
      }
   }
   pthread_mutex_unlock(&limina_enc_lock);
}

static void
limina_enc_mark(void *enc, uint32_t state)
{
   if (!enc)
      return;
   pthread_mutex_lock(&limina_enc_lock);
   struct limina_enc_slot *s = limina_enc_find(enc);
   /* Never walk a state backwards: end-then-release is the order, and a stray endEncoding on an
    * already-released pointer must not resurrect it to ENDED. */
   const uint32_t cur = s ? atomic_load(&s->state) : LIMINA_ENC_UNKNOWN;
   /* BROKEN is still an open encoder: KK ends and releases it like any other. */
   if (s && (cur < state || cur == LIMINA_ENC_BROKEN)) {
      /* Only the first transition out of LIVE is recorded: that is the call that took the
       * encoder away, and the release that follows it is bookkeeping. */
      if (cur == LIMINA_ENC_LIVE || cur == LIMINA_ENC_BROKEN) {
         s->ended_n = limina_enc_capture(s->ended);
         s->ended_thread = (uint64_t)(uintptr_t)pthread_self();
      }
      atomic_store(&s->state, state);
   }
   pthread_mutex_unlock(&limina_enc_lock);
}

void
mtl_encoder_note_released(void *encoder)
{
   limina_enc_mark(encoder, LIMINA_ENC_RELEASED);
}

uint64_t
mtl_encoder_generation(void *encoder)
{
   struct limina_enc_slot *s = encoder ? limina_enc_find(encoder) : NULL;
   return s ? atomic_load(&s->gen) : 0;
}

void
mtl_encoder_report_incarnation(void *encoder, const char *why)
{
   struct limina_enc_slot *s = encoder ? limina_enc_find(encoder) : NULL;
   if (!s) {
      fprintf(stderr, "[LIMINA-ENC] %s: encoder %p is not in the table (evicted, or never a "
                      "compute encoder) — no history to give\n", why, encoder);
      fflush(stderr);
      return;
   }
   fprintf(stderr,
           "[LIMINA-ENC] %s: encoder %p is now generation %llu, state %s, created on thread "
           "0x%llx, %llu uses\n",
           why, encoder, (unsigned long long)atomic_load(&s->gen),
           limina_enc_state_name(atomic_load(&s->state)),
           (unsigned long long)atomic_load(&s->thread),
           (unsigned long long)atomic_load(&s->uses));
   limina_enc_print_frames("created by", s->born, s->born_n);
   if (s->ended_n > 0) {
      fprintf(stderr, "[LIMINA-ENC]   ended on thread 0x%llx\n",
              (unsigned long long)s->ended_thread);
      limina_enc_print_frames("ended by", s->ended, s->ended_n);
   }
   fflush(stderr);
}

void
mtl_encoder_guard_stats(struct mtl_encoder_guard_stats *out)
{
   out->checks = atomic_load(&limina_enc_checks);
   out->untracked = atomic_load(&limina_enc_untracked);
   out->bad_null = atomic_load(&limina_enc_bad[LIMINA_ENC_UNKNOWN]);
   out->bad_ended = atomic_load(&limina_enc_bad[LIMINA_ENC_ENDED]);
   out->bad_released = atomic_load(&limina_enc_bad[LIMINA_ENC_RELEASED]);
   out->encoders_seen = atomic_load(&limina_enc_gen_next) - 1u;
   out->slots_total = LIMINA_ENC_SLOTS;

   /* Walked rather than counted incrementally: the reporter runs once every 2000 ticks, and a
    * counter maintained on the insert path would have to track evictions to stay honest. */
   uint32_t used = 0;
   for (uint32_t i = 0; i < LIMINA_ENC_SLOTS; i++) {
      if (atomic_load_explicit(&limina_enc_tab[i].ptr, memory_order_relaxed) != 0)
         used++;
   }
   out->slots_used = used;

   const int cs = atomic_load(&limina_ctx.canary_state);
   out->ctx_mode = cs > 0    ? limina_ctx.mode_armed
                   : cs == 0 ? limina_ctx.mode_unproven
                             : limina_ctx.mode_off;
   out->ctx_born_broken = atomic_load(&limina_ctx_born_broken);
   out->ctx_died = atomic_load(&limina_ctx_died);
   out->ctx_swapped = atomic_load(&limina_ctx_swapped);
   out->ctx_alloc_mismatch = atomic_load(&limina_ctx_alloc_mismatch);
   out->ctx_skipped = atomic_load(&limina_ctx_skipped);
}

/* Report a use of an encoder that is not live, and hand back what we knew about it. `gen_out` may
 * be NULL. Returns the state, so the dispatch ring can record it beside the copy. */
static uint32_t
limina_enc_check(void *enc, const char *site, uint64_t *gen_out)
{
   if (gen_out)
      *gen_out = 0;

   if (enc == NULL) {
      /* cs_get_compute hands NULL back when the allocator pool is empty; every downstream call
       * site uses it unchecked, and a nil ObjC receiver is a silent no-op -- so the copy simply
       * never happens and the corruption surfaces somewhere else entirely. Name it here. */
      uint64_t n = atomic_fetch_add(&limina_enc_bad[0], 1);
      if (n < 8 || (n & 4095u) == 0)
         fprintf(stderr, "[LIMINA-ENC] NULL compute encoder at %s (%llu so far)\n", site,
                 (unsigned long long)n + 1);
      if (limina_enc_guard_aborts())
         abort();
      return LIMINA_ENC_UNKNOWN;
   }

   atomic_fetch_add(&limina_enc_checks, 1);

   struct limina_enc_slot *s = limina_enc_find(enc);
   if (!s) {
      /* Evicted, or a render encoder. Not evidence of anything by itself -- but it is the one
       * shape in which a genuinely stale pointer sails through every check, so it is counted
       * rather than merely tolerated. */
      atomic_fetch_add(&limina_enc_untracked, 1);
      return LIMINA_ENC_UNKNOWN;
   }

   const uint32_t st = atomic_load(&s->state);
   const uint64_t gen = atomic_load(&s->gen);
   if (gen_out)
      *gen_out = gen;
   atomic_fetch_add(&s->uses, 1);
   if (st == LIMINA_ENC_LIVE)
      return limina_ctx_check(s, enc, site);
   if (st == LIMINA_ENC_BROKEN) {
      /* Already named once, at the trip. */
      atomic_fetch_add(&limina_ctx_skipped, 1);
      return st;
   }

   uint64_t n = atomic_fetch_add(&limina_enc_bad[st & 3u], 1);
   if (n < 8 || (n & 4095u) == 0) {
      fprintf(stderr,
              "[LIMINA-ENC] %s encoder %p used at %s — gen %llu, created on thread 0x%llx, "
              "%llu uses, this thread 0x%llx (%llu so far)\n",
              limina_enc_state_name(st), enc, site, (unsigned long long)gen,
              (unsigned long long)atomic_load(&s->thread),
              (unsigned long long)atomic_load(&s->uses),
              (unsigned long long)(uintptr_t)pthread_self(), (unsigned long long)n + 1);
      limina_enc_print_frames("created by", s->born, s->born_n);
      fprintf(stderr, "[LIMINA-ENC]   ended on thread 0x%llx\n",
              (unsigned long long)s->ended_thread);
      limina_enc_print_frames("ended by", s->ended, s->ended_n);
      fflush(stderr);
   }
   if (limina_enc_guard_aborts())
      abort();
   return st;
}

/* Common encoder utils */
void
mtl_end_encoding(void *encoder)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc endEncoding];
   }
   /* After this the encoder records nothing; a later op on it is the bug we are hunting. Render
    * encoders are not in the table, so marking them is a no-op. */
   limina_enc_mark(encoder, LIMINA_ENC_ENDED);
}

void
mtl_barrier_after_stages(void *encoder, enum mtl_stages after_stages,
                         enum mtl_stages before_queue_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      /* TODO_KOSMICKRISP Expose visibility option since resource alias should only be required
       * if occlusion queries are used */
      [enc barrierAfterStages:(MTLStages)after_stages beforeQueueStages:(MTLStages)before_queue_stages
           visibilityOptions:MTL4VisibilityOptionResourceAlias];
   }
}

void
mtl_barrier_after_encoder_stages(void *encoder, enum mtl_stages after_stages,
                                 enum mtl_stages before_queue_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc barrierAfterEncoderStages:(MTLStages)after_stages beforeEncoderStages:(MTLStages)before_queue_stages
           visibilityOptions:MTL4VisibilityOptionResourceAlias];
   }
}

void
mtl_barrier_after_queue_stages(void *encoder,
                               enum mtl_stages after_queue_stages,
                               enum mtl_stages before_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      [enc barrierAfterQueueStages:(MTLStages)after_queue_stages
                      beforeStages:(MTLStages)before_stages
                 visibilityOptions:MTL4VisibilityOptionResourceAlias];
   }
}

void
mtl_update_fence(void *encoder, mtl_fence *fence, enum mtl_stages after_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc updateFence:f afterEncoderStages:(MTLStages)after_stages];
   }
}

void
mtl_wait_for_fence(void *encoder, mtl_fence *fence,
                   enum mtl_stages before_stages)
{
   @autoreleasepool {
      id<MTL4CommandEncoder> enc = (id<MTL4CommandEncoder>)encoder;
      id<MTLFence> f = (id<MTLFence>)fence;
      [enc waitForFence:f beforeEncoderStages:(MTLStages)before_stages];
   }
}

/* limina: crash-survivable dispatch breadcrumbs.
 *
 * The AGX fault we are chasing kills the process INSIDE copyFromBuffer:toTexture: — an
 * unchecked next-segment pointer when a data-buffer request straddles the end of AGX's 1 MiB
 * segment. Anything buffered in stdio dies with the process, and the fault leaves no trace of
 * WHICH dispatch asked for the allocation that did not fit.
 *
 * So the record has to be on disk before the call is made. A MAP_SHARED file written with plain
 * stores does that: the kernel writes back dirty pages of a shared mapping even when the process
 * dies on SIGSEGV, and there is no syscall per entry, which is what lets this stay on by default
 * rather than being an option nobody has armed when the rare thing finally happens.
 *
 * `done` is the whole trick. It is stored 0 before the call and 1 after, so after a crash the
 * entry still holding 0 is the dispatch that was in flight — the culprit names itself, instead of
 * being inferred from whatever happened to be logged nearby.
 */
#define LIMINA_DT_ENTRIES 4096u
#define LIMINA_DT_MAGIC 0x4c444d32u /* 'LDM2' — v2 entries carry the encoder generation */

enum limina_dt_kind {
   LIMINA_DT_BUF_TO_IMG = 1,
   LIMINA_DT_IMG_TO_BUF = 2,
   LIMINA_DT_BUF_TO_BUF = 3,
};

struct limina_dt_entry {
   uint64_t seq; /* 0 = never written */
   uint64_t done; /* 0 while IN FLIGHT — a crash lands on one of these */
   uint64_t thread;
   uint64_t encoder;
   uint64_t buffer;
   uint64_t texture;
   uint64_t offset_B;
   uint64_t stride_B;
   uint64_t image_2d_B;
   uint32_t w, h, d;
   uint32_t x, y, z;
   uint32_t slice, level, options, kind;
   /* limina: which *incarnation* of `encoder` this was. The 2026-09-07 fault landed on a pointer
    * that had already served 58 copies, so the raw pointer cannot say whether the object was
    * reset under us or the address was recycled -- the generation can. */
   uint64_t gen;
   uint32_t enc_state; /* enum limina_enc_state as seen at record time */
   uint32_t pad2;
};

struct limina_dt_hdr {
   uint32_t magic;
   uint32_t entry_size;
   uint32_t entries;
   uint32_t pad;
   _Atomic uint64_t next;
   struct limina_dt_entry e[];
};

static struct limina_dt_hdr *limina_dt_map;

static void
limina_dt_init(void)
{
   static bool tried;
   if (tried)
      return;
   tried = true;

   /* Explicit path wins; otherwise ride the pool snapshot's directory, which the supervisor
    * already points at the VM bundle's logs/ — so a managed VM gets this with no extra wiring. */
   const char *path = getenv("LIMINA_KK_DISPATCH_TRACE");
   char derived[1024];
   if (!path || !path[0]) {
      const char *base = getenv("LIMINA_KK_POOL_SNAPSHOT");
      if (!base || !base[0])
         return;
      snprintf(derived, sizeof(derived), "%s.dispatch.%d", base, (int)getpid());
      path = derived;
   }

   const size_t size =
      sizeof(struct limina_dt_hdr) + (size_t)LIMINA_DT_ENTRIES * sizeof(struct limina_dt_entry);
   int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
   if (fd < 0)
      return;
   if (ftruncate(fd, (off_t)size) != 0) {
      close(fd);
      return;
   }
   void *m = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
   close(fd);
   if (m == MAP_FAILED)
      return;

   struct limina_dt_hdr *h = m;
   h->magic = LIMINA_DT_MAGIC;
   h->entry_size = (uint32_t)sizeof(struct limina_dt_entry);
   h->entries = LIMINA_DT_ENTRIES;
   atomic_store(&h->next, 0);
   limina_dt_map = h;
}

/* Claim a slot and fill it. Returns the entry so the caller can mark it done, or NULL when the
 * trace is off — in which case every call here is one predictable branch. */
static struct limina_dt_entry *
limina_dt_begin(enum limina_dt_kind kind, void *encoder, void *buffer, void *texture,
                const struct mtl_buffer_image_copy *d, uint64_t gen, uint32_t enc_state)
{
   limina_dt_init();
   struct limina_dt_hdr *h = limina_dt_map;
   if (!h)
      return NULL;

   const uint64_t seq = atomic_fetch_add(&h->next, 1) + 1;
   struct limina_dt_entry *e = &h->e[seq % h->entries];
   e->done = 0;
   e->seq = seq;
   e->thread = (uint64_t)(uintptr_t)pthread_self();
   e->encoder = (uint64_t)(uintptr_t)encoder;
   e->buffer = (uint64_t)(uintptr_t)buffer;
   e->texture = (uint64_t)(uintptr_t)texture;
   e->kind = (uint32_t)kind;
   e->gen = gen;
   e->enc_state = enc_state;
   if (d) {
      e->offset_B = d->buffer_offset_B;
      e->stride_B = d->buffer_stride_B;
      e->image_2d_B = d->buffer_2d_image_size_B;
      e->w = (uint32_t)d->image_size.x;
      e->h = (uint32_t)d->image_size.y;
      e->d = (uint32_t)d->image_size.z;
      e->x = (uint32_t)d->image_origin.x;
      e->y = (uint32_t)d->image_origin.y;
      e->z = (uint32_t)d->image_origin.z;
      e->slice = (uint32_t)d->image_slice;
      e->level = (uint32_t)d->image_level;
      e->options = (uint32_t)d->options;
   }
   return e;
}

/* MTLComputeEncoder */
mtl_compute_encoder *
mtl_new_compute_command_encoder(mtl_command_buffer *cmd_buffer)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd_buf = (id<MTL4CommandBuffer>)cmd_buffer;
      void *enc = limina_mtl_note_new([[cmd_buf computeCommandEncoder] retain]);
      limina_enc_note_new(enc);
      return (mtl_compute_encoder *)enc;
   }
}

void
mtl_copy_from_buffer_to_buffer(mtl_compute_encoder *encoder,
                               mtl_buffer *src_buf, size_t src_offset,
                               mtl_buffer *dst_buf, size_t dst_offset,
                               size_t size)
{
   if (limina_enc_check(encoder, "mtl_copy_from_buffer_to_buffer", NULL) == LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLBuffer> mtl_src_buffer = (id<MTLBuffer>)src_buf;
      id<MTLBuffer> mtl_dst_buffer = (id<MTLBuffer>)dst_buf;
      [enc copyFromBuffer:mtl_src_buffer sourceOffset:src_offset toBuffer:mtl_dst_buffer destinationOffset:dst_offset size:size];
   }
}

void
mtl_copy_from_buffer_to_texture(mtl_compute_encoder *encoder,
                                struct mtl_buffer_image_copy *data)
{
   uint64_t limina_gen = 0;
   const uint32_t limina_enc_st = limina_enc_check(encoder, "mtl_copy_from_buffer_to_texture", &limina_gen);
   if (limina_enc_st == LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      const MTLSize size = MTLSizeMake(data->image_size.x, data->image_size.y, data->image_size.z);
      const MTLOrigin origin = MTLOriginMake(data->image_origin.x, data->image_origin.y, data->image_origin.z);
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLBuffer> buffer = (id<MTLBuffer>)data->buffer;
      id<MTLTexture> image = (id<MTLTexture>)data->image;
      struct limina_dt_entry *bc =
         limina_dt_begin(LIMINA_DT_BUF_TO_IMG, encoder, data->buffer, data->image, data,
                         limina_gen, limina_enc_st);
      [enc copyFromBuffer:buffer
             sourceOffset:data->buffer_offset_B
        sourceBytesPerRow:data->buffer_stride_B
      sourceBytesPerImage:data->buffer_2d_image_size_B
               sourceSize:size
                toTexture:image
         destinationSlice:data->image_slice
         destinationLevel:data->image_level
        destinationOrigin:origin
                  options:(MTLBlitOption)data->options];
      if (bc)
         bc->done = 1;
   }
}

void
mtl_copy_from_texture_to_buffer(mtl_compute_encoder *encoder,
                                struct mtl_buffer_image_copy *data)
{
   if (limina_enc_check(encoder, "mtl_copy_from_texture_to_buffer", NULL) == LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      const MTLSize size = MTLSizeMake(data->image_size.x, data->image_size.y, data->image_size.z);
      const MTLOrigin origin = MTLOriginMake(data->image_origin.x, data->image_origin.y, data->image_origin.z);
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLBuffer> buffer = (id<MTLBuffer>)data->buffer;
      id<MTLTexture> image = (id<MTLTexture>)data->image;
      [enc copyFromTexture:image
               sourceSlice:data->image_slice
               sourceLevel:data->image_level
              sourceOrigin:origin
                sourceSize:size
                  toBuffer:buffer
         destinationOffset:data->buffer_offset_B
    destinationBytesPerRow:data->buffer_stride_B
  destinationBytesPerImage:data->buffer_2d_image_size_B
                   options:(MTLBlitOption)data->options];
   }
}

void
mtl_copy_from_texture_to_texture(mtl_compute_encoder *encoder,
                                 mtl_texture *src_tex_handle, size_t src_slice,
                                 size_t src_level, struct mtl_origin src_origin,
                                 struct mtl_size src_size,
                                 mtl_texture *dst_tex_handle, size_t dst_slice,
                                 size_t dst_level, struct mtl_origin dst_origin)
{
   if (limina_enc_check(encoder, "mtl_copy_from_texture_to_texture", NULL) == LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      MTLOrigin mtl_src_origin = MTLOriginMake(src_origin.x, src_origin.y, src_origin.z);
      MTLSize mtl_src_size = MTLSizeMake(src_size.x, src_size.y, src_size.z);
      MTLOrigin mtl_dst_origin = MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z);
      id<MTLTexture> mtl_dst_tex = (id<MTLTexture>)dst_tex_handle;
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLTexture> mtl_src_tex = (id<MTLTexture>)src_tex_handle;
      [enc copyFromTexture:mtl_src_tex
               sourceSlice:src_slice
               sourceLevel:src_level
              sourceOrigin:mtl_src_origin
                sourceSize:mtl_src_size
                 toTexture:mtl_dst_tex
          destinationSlice:dst_slice
          destinationLevel:dst_level
         destinationOrigin:mtl_dst_origin];
   }
}

void
mtl_compute_set_pipeline_state(mtl_compute_encoder *encoder,
                               mtl_compute_pipeline_state *state_handle)
{
   if (limina_enc_check(encoder, "mtl_compute_set_pipeline_state", NULL) == LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTLComputePipelineState> state = (id<MTLComputePipelineState>)state_handle;
      [enc setComputePipelineState:state];
   }
}

void
mtl_compute_set_argument_table(mtl_compute_encoder *encoder,
                               mtl_argument_table *table)
{
   if (limina_enc_check(encoder, "mtl_compute_set_argument_table", NULL) == LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTL4ArgumentTable> t = (id<MTL4ArgumentTable>)table;
      [enc setArgumentTable:t];
   }
}

void
mtl_dispatch_threads(mtl_compute_encoder *encoder,
                     struct mtl_size grid_size, struct mtl_size local_size)
{
   if (limina_enc_check(encoder, "mtl_dispatch_threads", NULL) == LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      MTLSize thread_count = MTLSizeMake(grid_size.x, grid_size.y, grid_size.z);
      MTLSize threads_per_threadgroup = MTLSizeMake(local_size.x, local_size.y,
                                                    local_size.z);

      [enc dispatchThreads:thread_count threadsPerThreadgroup:threads_per_threadgroup];
   }
}

void
mtl_dispatch_threadgroups_with_indirect_buffer(mtl_compute_encoder *encoder,
                                               uint64_t addr,
                                               struct mtl_size local_size)
{
   if (limina_enc_check(encoder, "mtl_dispatch_threadgroups_with_indirect_buffer", NULL) ==
       LIMINA_ENC_BROKEN)
      return;
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc = (id<MTL4ComputeCommandEncoder>)encoder;
      MTLSize threads_per_threadgroup = MTLSizeMake(local_size.x,
                                                    local_size.y,
                                                    local_size.z);

      [enc dispatchThreadgroupsWithIndirectBuffer:addr threadsPerThreadgroup:threads_per_threadgroup];
   }
}

/* MTLRenderEncoder */

/* Encoder commands */
/* limina: LIMINA_KK_RPLOG=1 — dump the fully-resolved render pass right BEFORE the encoder is
 * created. Creating the encoder is what makes Metal compile its background object, and that
 * compile is where MTLCompilerService aborts on `bitcode_url is NULL ... extension 'ds'`
 * (spikes/agx-compiler-abort). The abort takes the whole process down from inside Metal, so the
 * last line this prints on the aborting thread IS the trigger -- hence print before, and flush.
 *
 * Kept off the RTLOG knob on purpose: RTLOG is per-draw and would bury this in the compositor's
 * traffic. Thread id is logged because several threads encode concurrently and only one aborts. */
static inline bool
limina_kk_rplog_cached(void)
{
   static int v = -1;
   if (v < 0)
      v = getenv("LIMINA_KK_RPLOG") != NULL;
   return v;
}

static void
limina_log_attachment(const char *what, uint32_t idx,
                      MTLRenderPassAttachmentDescriptor *att)
{
   id<MTLTexture> t = att.texture;
   if (!t)
      return;
   /* Two different "not an ordinary private texture" flags, and conflating them cost a run:
    *   linear  = tex.buffer non-nil, i.e. buffer-backed.
    *   iosurf  = tex.iosurface non-nil, i.e. what the scanout path
    *             (LIMINA_KK_MTLTEXTURE_SCANOUT) produces.
    * An IOSurface-backed texture has buffer == nil, so logging only `linear` reported 0 for
    * every scanout attachment and made them look absent from every render pass. */
   fprintf(stderr,
           "[LIMINA-KK-RP]   %s[%u] tex=%p fmt=%lu storage=%lu usage=0x%lx %lux%lu "
           "samples=%lu linear=%d iosurf=%d load=%lu store=%lu\n",
           what, idx, (void *)t, (unsigned long)t.pixelFormat,
           (unsigned long)t.storageMode, (unsigned long)t.usage,
           (unsigned long)t.width, (unsigned long)t.height,
           (unsigned long)t.sampleCount, t.buffer ? 1 : 0, t.iosurface ? 1 : 0,
           (unsigned long)att.loadAction, (unsigned long)att.storeAction);
}

mtl_render_encoder *
mtl_new_render_command_encoder_with_descriptor(
   mtl_command_buffer *command_buffer, mtl_render_pass_descriptor *descriptor)
{
   @autoreleasepool {
      id<MTL4CommandBuffer> cmd = (id<MTL4CommandBuffer>)command_buffer;
      MTL4RenderPassDescriptor *desc = (MTL4RenderPassDescriptor *)descriptor;
      /* limina: an attachment-less pass whose defaultRasterSampleCount is still 0 makes AGX
       * abort the whole process from inside the Metal compiler, uncatchably. Counting the
       * attachments here (rather than inferring "no attachment lines were printed") is what
       * makes the log honest: a header with natt=0 and a header truncated by a concurrent
       * abort used to look identical, and I built a wrong theory on exactly that. When the
       * fatal shape is seen, dump a backtrace -- it names the caller instead of leaving the
       * origin to be guessed at. */
      uint32_t natt = 0;
      for (uint32_t i = 0; i < 8; i++)
         natt += desc.colorAttachments[i].texture ? 1 : 0;
      natt += desc.depthAttachment.texture ? 1 : 0;
      natt += desc.stencilAttachment.texture ? 1 : 0;

      if (natt == 0 && desc.defaultRasterSampleCount == 0) {
         /* Clamping to 1 is what turns an uncatchable process abort into, at worst, a lost
          * render: mtlrp.m measured attachment-less passes as green at sample count 1 and 2
          * and fatal only at 0. Anything reaching here is still an upstream bug -- the pass
          * has nothing to draw into -- so keep warning (rate-limited; the shape can repeat
          * every frame) even though it no longer kills us. */
         static _Atomic uint64_t seen;
         uint64_t n_seen = atomic_fetch_add_explicit(&seen, 1, memory_order_relaxed);
         if (n_seen < 8 || (n_seen & 1023) == 0) {
            fprintf(stderr,
                    "[LIMINA-KK-RP] attachment-less render pass with "
                    "defaultRasterSampleCount=0 (rt=%lux%lu, #%llu) -- clamping to 1; "
                    "without this AGX aborts the process from inside the Metal compiler\n",
                    (unsigned long)desc.renderTargetWidth,
                    (unsigned long)desc.renderTargetHeight,
                    (unsigned long long)n_seen + 1);
            /* Backtrace on the first one only: it names the caller that built the bad
             * descriptor, which is the only thing anyone needs from here. */
            if (n_seen == 0) {
               void *bt[32];
               int nframes = backtrace(bt, 32);
               fflush(stderr);
               backtrace_symbols_fd(bt, nframes, STDERR_FILENO);
            }
         }
         desc.defaultRasterSampleCount = 1;
      }

      if (limina_kk_rplog_cached()) {
         uint64_t tid = 0;
         pthread_threadid_np(NULL, &tid);
         /* imageblockSampleLength is the per-sample tile byte size -- the one field that could
          * plausibly index Apple's blit_fast_clear_gen2_{1,2,4,5,8,16} family directly. */
         fprintf(stderr,
                 "[LIMINA-KK-RP] natt=%u tid=0x%llx rt=%lux%lu arraylen=%lu samples=%lu "
                 "imageblock=%lu tile=%lux%lu tgmem=%lu\n",
                 natt, (unsigned long long)tid,
                 (unsigned long)desc.renderTargetWidth,
                 (unsigned long)desc.renderTargetHeight,
                 (unsigned long)desc.renderTargetArrayLength,
                 (unsigned long)desc.defaultRasterSampleCount,
                 (unsigned long)desc.imageblockSampleLength,
                 (unsigned long)desc.tileWidth, (unsigned long)desc.tileHeight,
                 (unsigned long)desc.threadgroupMemoryLength);
         for (uint32_t i = 0; i < 8; i++)
            limina_log_attachment("color", i, desc.colorAttachments[i]);
         limina_log_attachment("depth", 0, desc.depthAttachment);
         limina_log_attachment("stencil", 0, desc.stencilAttachment);
         fflush(stderr);
      }
      id<MTL4RenderCommandEncoder> new_enc =
         [[cmd renderCommandEncoderWithDescriptor:desc] retain];
      /* limina: RPLOG prints the descriptor BEFORE the encoder exists, so a log
       * reader has no way to tie draws (keyed by encoder) to the pass they
       * belong to. Encoder pointers recycle across command buffers, so binding
       * enc->pass on first sight silently attributes later draws to a dead
       * pass. This line closes the pass; parse it, never infer. */
      if (limina_kk_rplog_cached()) {
         fprintf(stderr, "[LIMINA-KK-RPENC] enc=%p\n", (void *)new_enc);
         fflush(stderr);
      }
      return (mtl_render_encoder *)limina_mtl_note_new(new_enc);
   }
}

void
mtl_set_viewports(mtl_render_encoder *encoder, struct mtl_viewport *viewports,
                  uint32_t count)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLViewport *vps = (MTLViewport *)viewports;
      if (limina_kk_rtlog_cached())
         for (uint32_t i = 0; i < count; i++)
            fprintf(stderr, "[LIMINA-KK-VP] enc=%p vp[%u]=(%.1f,%.1f %.1fx%.1f z=%.2f..%.2f)\n",
                    (void *)enc, i, vps[i].originX, vps[i].originY, vps[i].width,
                    vps[i].height, vps[i].znear, vps[i].zfar);
      [enc setViewports:vps count:count];
   }
}

void
mtl_set_scissor_rects(mtl_render_encoder *encoder,
                      struct mtl_scissor_rect *scissor_rects, uint32_t count)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLScissorRect *rects = (MTLScissorRect *)scissor_rects;
      if (limina_kk_rtlog_cached())
         for (uint32_t i = 0; i < count; i++)
            fprintf(stderr, "[LIMINA-KK-SC] enc=%p sc[%u]=(%lu,%lu %lux%lu)\n",
                    (void *)enc, i, (unsigned long)rects[i].x, (unsigned long)rects[i].y,
                    (unsigned long)rects[i].width, (unsigned long)rects[i].height);
      [enc setScissorRects:rects count:count];
   }
}

void
mtl_render_set_pipeline_state(mtl_render_encoder *encoder,
                              mtl_render_pipeline_state *pipeline)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTLRenderPipelineState> pipe = (id<MTLRenderPipelineState>)pipeline;
      /* limina: the pipeline identity is the discriminator between the cogl
       * journal draw (which inks) and the clutter/pango glyph draw (which does
       * not) -- see spikes/notification-text-corruption. Cheap and needs no
       * fence, unlike every dimension the investigation reached for first. */
      if (limina_kk_rtlog_cached())
         fprintf(stderr, "[LIMINA-KK-PIPE] enc=%p pipe=%p\n", (void *)enc,
                 (void *)pipe);
      [enc setRenderPipelineState:pipe];
   }
}

void
mtl_set_depth_stencil_state(mtl_render_encoder *encoder,
                            mtl_depth_stencil_state *state)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTLDepthStencilState> s = (id<MTLDepthStencilState>)state;
      [enc setDepthStencilState:s];
   }
}

void
mtl_set_stencil_references(mtl_render_encoder *encoder, uint32_t front,
                           uint32_t back)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setStencilFrontReferenceValue:front backReferenceValue:back];
   }
}

void
mtl_set_front_face_winding(mtl_render_encoder *encoder,
                           enum mtl_winding winding)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setFrontFacingWinding:(MTLWinding)winding];
   }
}

void
mtl_set_cull_mode(mtl_render_encoder *encoder, enum mtl_cull_mode mode)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setCullMode:(MTLCullMode)mode];
   }
}

void
mtl_set_visibility_result_mode(mtl_render_encoder *encoder,
                               enum mtl_visibility_result_mode mode,
                               size_t offset)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setVisibilityResultMode:(MTLVisibilityResultMode)mode offset:offset];
   }
}

void
mtl_set_depth_bias(mtl_render_encoder *encoder, float depth_bias,
                   float slope_scale, float clamp)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setDepthBias:depth_bias slopeScale:slope_scale clamp:clamp];
   }
}

void
mtl_set_depth_clip_mode(mtl_render_encoder *encoder,
                        enum mtl_depth_clip_mode mode)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      [enc setDepthClipMode:(MTLDepthClipMode)mode];
   }
}

void
mtl_set_vertex_amplification_count(mtl_render_encoder *encoder,
                                   uint32_t *layer_ids, uint32_t id_count)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLVertexAmplificationViewMapping mappings[32];
      for (uint32_t i = 0u; i < id_count; ++i) {
         mappings[i].renderTargetArrayIndexOffset = layer_ids[i];
         mappings[i].viewportArrayIndexOffset = 0u;
      }
      [enc setVertexAmplificationCount:id_count viewMappings:mappings];
   }
}

void
mtl_render_set_argument_table(mtl_render_encoder *encoder,
                              mtl_argument_table *table,
                              enum mtl_render_stages stages)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTL4ArgumentTable> t = (id<MTL4ArgumentTable>)table;
      [enc setArgumentTable:t atStages:(MTLRenderStages)stages];
   }
}

void
mtl_draw_primitives(mtl_render_encoder *encoder,
                    enum mtl_primitive_type primitve_type, uint32_t vertexStart,
                    uint32_t vertexCount, uint32_t instanceCount,
                    uint32_t baseInstance)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLPrimitiveType type = (MTLPrimitiveType)primitve_type;
      if (limina_kk_rtlog_cached())
         fprintf(stderr, "[LIMINA-KK-DRAW] enc=%p type=%lu count=%u\n", (void *)enc,
                 (unsigned long)type, vertexCount);
      limina_stats_bump(&limina_st_draw);
      [enc drawPrimitives:type vertexStart:vertexStart vertexCount:vertexCount instanceCount:instanceCount baseInstance:baseInstance];
   }
}

void
mtl_draw_indexed_primitives(mtl_render_encoder *encoder,
                            enum mtl_primitive_type primitve_type,
                            uint32_t index_count,
                            enum mtl_index_type index_type, uint64_t index_addr,
                            uint64_t index_buffer_length,
                            uint32_t instance_count, int32_t base_vertex,
                            uint32_t base_instance)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLIndexType ndx_type = (MTLIndexType)index_type;
      MTLPrimitiveType primitive = (MTLPrimitiveType)primitve_type;
      if (limina_kk_rtlog_cached())
         fprintf(stderr,
                 "[LIMINA-KK-DRAW] enc=%p type=%lu indexed count=%u inst=%u\n",
                 (void *)enc, (unsigned long)primitive, index_count,
                 instance_count);
      limina_stats_bump(&limina_st_draw);
      [enc drawIndexedPrimitives:primitive
                      indexCount:index_count
                       indexType:ndx_type
                     indexBuffer:index_addr
               indexBufferLength:index_buffer_length
                   instanceCount:instance_count
                      baseVertex:base_vertex
                    baseInstance:base_instance];
   }
}

void
mtl_draw_primitives_indirect(mtl_render_encoder *encoder,
                             enum mtl_primitive_type primitve_type,
                             uint64_t addr)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      if (limina_kk_rtlog_cached())
         fprintf(stderr, "[LIMINA-KK-DRAW] enc=%p type=%lu indirect\n",
                 (void *)enc, (unsigned long)primitve_type);
      limina_stats_bump(&limina_st_draw);
      [enc drawPrimitives:(MTLPrimitiveType)primitve_type indirectBuffer:addr];
   }
}

void
mtl_draw_indexed_primitives_indirect(mtl_render_encoder *encoder,
                                     enum mtl_primitive_type primitve_type,
                                     enum mtl_index_type index_type,
                                     uint64_t index_addr,
                                     uint64_t index_buffer_length,
                                     uint64_t addr)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      MTLPrimitiveType type = (MTLPrimitiveType)primitve_type;
      MTLIndexType ndx_type = (MTLIndexType)index_type;
      if (limina_kk_rtlog_cached())
         fprintf(stderr,
                 "[LIMINA-KK-DRAW] enc=%p type=%lu indexed-indirect\n",
                 (void *)enc, (unsigned long)type);
      limina_stats_bump(&limina_st_draw);
      [enc drawIndexedPrimitives:type
                       indexType:ndx_type
                     indexBuffer:index_addr
               indexBufferLength:index_buffer_length
                  indirectBuffer:addr];
   }
}

void
mtl_compute_write_timestamp(mtl_compute_encoder *encoder,
                            mtl_counter_heap *heap, uint32_t index)
{
   @autoreleasepool {
      id<MTL4ComputeCommandEncoder> enc =
         (id<MTL4ComputeCommandEncoder>)encoder;
      id<MTL4CounterHeap> h = (id<MTL4CounterHeap>)heap;
      [enc writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
                                intoHeap:h
                                 atIndex:index];
   }
}

void
mtl_render_write_timestamp(mtl_render_encoder *encoder,
                           enum mtl_render_stages stage, mtl_counter_heap *heap,
                           uint32_t index)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      id<MTL4CounterHeap> h = (id<MTL4CounterHeap>)heap;
      [enc writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed
                              afterStage:(MTLRenderStages)stage
                                intoHeap:h
                                 atIndex:index];
   }
}


void
mtl_render_set_color_store_action(mtl_render_encoder *encoder,
                                  enum mtl_store_action action,
                                  uint32_t index)
{
    @autoreleasepool {
        id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
        [enc setColorStoreAction:(MTLStoreAction)action
                         atIndex:index];
    }
}

void mtl_render_set_depth_store_action(mtl_render_encoder *encoder,
                                       enum mtl_store_action action)
{
    @autoreleasepool {
        id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
        [enc setDepthStoreAction:(MTLStoreAction)action];
    }
}

void mtl_render_set_stencil_store_action(mtl_render_encoder *encoder,
                                       enum mtl_store_action action)
{
    @autoreleasepool {
        id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
        [enc setStencilStoreAction:(MTLStoreAction)action];
    }
}

/* limina: see mtl_encoder.h. Labelling only the passes a triggered capture cares about keeps the
 * Xcode frame navigator readable -- the pair under investigation shows up by name. */
void
mtl_render_encoder_set_label(mtl_render_encoder *encoder, const char *label)
{
   @autoreleasepool {
      id<MTL4RenderCommandEncoder> enc = (id<MTL4RenderCommandEncoder>)encoder;
      enc.label = [NSString stringWithUTF8String:label];
   }
}
