/*
 * Copyright 2026 limina
 * SPDX-License-Identifier: MIT
 */

#include "kk_limina_capture.h"

#include "kk_device.h"

#include "kosmickrisp/bridge/mtl_device.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum capture_phase {
   PHASE_OFF = 0,   /* KK_LIMINA_CAPTURE unset: the lever is not armed at all */
   PHASE_WAITING,   /* armed, waiting for the trigger file to appear */
   PHASE_TRIGGERED, /* trigger seen; waiting for the arming pass (KK_LIMINA_CAPTURE_ARM) */
   PHASE_CAPTURING, /* trace is open */
   PHASE_DONE,      /* trace written; one-shot, we never reopen */
};

static enum capture_phase phase = PHASE_OFF;
static bool configured = false;

static uint32_t want_w, want_h;
static uint32_t arm_w, arm_h;   /* the pass that opens the window; 0 = open at the trigger */
static bool pending_start;
static bool skip_first;         /* arm on the SECOND render of a pair, not the first */
static bool want_seen_since_trigger;
static bool arm_on_repeat;      /* arm on a matching pass into an attachment already rendered */

/* Small ring of matching attachments seen so far. Only ever grows within one trigger window, and
 * a wrap merely costs a missed arming opportunity, never a wrong one. */
#define SEEN_MAX 64
static const void *seen_atts[SEEN_MAX];
static unsigned seen_att_n;

static bool
seen_attachment(const void *att)
{
   if (att == NULL)
      return false;
   for (unsigned i = 0; i < seen_att_n; i++)
      if (seen_atts[i] == att)
         return true;
   if (seen_att_n < SEEN_MAX)
      seen_atts[seen_att_n++] = att;
   return false;
}
static const char *dir_base;
static const char *trigger_path;
static unsigned want_passes;  /* close after this many matching passes */
static unsigned max_cbs;      /* hard cap so a run with no matching pass still bounds the file */

static unsigned seen_passes;
static unsigned seen_cbs;
static bool close_after_next_commit;
static char out_dir[512];
static unsigned run_index;   /* one output directory per capture, so re-arming never collides */
static unsigned runs_left;

static void
configure(void)
{
   configured = true;

   const char *spec = getenv("KK_LIMINA_CAPTURE");
   if (spec == NULL)
      return;

   if (sscanf(spec, "%ux%u", &want_w, &want_h) != 2) {
      fprintf(stderr, "[LIMINA-KK-CAPTURE] KK_LIMINA_CAPTURE=\"%s\" is not WxH; disabled\n", spec);
      fflush(stderr);
      return;
   }

   /* A device-scope Metal capture is brutally expensive -- left open across a few seconds it
    * writes gigabytes and drags the compositor to a crawl, which is its own kind of lie about
    * what the GPU was doing. So the window is opened by the pass that immediately PRECEDES the
    * one under investigation: KK's cs_start_render makes a fresh Metal command buffer per pass,
    * and the hook runs before that creation, so arming here captures the very next pass. */
   const char *arm = getenv("KK_LIMINA_CAPTURE_ARM");
   if (arm != NULL && sscanf(arm, "%ux%u", &arm_w, &arm_h) != 2) {
      fprintf(stderr, "[LIMINA-KK-CAPTURE] KK_LIMINA_CAPTURE_ARM=\"%s\" is not WxH; ignored\n", arm);
      arm_w = arm_h = 0;
   }

   /* KK_LIMINA_CAPTURE_SKIP=1 targets the pass that FAILS rather than the one that works. A
    * card's label is rendered twice into the same texture, and it is the second render that
    * produces nothing; the passes cycle 568x44 -> 968x44 -> ... -> 568x44 -> 968x44, so the arming
    * pass to use is the first 568x44 that follows a 968x44. Anchoring on "after a 968x44" rather
    * than counting arming passes keeps this correct if anything else on screen renders 568x44. */
   skip_first = getenv("KK_LIMINA_CAPTURE_SKIP") != NULL &&
                strcmp(getenv("KK_LIMINA_CAPTURE_SKIP"), "0") != 0;
   arm_on_repeat = getenv("KK_LIMINA_CAPTURE_REPEAT") != NULL &&
                   strcmp(getenv("KK_LIMINA_CAPTURE_REPEAT"), "0") != 0;

   /* No directory means the DeveloperTools destination: the capture is handed to an attached
    * Xcode instead of written as a .gputrace. That path does not run the resource download that
    * segfaults on this command stream, so it is the only way to aim a capture at the failing
    * pass -- at the cost of needing a human with Xcode attached before the trigger. */
   dir_base = getenv("KK_LIMINA_CAPTURE_DIR");
   trigger_path = getenv("KK_LIMINA_CAPTURE_TRIGGER");

   const char *p = getenv("KK_LIMINA_CAPTURE_PASSES");
   want_passes = p ? (unsigned)atoi(p) : 2u;
   if (want_passes == 0u)
      want_passes = 2u;

   const char *r = getenv("KK_LIMINA_CAPTURE_RUNS");
   runs_left = r ? (unsigned)atoi(r) : 8u;
   if (runs_left == 0u)
      runs_left = 8u;

   const char *m = getenv("KK_LIMINA_CAPTURE_MAX_CBS");
   max_cbs = m ? (unsigned)atoi(m) : 24u;
   if (max_cbs == 0u)
      max_cbs = 24u;

   /* Metal refuses programmatic capture without this, and the refusal is the only sign: the
    * trace simply never appears. Say it here, once, while there is still time to fix the run. */
   if (getenv("MTL_CAPTURE_ENABLED") == NULL)
      fprintf(stderr, "[LIMINA-KK-CAPTURE] WARNING: MTL_CAPTURE_ENABLED is unset -- Metal will "
                      "refuse to start the capture\n");

   phase = PHASE_WAITING;
   fprintf(stderr,
           "[LIMINA-KK-CAPTURE] armed for %ux%u arm=%ux%u passes=%u max_cbs=%u runs=%u dir=%s "
           "trigger=%s skip=%d repeat=%d\n",
           want_w, want_h, arm_w, arm_h, want_passes, max_cbs, runs_left,
           dir_base ? dir_base : "(attached developer tool)",
           trigger_path ? trigger_path : "(none: opens at the first command buffer)", skip_first, arm_on_repeat);
   fflush(stderr);
}

void
kk_limina_capture_cmdbuf_begin(struct kk_device *dev)
{
   if (!configured)
      configure();

   if (phase == PHASE_WAITING) {
      /* The trigger file says "the harness is about to post a card". It does not open the window
       * by itself when an arming pass is configured -- that would leave the capture running
       * across the whole gap before the card arrives. */
      if (trigger_path != NULL && access(trigger_path, F_OK) != 0)
         return;
      if (trigger_path != NULL)
         unlink(trigger_path); /* one-shot */
      if (arm_w != 0u || arm_h != 0u) {
         phase = PHASE_TRIGGERED;
         want_seen_since_trigger = false;
         seen_att_n = 0;
         fprintf(stderr, "[LIMINA-KK-CAPTURE] triggered; waiting for a %ux%u pass\n", arm_w,
                 arm_h);
         fflush(stderr);
         return;
      }
   } else if (phase == PHASE_TRIGGERED) {
      if (!pending_start)
         return;
      pending_start = false;
   } else {
      return;
   }

   ++run_index;
   if (dir_base != NULL) {
      /* A fresh directory per run: mtl_start_gpu_capture names the trace
       * <dir>/<process>.gputrace, and Metal refuses to start when that bundle already exists. */
      snprintf(out_dir, sizeof(out_dir), "%s/cap-%d-%u", dir_base, (int)getpid(), run_index);
      if (mkdir(out_dir, 0755) != 0 && errno != EEXIST) {
         fprintf(stderr, "[LIMINA-KK-CAPTURE] cannot create %s; disabled\n", out_dir);
         fflush(stderr);
         phase = PHASE_DONE;
         return;
      }
   } else {
      snprintf(out_dir, sizeof(out_dir), "(attached developer tool)");
   }

   mtl_start_gpu_capture(dev->mtl_handle, dir_base != NULL ? out_dir : NULL);
   phase = PHASE_CAPTURING;
   seen_passes = 0;
   seen_cbs = 0;
   close_after_next_commit = false;
   fprintf(stderr, "[LIMINA-KK-CAPTURE] started, writing into %s\n", out_dir);
   fflush(stderr);
}

char kk_limina_capture_pending_label[64];

void
kk_limina_capture_note_pass(uint32_t width, uint32_t height, const void *attachment)
{
   bool matches = width == want_w && height == want_h;

   if (phase == PHASE_TRIGGERED) {
      if (arm_on_repeat) {
         /* The pass whose result is wrong is the one drawn into a label that has already been
          * rendered once. Nothing about the preceding pass identifies it; the attachment does. */
         if (matches && seen_attachment(attachment)) {
            pending_start = true;
            fprintf(stderr, "[LIMINA-KK-CAPTURE] repeat render of attachment %p -- opening\n",
                    attachment);
            fflush(stderr);
         }
         return;
      }
      if (matches)
         want_seen_since_trigger = true;
      else if (width == arm_w && height == arm_h &&
               (!skip_first || want_seen_since_trigger))
         pending_start = true;
      return;
   }

   if (phase != PHASE_CAPTURING || !matches)
      return;

   seen_passes++;
   snprintf(kk_limina_capture_pending_label, sizeof(kk_limina_capture_pending_label),
            "LIMINA %ux%u pass #%u", width, height, seen_passes);
   fprintf(stderr, "[LIMINA-KK-CAPTURE] pass #%u matches %ux%u (cb %u)\n", seen_passes, width,
           height, seen_cbs);
   fflush(stderr);

   /* Close after the commit of the command buffer that holds this pass, not at the next pass
    * start: stopping with an encoded-but-uncommitted command buffer truncates the trace. */
   if (seen_passes >= want_passes)
      close_after_next_commit = true;
}

void
kk_limina_capture_after_commit(void)
{
   if (phase != PHASE_CAPTURING)
      return;

   seen_cbs++;
   if (!close_after_next_commit && seen_cbs < max_cbs)
      return;

   mtl_stop_gpu_capture();
   bool hit_cap = seen_cbs >= max_cbs && !close_after_next_commit;

   /* Re-arm when the harness can trigger again. Capture is itself a synchronisation, and
    * everything that synchronises has cured this bug so far, so a capture may well come back
    * clean -- several attempts per boot is the difference between one shot and a measurement. */
   if (--runs_left > 0u && trigger_path != NULL)
      phase = PHASE_WAITING;
   else
      phase = PHASE_DONE;

   fprintf(stderr, "[LIMINA-KK-CAPTURE] stopped: passes=%u commits=%u%s -> %s (%u run%s left)\n",
           seen_passes, seen_cbs, hit_cap ? " (hit max_cbs)" : "", out_dir, runs_left,
           runs_left == 1u ? "" : "s");
   fflush(stderr);
}
