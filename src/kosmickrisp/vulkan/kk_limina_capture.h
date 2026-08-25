/*
 * Copyright 2026 limina
 * SPDX-License-Identifier: MIT
 */

#ifndef KK_LIMINA_CAPTURE_H
#define KK_LIMINA_CAPTURE_H 1

#include <stdbool.h>
#include <stdint.h>

struct kk_device;

/* limina debug lever: a *triggered* Metal GPU capture, for the notification-text fault where a
 * label offscreen's second render pass rasterises nothing while every state we can read at encode
 * time is byte-identical to the pass that worked. See spikes/notification-text-corruption.
 *
 * Why triggered and not MESA_KK_GPU_CAPTURE: that one runs device-create to device-destroy, i.e.
 * the whole VM lifetime, which is unusable. This one opens on a host-side trigger file and closes
 * a few passes later, so the trace holds the working/failing pair and little else.
 *
 * Metal records at the API layer: commands encoded BEFORE startCapture are absent from the trace.
 * So the capture must open at a command-buffer boundary, never at the pass that interests us --
 * by then the pass is already inside a command buffer that Metal will not show.
 *
 * KK_LIMINA_CAPTURE_REPEAT=1 is the precise form: it opens the window on a matching pass whose
 * colour attachment this lever has already seen rendered. That is exactly the second render of a
 * card's label -- the one that produces nothing -- and because cs_start_render notes the pass
 * before it creates the pass's Metal command buffer, arming there still captures that very pass.
 * The alternative below (arming on the preceding pass size) only gets near it by counting.
 *
 * Two levers together keep the window small enough to be usable. KK_LIMINA_CAPTURE_TRIGGER names
 * a file the harness touches just before posting a card; KK_LIMINA_CAPTURE_ARM=WxH names the pass
 * that actually opens the window, which should be the one rendered immediately BEFORE the pass
 * under investigation (cs_start_render makes a fresh Metal command buffer per pass, so arming on
 * a pass captures the next one). Opening on the trigger alone leaves the capture running across
 * the whole gap before the card arrives: measured, that is gigabytes and a compositor reduced to
 * a crawl, which distorts the very timing being investigated.
 *
 * Requires MTL_CAPTURE_ENABLED=1 in the process environment; without it Metal refuses
 * programmatic capture outright. kk_limina_capture_cmdbuf_begin says so loudly rather than
 * leaving a silently empty trace.
 */

/* Called immediately before mtl_begin_command_buffer. May open the capture. */
void kk_limina_capture_cmdbuf_begin(struct kk_device *dev);

/* Called per colour attachment at render-pass start, before the encoder exists. Counts the
 * passes whose extent matches KK_LIMINA_CAPTURE=WxH and, for those, leaves a name in
 * kk_limina_capture_pending_label for the encoder to pick up. */
void kk_limina_capture_note_pass(uint32_t width, uint32_t height, const void *attachment);

/* Empty string when there is nothing to label. Consumed (and cleared) at encoder creation. */
extern char kk_limina_capture_pending_label[64];

/* Called immediately after mtl_command_queue_commit. May close the capture. */
void kk_limina_capture_after_commit(void);

#endif /* KK_LIMINA_CAPTURE_H */
