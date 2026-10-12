/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 *
 * The queue: command buffers and their allocators, commits, queue-ordered events, and the
 * completion thread that runs Metal's feedback handlers once the host reports a commit done.
 */
#include "lmn_bridge.h"

#include <string.h>

#define LMN_MAX_HANDLERS 8

struct lmn_handler {
   mtl_feedback_handler_callback callback;
   void *data;
};

struct lmn_pending {
   struct lmn_pending *next;
   uint64_t seqno;
   struct lmn_slot feedback;
   struct lmn_handler handlers[LMN_MAX_HANDLERS];
   unsigned handler_count;
};

/* -- deferred slots -- */

void
lmn_slot_free_after_next_commit(struct lmn_device *dev, struct lmn_slot *slot)
{
   pthread_mutex_lock(&dev->lock);
   uint64_t seqno = dev->seqno + 1;
   pthread_mutex_unlock(&dev->lock);

   pthread_mutex_lock(&dev->slot_lock);
   if (dev->deferred_count == dev->deferred_cap) {
      uint32_t cap = dev->deferred_cap ? dev->deferred_cap * 2 : 64;
      void *p = realloc(dev->deferred, cap * sizeof(*dev->deferred));
      if (!p) {
         /* Keeping the slot forever is the safe failure. */
         pthread_mutex_unlock(&dev->slot_lock);
         return;
      }
      dev->deferred = p;
      dev->deferred_cap = cap;
   }
   dev->deferred[dev->deferred_count++] =
      (struct lmn_deferred_slot){.index = slot->index, .seqno = seqno};
   pthread_mutex_unlock(&dev->slot_lock);
}

static void
free_deferred(struct lmn_device *dev, uint64_t completed)
{
   pthread_mutex_lock(&dev->slot_lock);
   uint32_t kept = 0;
   for (uint32_t i = 0; i < dev->deferred_count; i++) {
      if (dev->deferred[i].seqno <= completed)
         dev->free_slots[dev->free_count++] = dev->deferred[i].index;
      else
         dev->deferred[kept++] = dev->deferred[i];
   }
   dev->deferred_count = kept;
   pthread_mutex_unlock(&dev->slot_lock);
}

/* -- completion -- */

static void
run_handlers(struct lmn_device *dev, struct lmn_pending *c, const struct lmn_feedback *fb,
             uint32_t lost)
{
   char message[96] = "";
   struct mtl_feedback_data data = {
      .gpu_start = fb ? fb->gpu_start : 0.0,
      .gpu_end = fb ? fb->gpu_end : 0.0,
      .error = MTL_COMMAND_QUEUE_ERROR_NONE,
   };
   if (lost) {
      snprintf(message, sizeof(message), "limina: the host lost the GPU context (error %u)", lost);
      data.error = MTL_COMMAND_QUEUE_ERROR_INTERNAL;
      data.error_message = message;
   } else if (fb && fb->error) {
      snprintf(message, sizeof(message), "limina: the host's commit failed (Metal error %u)",
               fb->error);
      data.error = (enum mtl_command_queue_error)fb->error;
      data.error_message = message;
   }
   for (unsigned i = 0; i < c->handler_count; i++) {
      data.user_data = c->handlers[i].data;
      c->handlers[i].callback(&data);
   }
}

static void *
completion_main(void *arg)
{
   struct lmn_device *dev = arg;
   unsigned round = 0;
   pthread_mutex_lock(&dev->commit_lock);
   for (;;) {
      while (!dev->commits_head && !dev->stopping)
         pthread_cond_wait(&dev->commit_cond, &dev->commit_lock);
      struct lmn_pending *c = dev->commits_head;
      if (!c)
         break;
      pthread_mutex_unlock(&dev->commit_lock);

      const struct lmn_feedback *fb = c->feedback.ptr;
      uint32_t lost = 0;
      bool done = __atomic_load_n(&fb->done, __ATOMIC_ACQUIRE) == LMN_REPLY_DONE;
      if (!done && lmn_device_lost(dev, &lost, NULL))
         done = true;
      if (!done) {
         lmn_backoff(round++);
         pthread_mutex_lock(&dev->commit_lock);
         continue;
      }
      round = 0;
      run_handlers(dev, c, lost ? NULL : fb, lost);
      lmn_slot_free(dev, &c->feedback);
      free_deferred(dev, c->seqno);

      pthread_mutex_lock(&dev->commit_lock);
      dev->commits_head = c->next;
      if (!dev->commits_head)
         dev->commits_tail = NULL;
      free(c);
   }
   pthread_mutex_unlock(&dev->commit_lock);
   return NULL;
}

void
lmn_completion_start(struct lmn_device *dev)
{
   dev->completion_running = pthread_create(&dev->completion, NULL, completion_main, dev) == 0;
}

void
lmn_completion_stop(struct lmn_device *dev)
{
   if (!dev->completion_running)
      return;
   pthread_mutex_lock(&dev->commit_lock);
   dev->stopping = true;
   pthread_cond_broadcast(&dev->commit_cond);
   pthread_mutex_unlock(&dev->commit_lock);
   /* Pending commits finish first: their handlers release Vulkan objects. */
   pthread_join(dev->completion, NULL);
   dev->completion_running = false;
}

/* -- commit options -- */

struct lmn_commit_options {
   struct lmn_obj base;
   struct lmn_handler handlers[LMN_MAX_HANDLERS];
   unsigned handler_count;
};

mtl_commit_options *
mtl_new_commit_options(void)
{
   return lmn_obj_new(NULL, LMN_KIND_COMMIT_OPTIONS, sizeof(struct lmn_commit_options), NULL);
}

void
mtl_commit_options_add_feedback_handler(mtl_commit_options *options,
                                        mtl_feedback_handler_callback callback, void *data)
{
   struct lmn_commit_options *o =
      LMN_CAST(lmn_commit_options, options, LMN_KIND_COMMIT_OPTIONS);
   if (o->handler_count == LMN_MAX_HANDLERS) {
      fprintf(stderr, "[limina-kk] more than %u feedback handlers on one commit\n",
              LMN_MAX_HANDLERS);
      abort();
   }
   o->handlers[o->handler_count++] = (struct lmn_handler){callback, data};
}

/* -- the queue -- */

mtl_command_queue *
mtl_new_command_queue(mtl_device *device)
{
   return lmn_obj_new(LMN_CAST(lmn_device, device, LMN_KIND_DEVICE), LMN_KIND_QUEUE,
                      sizeof(struct lmn_obj), NULL);
}

void
mtl_command_queue_add_residency_set(mtl_command_queue *cmd_queue, mtl_residency_set *set)
{
}

void
mtl_command_queue_remove_residency_set(mtl_command_queue *cmd_queue, mtl_residency_set *set)
{
}

static void
queue_event(mtl_command_queue *queue, mtl_event *event, uint64_t value, bool wait)
{
   struct lmn_obj *q = lmn_cast(queue, LMN_KIND_QUEUE, __func__);
   struct lmn_obj *e = event;
   if (!e || (e->kind != LMN_KIND_EVENT && e->kind != LMN_KIND_SHARED_EVENT)) {
      fprintf(stderr, "[limina-kk] queue event on a non-event %p\n", event);
      abort();
   }
   struct lmn_queue_event c = {.event = e->handle, .wait = wait, .value = value};
   lmn_push(q->dev, LMN_OP_QUEUE_EVENT, &c, sizeof(c));
}

void
mtl_signal_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   queue_event(queue, event, value, false);
}

void
mtl_wait_for_event(mtl_command_queue *queue, mtl_event *event, uint64_t value)
{
   queue_event(queue, event, value, true);
}

void
mtl_command_queue_commit(mtl_command_queue *queue, mtl_command_buffer **command_buffers,
                         uint32_t count, mtl_commit_options *options)
{
   struct lmn_obj *q = lmn_cast(queue, LMN_KIND_QUEUE, __func__);
   struct lmn_device *dev = q->dev;
   struct lmn_commit_options *o =
      options ? LMN_CAST(lmn_commit_options, options, LMN_KIND_COMMIT_OPTIONS) : NULL;

   struct lmn_pending *c = calloc(1, sizeof(*c));
   if (!c || !lmn_slot_alloc(dev, &c->feedback)) {
      fprintf(stderr, "[limina-kk] out of memory committing\n");
      abort();
   }
   if (o) {
      memcpy(c->handlers, o->handlers, sizeof(c->handlers));
      c->handler_count = o->handler_count;
   }

   bool ok = true;
   pthread_mutex_lock(&dev->lock);
   for (uint32_t i = 0; i < count && ok; i++) {
      struct lmn_command_buffer *cb =
         LMN_CAST(lmn_command_buffer, command_buffers[i], LMN_KIND_COMMAND_BUFFER);
      ok = lmn_command_buffer_replay_locked(dev, cb);
   }
   c->seqno = ++dev->seqno;
   struct lmn_commit w = {.seqno = c->seqno, .feedback = c->feedback.ref};
   if (ok)
      ok = lmn_push_locked(dev, LMN_OP_COMMIT, &(struct lmn_iov){&w, sizeof(w)}, 1);
   lmn_flush_locked(dev);
   pthread_mutex_unlock(&dev->lock);
   /* A failed commit still completes, as lost, so its handlers run. */
   (void)ok;

   pthread_mutex_lock(&dev->commit_lock);
   if (dev->commits_tail)
      dev->commits_tail->next = c;
   else
      dev->commits_head = c;
   dev->commits_tail = c;
   pthread_cond_signal(&dev->commit_cond);
   pthread_mutex_unlock(&dev->commit_lock);
}

void
mtl_command_queue_wait_for_drawable(mtl_command_queue *queue, void *drawable)
{
   LMN_UNIMPLEMENTED();
}

void
mtl_command_queue_signal_drawable(mtl_command_queue *queue, void *drawable)
{
   LMN_UNIMPLEMENTED();
}

/* -- allocators: the host pools its own -- */

mtl_command_allocator *
mtl_new_command_allocator(mtl_device *device)
{
   return lmn_obj_new(LMN_CAST(lmn_device, device, LMN_KIND_DEVICE), LMN_KIND_ALLOCATOR,
                      sizeof(struct lmn_obj), NULL);
}

void
mtl_command_allocator_reset(mtl_command_allocator *allocator)
{
}

uint64_t
mtl_command_allocator_allocated_size(mtl_command_allocator *allocator)
{
   return 0;
}

int
limina_kk_alloc_stats_on(void)
{
   return 0;
}

uint64_t
limina_kk_alloc_track_commit(void **cbs, uint32_t count)
{
   return 0;
}

void
limina_kk_alloc_track_complete(uint64_t token)
{
}

/* -- command buffers -- */

static _Atomic uint64_t lmn_cb_serial;

void
lmn_cb_ref(struct lmn_command_buffer *cb, void *obj)
{
   struct lmn_obj *o = obj;
   /* Encoders name the same objects over and over; most repeats are back to back. */
   if (!o || (cb->ref_count && cb->refs[cb->ref_count - 1] == o))
      return;
   if (cb->ref_count == cb->ref_cap) {
      uint32_t cap = cb->ref_cap ? cb->ref_cap * 2 : 64;
      void *p = realloc(cb->refs, cap * sizeof(*cb->refs));
      if (!p) {
         cb->failed = true;
         return;
      }
      cb->refs = p;
      cb->ref_cap = cap;
   }
   cb->refs[cb->ref_count++] = mtl_retain(o);
}

static void
drop_refs(struct lmn_command_buffer *cb)
{
   for (uint32_t i = 0; i < cb->ref_count; i++)
      mtl_release(cb->refs[i]);
   cb->ref_count = 0;
}

static void
command_buffer_destroy(struct lmn_obj *obj)
{
   struct lmn_command_buffer *cb = (struct lmn_command_buffer *)obj;
   drop_refs(cb);
   free(cb->refs);
   free(cb->data);
}

mtl_command_buffer *
mtl_new_command_buffer(mtl_device *device)
{
   struct lmn_command_buffer *cb =
      lmn_obj_new(LMN_CAST(lmn_device, device, LMN_KIND_DEVICE), LMN_KIND_COMMAND_BUFFER,
                  sizeof(*cb), command_buffer_destroy);
   if (cb)
      cb->serial = atomic_fetch_add(&lmn_cb_serial, 1) + 1;
   return cb;
}

void
mtl_begin_command_buffer(mtl_command_buffer *command_buffer, mtl_command_allocator *allocator)
{
   struct lmn_command_buffer *cb =
      LMN_CAST(lmn_command_buffer, command_buffer, LMN_KIND_COMMAND_BUFFER);
   /* A command buffer is recorded again only once its last commit has completed. */
   drop_refs(cb);
   cb->len = 0;
   cb->failed = false;
   cb->serial = atomic_fetch_add(&lmn_cb_serial, 1) + 1;
}

void
mtl_end_command_buffer(mtl_command_buffer *command_buffer)
{
}

void
mtl_command_resolve_counter_heap(mtl_command_buffer *command_buffer, mtl_counter_heap *heap,
                                 uint32_t first_index, uint32_t count, uint64_t dst_addr)
{
   LMN_UNIMPLEMENTED();
}

void
mtl_command_buffer_set_label(mtl_command_buffer *cb, const char *label)
{
}

void
lmn_cb_append(struct lmn_command_buffer *cb, uint16_t op, const struct lmn_iov *iov, unsigned n)
{
   uint32_t payload = 0;
   for (unsigned i = 0; i < n; i++)
      payload += iov[i].len;
   uint32_t len = lmn_ring_encoded_len(payload);
   if (cb->len + len > cb->cap) {
      size_t cap = cb->cap ? cb->cap * 2 : 16384;
      while (cap < cb->len + len)
         cap *= 2;
      uint8_t *p = realloc(cb->data, cap);
      if (!p) {
         cb->failed = true;
         return;
      }
      cb->data = p;
      cb->cap = cap;
   }
   uint8_t *p = cb->data + cb->len;
   struct lmn_cmd_header h = {.op = op, .flags = 0, .len = len};
   memcpy(p, &h, sizeof(h));
   uint8_t *q = p + sizeof(h);
   for (unsigned i = 0; i < n; i++) {
      memcpy(q, iov[i].base, iov[i].len);
      q += iov[i].len;
   }
   memset(q, 0, (size_t)(p + len - q));
   cb->len += len;
}

bool
lmn_command_buffer_replay_locked(struct lmn_device *dev, struct lmn_command_buffer *cb)
{
   if (cb->failed) {
      fprintf(stderr, "[limina-kk] a command buffer ran out of memory while recording\n");
      dev->lost = true;
      return false;
   }
   if (!lmn_push_locked(dev, LMN_OP_BEGIN_COMMAND_BUFFER, NULL, 0))
      return false;
   for (size_t at = 0; at < cb->len;) {
      struct lmn_cmd_header h;
      memcpy(&h, cb->data + at, sizeof(h));
      struct lmn_iov iov = {cb->data + at + sizeof(h), h.len - (uint32_t)sizeof(h)};
      if (!lmn_push_locked(dev, h.op, &iov, 1))
         return false;
      at += h.len;
   }
   return lmn_push_locked(dev, LMN_OP_END_COMMAND_BUFFER, NULL, 0);
}
