/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 *
 * KosmicKrisp's Metal bridge for a Linux guest of limina: every mtl_* call becomes a command of
 * limina's Metal-level GPU protocol, written into a ring in memory the host's replayer reads, and
 * the replayer makes the Metal call. limina_gpu_wire.h and limina_gpu_ring.h are copies of the
 * protocol's C rendering (generated from limina's crates/limina-gpu-wire); virtgpu_limina_drm.h
 * is the UAPI of limina's virtio-gpu extension.
 *
 * Model:
 * - Every bridge object is an lmn_obj: a kind, a reference count (mtl_retain/mtl_release) and
 *   the device it belongs to. Objects the host owns also carry the u32 handle the guest assigned
 *   them; their last release sends DESTROY.
 * - Heaps are arenas: guest memory the host has made a Metal heap of, mapped here. Buffers are
 *   ranges of a heap and exist only in the guest.
 * - Object creation goes into the ring at once. Command buffers record into a private byte stream
 *   in the ring's own encoding, copied into the ring when committed, so a command buffer can be
 *   recorded on any thread while others commit.
 * - Anything that needs an answer (a resource ID, a size, whether MSL compiled) names a reply
 *   slot in a records arena and waits for the host to mark it done.
 * - A ring error is a lost device: every wait checks it.
 */
#ifndef LMN_BRIDGE_H
#define LMN_BRIDGE_H 1

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "limina_gpu_ring.h"
#include "limina_gpu_wire.h"
#include "virtgpu_limina_drm.h"

#include "../mtl_bridge.h"
#include "util/hash_table.h"
#include "util/list.h"
#include "util/vma.h"

#define ARRAY_SIZE_LMN(a) (sizeof(a) / sizeof((a)[0]))

enum lmn_kind {
   LMN_KIND_DEVICE = 1,
   LMN_KIND_HEAP,
   LMN_KIND_BUFFER,
   LMN_KIND_TEXTURE,
   LMN_KIND_SAMPLER_DESC,
   LMN_KIND_SAMPLER,
   LMN_KIND_COMPILER,
   LMN_KIND_LIBRARY,
   LMN_KIND_FUNCTION,
   LMN_KIND_COMPUTE_PIPELINE,
   LMN_KIND_RENDER_PIPELINE_DESC,
   LMN_KIND_RENDER_PIPELINE,
   LMN_KIND_RENDER_PASS_DESC,
   LMN_KIND_STENCIL_DESC,
   LMN_KIND_DEPTH_STENCIL_DESC,
   LMN_KIND_DEPTH_STENCIL,
   LMN_KIND_ARGUMENT_TABLE_DESC,
   LMN_KIND_ARGUMENT_TABLE,
   LMN_KIND_FENCE,
   LMN_KIND_EVENT,
   LMN_KIND_SHARED_EVENT,
   LMN_KIND_RESIDENCY_SET,
   LMN_KIND_QUEUE,
   LMN_KIND_ALLOCATOR,
   LMN_KIND_COMMAND_BUFFER,
   LMN_KIND_ENCODER,
   LMN_KIND_COMMIT_OPTIONS,
   LMN_KIND_COUNTER_HEAP,
};

struct lmn_device;
struct lmn_obj;

typedef void (*lmn_destroy_fn)(struct lmn_obj *obj);

struct lmn_obj {
   uint32_t kind;
   /* The host object's handle, or 0 for guest-only objects. */
   uint32_t handle;
   atomic_int refs;
   struct lmn_device *dev;
   lmn_destroy_fn destroy;
};

/* Allocates a zeroed object of `size` bytes with one reference. */
void *lmn_obj_new(struct lmn_device *dev, uint32_t kind, size_t size,
                  lmn_destroy_fn destroy);

/* `p` as an lmn_obj of `kind`; aborts on anything else. */
struct lmn_obj *lmn_cast(void *p, uint32_t kind, const char *fn);
#define LMN_CAST(type, p, kind) ((struct type *)lmn_cast((p), (kind), __func__))

/* Aborts, naming a bridge function this implementation does not provide. */
_Noreturn void lmn_unimplemented(const char *fn);
#define LMN_UNIMPLEMENTED() lmn_unimplemented(__func__)

/* A slot in the records arena: host-written replies, feedback and shared-event mirrors. */
struct lmn_slot {
   void *ptr;
   struct lmn_buf_ref ref;
   uint32_t index;
};

#define LMN_SLOT_SIZE 64u

/* Arenas small heaps are carved from, and heaps above which an arena is a heap's own. */
#define LMN_POOL_ARENA (64ull << 20)
#define LMN_POOL_MAX_HEAP (16ull << 20)
/* util_vma_heap takes 0 for failure; pool offsets are biased by this. */
#define LMN_POOL_BIAS (1ull << 32)

struct lmn_arena {
   uint32_t id;
   uint64_t size;
   uint64_t gpu_base;
   uint64_t mmap_offset;
   uint8_t *map;
   /* A shared arena's GEM object (its export, or the object it was imported from), which it is
    * mapped and exported through; 0 for one in the arena window. */
   uint32_t gem;
};

/* An arena shared by small heaps. */
struct lmn_pool {
   struct list_head link;
   struct lmn_arena arena;
   struct util_vma_heap vma;
   uint64_t used;
};

/* A heap: `size` bytes at `offset` in arena `arena`, which is either a pool's or its own. The
 * other fields describe the heap's first byte. */
struct lmn_heap {
   struct lmn_obj base;
   uint32_t arena;
   uint64_t offset;
   uint64_t size;
   uint64_t gpu_base;
   uint64_t mmap_offset;
   uint8_t *map;
   struct lmn_pool *pool;
   struct lmn_arena own;
};

struct lmn_buffer {
   struct lmn_obj base;
   struct lmn_heap *heap;
   uint64_t offset;
   uint64_t length;
};

struct lmn_texture {
   struct lmn_obj base;
   uint64_t resource_id;
   /* A shareable texture's GEM object: the surface it is made over. */
   uint32_t gem;
   struct mtl_texture_props props;
   /* What keeps the memory alive: the heap or buffer it is in, or the texture it views. */
   struct lmn_obj *parent;
};

struct lmn_argument_table {
   struct lmn_obj base;
   uint32_t max;
   uint32_t set_mask;
   uint32_t dirty;
   /* The command buffer recording this table's state was last written into. */
   uint64_t last_serial;
   uint64_t addr[32];
};

struct lmn_device {
   struct lmn_obj base;
   int fd;
   volatile uint32_t *doorbell;
   size_t doorbell_len;

   /* Serialises the ring, handle allocation and arena ids. */
   pthread_mutex_t lock;
   struct lmn_ring ring;
   struct lmn_heap *ring_heap;
   uint32_t next_handle;
   uint64_t arena_used[VIRTGPU_LIMINA_MAX_ARENAS / 64];
   uint64_t seqno;
   /* A queue wait was pushed after the last commit: a signal pushed now is not known to follow
    * that commit's completion. */
   bool waited_since_commit;
   bool lost;
   /* Orders timestamp resolves, which no barrier reaches: an encoder that wrote a timestamp
    * updates it as it ends, a resolve waits on it and updates it, and the encoder after a
    * resolve waits on it. Made with the first counter heap. */
   _Atomic(struct lmn_obj *) ts_fence;

   /* Host-written records. */
   pthread_mutex_t slot_lock;
   struct lmn_heap *records;
   uint32_t *free_slots;
   uint32_t free_count;
   uint32_t slot_count;

   struct lmn_device_info info;
   _Atomic uint64_t allocated;

   /* Pooled arenas for small heaps; see lmn_memory.c. */
   pthread_mutex_t pool_lock;
   struct list_head pools;
   /* GEM handle -> how many shared arenas hold it: importing what this file exported gives back
    * the same handle, which must close only with the last of them. Under `pool_lock`. */
   struct hash_table_u64 *gems;
   /* The surfaces of GEM objects that are ones (struct lmn_gem_surface), for as long as a handle
    * is held: the host gives a context a surface once. Under `pool_lock`. */
   struct hash_table_u64 *gem_surfaces;
   /* The largest placement alignment the host has reported for a texture. */
   _Atomic uint64_t max_texture_align;

   /* Staging for texture <-> host-pointer transfers, under `lock`. */
   struct lmn_heap *staging;

   /* Commits waiting for their feedback; see lmn_queue.c. */
   pthread_mutex_t commit_lock;
   pthread_cond_t commit_cond;
   struct lmn_pending *commits_head, *commits_tail;
   /* Slots freed once commit `seqno` has completed: (index, seqno) pairs. */
   struct lmn_deferred_slot {
      uint32_t index;
      uint64_t seqno;
   } *deferred;
   uint32_t deferred_count, deferred_cap;
   pthread_t completion;
   bool completion_running;
   bool stopping;

   /* Answers to size queries, which KK asks per image. */
   pthread_mutex_t cache_lock;
   struct lmn_size_cache *size_cache;
   uint64_t linear_align[1024];
};

/* -- the ring (lmn_device.c) -- */

/* Assigns a handle for a new host object. */
uint32_t lmn_new_handle(struct lmn_device *dev);

/* Pushes a command; with `dev->lock` held. Waits for space when the ring is full. Returns false
 * if the device is lost, in which case nothing was pushed. */
bool lmn_push_locked(struct lmn_device *dev, uint16_t op, const struct lmn_iov *iov,
                     unsigned n);

/* Publishes what was pushed and rings the doorbell if the host parked; with `dev->lock` held. */
void lmn_flush_locked(struct lmn_device *dev);

/* lmn_push_locked of one payload, taking the lock. */
bool lmn_push(struct lmn_device *dev, uint16_t op, const void *payload, uint32_t len);
bool lmn_pushv(struct lmn_device *dev, uint16_t op, const struct lmn_iov *iov, unsigned n);

/* Pushes and publishes a command whose reply is `slot`, waits for the reply and returns it.
 * `status` is 0, the LMN_ERROR_* the command failed with, or the one that lost the device. */
struct lmn_reply lmn_call(struct lmn_device *dev, uint16_t op, const struct lmn_iov *iov,
                          unsigned n, struct lmn_slot *slot);

/* Whether the device is lost, and why. */
bool lmn_device_lost(struct lmn_device *dev, uint32_t *error, uint64_t *at);

/* Sleeps or spins between polls of host-written memory; `round` counts the polls so far. */
void lmn_backoff(unsigned round);

/* -- records -- */
bool lmn_slot_alloc(struct lmn_device *dev, struct lmn_slot *slot);
void lmn_slot_free(struct lmn_device *dev, struct lmn_slot *slot);
/* Frees `slot` once a commit made after this call has completed. */
void lmn_slot_free_after_next_commit(struct lmn_device *dev, struct lmn_slot *slot);

/* -- heaps (lmn_memory.c) -- */
/* A heap that is a whole arena of its own. */
struct lmn_heap *lmn_heap_create(struct lmn_device *dev, uint64_t size);
/* Returns the pooled arenas to the host; every pooled heap must be gone. */
void lmn_pools_finish(struct lmn_device *dev);

/* Notes that `event` (a shared event) is signaled to `value` on the queue right after commit
 * `seqno`; with `dev->lock` held. What mtl_shared_event_export_sync_file goes by. */
void lmn_shared_event_note_signal(void *event, uint64_t value, uint64_t seqno);

/* -- objects -- */
void lmn_destroy_host_object(struct lmn_obj *obj);
void lmn_texture_desc_from_layout(const struct kk_image_layout *layout,
                                  struct lmn_texture_desc *desc);

/* -- queue (lmn_queue.c) -- */
void lmn_completion_start(struct lmn_device *dev);
void lmn_completion_stop(struct lmn_device *dev);

/* -- command buffers (lmn_queue.c, lmn_encoder.c) -- */

/* A command buffer: commands in the ring's own encoding, copied into the ring at commit. */
struct lmn_command_buffer {
   struct lmn_obj base;
   uint8_t *data;
   size_t len, cap;
   /* Distinguishes recordings, for argument-table state; see lmn_encoder.c. */
   uint64_t serial;
   /* An append failed for lack of memory; the commit reports a lost device. */
   bool failed;
   /* The objects the stream names, held until the next recording: their DESTROY must not reach
    * the ring before the commit that uses them, and KosmicKrisp releases some (depth-stencil
    * states, meta pipelines) while still recording, as Metal encoders retain what they are set. */
   struct lmn_obj **refs;
   uint32_t ref_count, ref_cap;
   /* A counter-heap resolve is the last thing recorded: the next encoder waits on its fence. */
   bool resolved;
};

/* Holds `obj` for as long as `cb`'s current recording. */
void lmn_cb_ref(struct lmn_command_buffer *cb, void *obj);

/* Appends a command to `cb`'s stream. */
void lmn_cb_append(struct lmn_command_buffer *cb, uint16_t op, const struct lmn_iov *iov,
                   unsigned n);
/* Copies a command buffer's commands into the ring; with `dev->lock` held. */
bool lmn_command_buffer_replay_locked(struct lmn_device *dev, struct lmn_command_buffer *cb);

/* The host's macOS version, from the last device created. */
extern uint32_t lmn_host_os_version[3];

#endif /* LMN_BRIDGE_H */
