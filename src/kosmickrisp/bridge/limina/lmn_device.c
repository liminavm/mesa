/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 *
 * The device: a limina extension context on a virtio-gpu render node, its command ring, the
 * records arena replies land in, and what the host told us about its GPU.
 */
#include "lmn_bridge.h"

#include "kosmickrisp/vulkan/kk_image_layout.h"
#include "../ns_process_info.h"
#include "drm-uapi/drm_fourcc.h"

#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <time.h>
#include <unistd.h>

#include "virtgpu_limina_drm.h"

uint32_t lmn_host_os_version[3];

/* The ring's data area. A command is at most half of it, which bounds an MSL source. */
#define LMN_RING_DATA (8u << 20)
/* The records arena: reply, feedback and mirror slots. The device info goes in its last page. */
#define LMN_RECORDS_SIZE (1u << 20)
#define LMN_RECORDS_INFO_OFFSET (LMN_RECORDS_SIZE - LMN_ARENA_PAGE)
/* How long a reply may take before the device is declared lost. */
#define LMN_REPLY_TIMEOUT_NS (30ull * 1000 * 1000 * 1000)

_Noreturn void
lmn_unimplemented(const char *fn)
{
   fprintf(stderr, "[limina-kk] %s is not implemented on limina's Metal transport\n", fn);
   fflush(stderr);
   abort();
}

void *
lmn_obj_new(struct lmn_device *dev, uint32_t kind, size_t size, lmn_destroy_fn destroy)
{
   struct lmn_obj *o = calloc(1, size);
   if (!o)
      return NULL;
   o->kind = kind;
   atomic_init(&o->refs, 1);
   o->dev = dev;
   o->destroy = destroy;
   return o;
}

struct lmn_obj *
lmn_cast(void *p, uint32_t kind, const char *fn)
{
   struct lmn_obj *o = p;
   if (!o || o->kind != kind) {
      fprintf(stderr, "[limina-kk] %s: expected an object of kind %u, got %p (kind %u)\n", fn,
              kind, p, o ? o->kind : 0);
      abort();
   }
   return o;
}

void *
mtl_retain(void *handle)
{
   struct lmn_obj *o = handle;
   if (o)
      atomic_fetch_add_explicit(&o->refs, 1, memory_order_relaxed);
   return handle;
}

void
mtl_release(void *handle)
{
   struct lmn_obj *o = handle;
   if (!o)
      return;
   int left = atomic_fetch_sub_explicit(&o->refs, 1, memory_order_acq_rel) - 1;
   if (left > 0)
      return;
   if (o->destroy)
      o->destroy(o);
   free(o);
}

void *
limina_mtl_note_new(void *handle)
{
   return handle;
}

void
limina_mtl_class_census_dump(void)
{
}

long
mtl_limina_retain_count(void *obj)
{
   struct lmn_obj *o = obj;
   return o ? atomic_load(&o->refs) : -1;
}

void
mtl_limina_texture_census(char *buf, unsigned long len)
{
   snprintf(buf, len, "texture census: not kept on limina's Metal transport");
}

bool
ns_is_os_version_at_least(uint32_t major, uint32_t minor, uint32_t patch)
{
   const uint32_t *v = lmn_host_os_version;
   if (v[0] != major)
      return v[0] > major;
   if (v[1] != minor)
      return v[1] > minor;
   return v[2] >= patch;
}

/* Destroys a host object: the handle goes back to the host, which keeps the object until the
 * GPU is done with it. */
void
lmn_destroy_host_object(struct lmn_obj *obj)
{
   if (!obj->handle)
      return;
   struct lmn_destroy d = {.handle = obj->handle};
   lmn_push(obj->dev, LMN_OP_DESTROY, &d, sizeof(d));
}

/* -- waiting -- */

void
lmn_backoff(unsigned round)
{
   if (round < 64) {
#if defined(__aarch64__)
      __asm__ volatile("yield");
#endif
      return;
   }
   if (round < 256) {
      sched_yield();
      return;
   }
   struct timespec ts = {.tv_sec = 0, .tv_nsec = round < 2048 ? 20000 : 200000};
   nanosleep(&ts, NULL);
}

static uint64_t
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

bool
lmn_device_lost(struct lmn_device *dev, uint32_t *error, uint64_t *at)
{
   uint64_t pos = 0;
   uint32_t e = lmn_ring_error(&dev->ring, &pos);
   if (!e && dev->lost)
      e = LMN_ERROR_CORRUPT;
   if (e) {
      if (error)
         *error = e;
      if (at)
         *at = pos;
      return true;
   }
   return false;
}

static void
report_lost(struct lmn_device *dev)
{
   static atomic_bool reported;
   uint32_t e;
   uint64_t at;
   if (lmn_device_lost(dev, &e, &at) && !atomic_exchange(&reported, true))
      fprintf(stderr, "[limina-kk] the host lost this GPU context: error %u at ring position %llu\n",
              e, (unsigned long long)at);
}

/* -- the ring -- */

uint32_t
lmn_new_handle(struct lmn_device *dev)
{
   pthread_mutex_lock(&dev->lock);
   uint32_t h = ++dev->next_handle;
   if (h == 0)
      h = ++dev->next_handle;
   pthread_mutex_unlock(&dev->lock);
   return h;
}

void
lmn_flush_locked(struct lmn_device *dev)
{
   if (lmn_ring_publish(&dev->ring))
      *dev->doorbell = 1;
}

bool
lmn_push_locked(struct lmn_device *dev, uint16_t op, const struct lmn_iov *iov, unsigned n)
{
   for (unsigned round = 0;; round++) {
      if (lmn_device_lost(dev, NULL, NULL)) {
         report_lost(dev);
         return false;
      }
      switch (lmn_ring_pushv(&dev->ring, op, iov, n)) {
      case LMN_PUSH_OK:
         return true;
      case LMN_PUSH_TOO_LARGE:
         fprintf(stderr, "[limina-kk] a command of op 0x%x is too large for the ring\n", op);
         abort();
      case LMN_PUSH_FULL:
         /* The host frees space as it consumes; make sure it is consuming. */
         lmn_flush_locked(dev);
         lmn_backoff(round);
         break;
      }
   }
}

bool
lmn_pushv(struct lmn_device *dev, uint16_t op, const struct lmn_iov *iov, unsigned n)
{
   pthread_mutex_lock(&dev->lock);
   bool ok = lmn_push_locked(dev, op, iov, n);
   pthread_mutex_unlock(&dev->lock);
   return ok;
}

bool
lmn_push(struct lmn_device *dev, uint16_t op, const void *payload, uint32_t len)
{
   struct lmn_iov iov = {payload, len};
   return lmn_pushv(dev, op, &iov, 1);
}

struct lmn_reply
lmn_call(struct lmn_device *dev, uint16_t op, const struct lmn_iov *iov, unsigned n,
         struct lmn_slot *slot)
{
   struct lmn_reply *r = slot->ptr;
   __atomic_store_n(&r->done, 0, __ATOMIC_RELAXED);
   pthread_mutex_lock(&dev->lock);
   bool ok = lmn_push_locked(dev, op, iov, n);
   if (ok)
      lmn_flush_locked(dev);
   pthread_mutex_unlock(&dev->lock);

   struct lmn_reply out = {0};
   uint32_t e = 0;
   if (!ok) {
      lmn_device_lost(dev, &e, NULL);
      out.status = e ? e : LMN_ERROR_CORRUPT;
      return out;
   }
   uint64_t start = 0;
   for (unsigned round = 0;; round++) {
      if (lmn_reply_done(r)) {
         memcpy(&out, r, sizeof(out));
         return out;
      }
      if (lmn_device_lost(dev, &e, NULL)) {
         report_lost(dev);
         out.status = e;
         return out;
      }
      if (round == 256)
         start = now_ns();
      if (round > 256 && now_ns() - start > LMN_REPLY_TIMEOUT_NS) {
         fprintf(stderr, "[limina-kk] no reply to op 0x%x after %llu s; the device is lost\n",
                 op, LMN_REPLY_TIMEOUT_NS / 1000000000ull);
         dev->lost = true;
         out.status = LMN_ERROR_CORRUPT;
         return out;
      }
      lmn_backoff(round);
   }
}

/* -- records -- */

bool
lmn_slot_alloc(struct lmn_device *dev, struct lmn_slot *slot)
{
   pthread_mutex_lock(&dev->slot_lock);
   if (dev->free_count == 0) {
      pthread_mutex_unlock(&dev->slot_lock);
      fprintf(stderr, "[limina-kk] out of reply slots\n");
      return false;
   }
   uint32_t i = dev->free_slots[--dev->free_count];
   pthread_mutex_unlock(&dev->slot_lock);
   uint64_t off = (uint64_t)i * LMN_SLOT_SIZE;
   slot->index = i;
   slot->ptr = dev->records->map + off;
   slot->ref = (struct lmn_buf_ref){.arena = dev->records->arena, .offset = off};
   memset(slot->ptr, 0, LMN_SLOT_SIZE);
   return true;
}

void
lmn_slot_free(struct lmn_device *dev, struct lmn_slot *slot)
{
   pthread_mutex_lock(&dev->slot_lock);
   dev->free_slots[dev->free_count++] = slot->index;
   pthread_mutex_unlock(&dev->slot_lock);
   slot->ptr = NULL;
}

/* -- the device -- */

static int
open_node(struct drm_virtgpu_limina_ctx_create *ctx)
{
   const char *forced = getenv("LIMINA_GPU_NODE");
   for (int n = 128; n < 192; n++) {
      char path[64];
      if (forced)
         snprintf(path, sizeof(path), "%s", forced);
      else
         snprintf(path, sizeof(path), "/dev/dri/renderD%d", n);
      int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd >= 0) {
         memset(ctx, 0, sizeof(*ctx));
         if (ioctl(fd, DRM_IOCTL_VIRTGPU_LIMINA_CTX_CREATE, ctx) == 0)
            return fd;
         close(fd);
      }
      if (forced)
         break;
   }
   return -1;
}

static void
device_destroy(struct lmn_obj *obj)
{
   struct lmn_device *dev = (struct lmn_device *)obj;
   lmn_completion_stop(dev);
   if (dev->staging)
      mtl_release(dev->staging);
   if (dev->records)
      mtl_release(dev->records);
   if (dev->pools.next)
      lmn_pools_finish(dev);
   /* The ring's arena goes last: the host reads it until the context is gone, which closing the
    * file does. */
   if (dev->doorbell)
      munmap((void *)dev->doorbell, dev->doorbell_len);
   if (dev->ring_heap)
      munmap(dev->ring_heap->map, dev->ring_heap->size);
   free(dev->ring_heap);
   free(dev->free_slots);
   free(dev->size_cache);
   close(dev->fd);
}

mtl_device *
mtl_device_create(void)
{
   struct drm_virtgpu_limina_ctx_create ctx;
   int fd = open_node(&ctx);
   if (fd < 0)
      return NULL;

   struct lmn_device *dev =
      lmn_obj_new(NULL, LMN_KIND_DEVICE, sizeof(*dev), device_destroy);
   if (!dev) {
      close(fd);
      return NULL;
   }
   dev->base.dev = dev;
   dev->fd = fd;
   pthread_mutex_init(&dev->lock, NULL);
   pthread_mutex_init(&dev->slot_lock, NULL);
   pthread_mutex_init(&dev->commit_lock, NULL);
   pthread_mutex_init(&dev->cache_lock, NULL);
   pthread_mutex_init(&dev->pool_lock, NULL);
   pthread_mutex_init(&dev->staging_lock, NULL);
   list_inithead(&dev->pools);
   pthread_cond_init(&dev->commit_cond, NULL);
   /* Arena ids 0 (none) is never handed out. */
   dev->arena_used[0] = 1;

   long page = sysconf(_SC_PAGESIZE);
   dev->doorbell_len = page > 4096 ? (size_t)page : 4096;
   void *db = mmap(NULL, dev->doorbell_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                   (off_t)ctx.doorbell_offset);
   if (db == MAP_FAILED)
      goto fail;
   dev->doorbell = db;

   dev->ring_heap = lmn_heap_create(dev, LMN_RING_DATA_OFFSET + LMN_RING_DATA);
   dev->records = lmn_heap_create(dev, LMN_RECORDS_SIZE);
   if (!dev->ring_heap || !dev->records)
      goto fail;

   lmn_ring_init(&dev->ring, dev->ring_heap->map, LMN_RING_DATA);
   struct drm_virtgpu_limina_ring_attach attach = {
      .arena_id = dev->ring_heap->arena,
      .data_size = LMN_RING_DATA,
      .offset = 0,
   };
   if (ioctl(fd, DRM_IOCTL_VIRTGPU_LIMINA_RING_ATTACH, &attach) != 0) {
      fprintf(stderr, "[limina-kk] ring attach failed: %s\n", strerror(errno));
      goto fail;
   }

   dev->slot_count = LMN_RECORDS_INFO_OFFSET / LMN_SLOT_SIZE;
   dev->free_slots = calloc(dev->slot_count, sizeof(uint32_t));
   if (!dev->free_slots)
      goto fail;
   for (uint32_t i = 0; i < dev->slot_count; i++)
      dev->free_slots[i] = dev->slot_count - 1 - i;
   dev->free_count = dev->slot_count;

   struct lmn_slot slot;
   if (!lmn_slot_alloc(dev, &slot))
      goto fail;
   struct lmn_query_device q = {
      .dst = {.arena = dev->records->arena, .offset = LMN_RECORDS_INFO_OFFSET},
      .reply = slot.ref,
   };
   struct lmn_iov iov = {&q, sizeof(q)};
   struct lmn_reply r = lmn_call(dev, LMN_OP_QUERY_DEVICE, &iov, 1, &slot);
   lmn_slot_free(dev, &slot);
   if (r.status != 0) {
      fprintf(stderr, "[limina-kk] the host did not describe its GPU (error %u)\n", r.status);
      goto fail;
   }
   memcpy(&dev->info, dev->records->map + LMN_RECORDS_INFO_OFFSET, sizeof(dev->info));
   if (dev->info.version != LMN_WIRE_VERSION) {
      fprintf(stderr, "[limina-kk] the host speaks protocol %u, this driver %u\n",
              dev->info.version, LMN_WIRE_VERSION);
      goto fail;
   }
   memcpy(lmn_host_os_version, dev->info.os_version, sizeof(lmn_host_os_version));

   lmn_completion_start(dev);
   return dev;

fail:
   mtl_release(dev);
   return NULL;
}

void
mtl_start_gpu_capture(mtl_device *mtl_dev_handle, const char *directory)
{
   fprintf(stderr, "[limina-kk] GPU capture is not available from a guest\n");
}

void
mtl_stop_gpu_capture(void)
{
}

static struct lmn_device *
D(mtl_device *dev)
{
   return LMN_CAST(lmn_device, dev, LMN_KIND_DEVICE);
}

void
mtl_device_get_name(mtl_device *dev, char buffer[256])
{
   snprintf(buffer, 256, "%.*s", (int)sizeof(D(dev)->info.name), (char *)D(dev)->info.name);
}

void
mtl_device_get_architecture_name(mtl_device *dev, char buffer[256])
{
   snprintf(buffer, 256, "%.*s", (int)sizeof(D(dev)->info.architecture),
            (char *)D(dev)->info.architecture);
}

uint32_t
mtl_device_get_gpu_apple_family(mtl_device *dev)
{
   return D(dev)->info.apple_family;
}

bool
mtl_device_supports_shareable_textures(mtl_device *device)
{
   /* A module and host that have surfaces make the smallest one; it goes again at once. */
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   struct drm_virtgpu_limina_surface_create s = {
      .width = 1,
      .height = 1,
      .drm_fourcc = DRM_FORMAT_ARGB8888,
      .stride = 4,
   };
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_SURFACE_CREATE, &s) != 0)
      return false;
   struct drm_gem_close c = {.handle = s.handle};
   ioctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &c);
   return true;
}

bool
mtl_device_get_drm_render_node(mtl_device *dev, int64_t *major, int64_t *minor)
{
   struct stat st;
   if (fstat(D(dev)->fd, &st) != 0 || !S_ISCHR(st.st_mode))
      return false;
   *major = major(st.st_rdev);
   *minor = minor(st.st_rdev);
   return true;
}

uint64_t
mtl_device_get_registry_id(mtl_device *dev)
{
   return D(dev)->info.registry_id;
}

bool
mtl_device_supports_sample_count(mtl_device *dev, uint32_t sample_count)
{
   return sample_count < 32 && (D(dev)->info.sample_counts & (1u << sample_count));
}

struct mtl_size
mtl_device_max_threads_per_threadgroup(mtl_device *dev)
{
   const uint32_t *t = D(dev)->info.max_threads_per_threadgroup;
   return (struct mtl_size){.x = t[0], .y = t[1], .z = t[2]};
}

uint32_t
mtl_device_max_threadgroup_memory_length(mtl_device *dev)
{
   return D(dev)->info.max_threadgroup_memory;
}

uint64_t
mtl_device_max_buffer_length(mtl_device *dev)
{
   return D(dev)->info.max_buffer_length;
}

uint64_t
mtl_device_recommended_max_working_set_size(mtl_device *dev)
{
   return D(dev)->info.recommended_max_working_set;
}

uint64_t
mtl_device_current_allocated_size(mtl_device *dev)
{
   return atomic_load(&D(dev)->allocated);
}

uint32_t
mtl_device_max_argument_buffer_sampler_count(mtl_device *dev)
{
   return D(dev)->info.max_argument_buffer_samplers;
}

uint64_t
mtl_device_get_gpu_timestamp(mtl_device *dev)
{
   struct lmn_device *d = D(dev);
   struct lmn_slot slot;
   if (!lmn_slot_alloc(d, &slot))
      return 0;
   struct lmn_query_timestamp q = {.reply = slot.ref};
   struct lmn_iov iov = {&q, sizeof(q)};
   struct lmn_reply r = lmn_call(d, LMN_OP_QUERY_TIMESTAMP, &iov, 1, &slot);
   lmn_slot_free(d, &slot);
   return r.status ? 0 : r.values[0];
}

uint64_t
mtl_device_timestamp_frequency(mtl_device *dev)
{
   return D(dev)->info.timestamp_frequency;
}

bool
mtl_device_supports_timestamps(mtl_device *dev)
{
   return true;
}

bool
mtl_handle_is_iosurface(void *handle)
{
   return false;
}

mtl_texture *
mtl_new_texture_with_descriptor_iosurface(mtl_device *device,
                                          const struct kk_image_layout *layout,
                                          void *iosurface, uint32_t plane)
{
   LMN_UNIMPLEMENTED();
}

void
mtl_heap_buffer_size_and_align_with_length(mtl_device *device, uint64_t *size_B,
                                           uint64_t *align_B)
{
   /* The size is the length; the alignment is the host's for the arena options. */
   *align_B = D(device)->info.heap_buffer_align;
}

uint64_t
mtl_minimum_linear_texture_alignment_for_pixel_format(mtl_device *device,
                                                      enum mtl_pixel_format format)
{
   struct lmn_device *d = D(device);
   uint32_t f = (uint32_t)format;
   if (f < ARRAY_SIZE_LMN(d->linear_align)) {
      pthread_mutex_lock(&d->cache_lock);
      uint64_t cached = d->linear_align[f];
      pthread_mutex_unlock(&d->cache_lock);
      if (cached)
         return cached;
   }
   struct lmn_slot slot;
   if (!lmn_slot_alloc(d, &slot))
      return 0;
   struct lmn_query_linear_alignment q = {.pixel_format = f, .reply = slot.ref};
   struct lmn_iov iov = {&q, sizeof(q)};
   struct lmn_reply r = lmn_call(d, LMN_OP_QUERY_LINEAR_ALIGNMENT, &iov, 1, &slot);
   lmn_slot_free(d, &slot);
   if (r.status)
      return 0;
   if (f < ARRAY_SIZE_LMN(d->linear_align)) {
      pthread_mutex_lock(&d->cache_lock);
      d->linear_align[f] = r.values[0];
      pthread_mutex_unlock(&d->cache_lock);
   }
   return r.values[0];
}

/* Texture size answers, keyed by descriptor. Direct-mapped: a collision only costs a query. */
#define LMN_SIZE_CACHE 512u

struct lmn_size_cache {
   bool valid;
   struct lmn_texture_desc desc;
   uint64_t size, align;
};

static uint32_t
desc_hash(const struct lmn_texture_desc *d)
{
   const uint8_t *p = (const uint8_t *)d;
   uint32_t h = 2166136261u;
   for (size_t i = 0; i < sizeof(*d); i++)
      h = (h ^ p[i]) * 16777619u;
   return h;
}

void
lmn_texture_desc_from_layout(const struct kk_image_layout *layout, struct lmn_texture_desc *desc)
{
   memset(desc, 0, sizeof(*desc));
   desc->texture_type = layout->type;
   desc->pixel_format = layout->format.mtl;
   desc->width = layout->width_px;
   desc->height = layout->height_px;
   desc->depth = layout->depth_px;
   desc->mip_levels = layout->levels;
   desc->array_length = layout->layers;
   desc->sample_count = layout->sample_count_sa;
   desc->usage = layout->usage;
   desc->allow_gpu_optimized = layout->optimized_layout;
}

void
mtl_heap_texture_size_and_align_with_descriptor(mtl_device *device,
                                                struct kk_image_layout *layout,
                                                uint64_t *size_B, uint64_t *align_B)
{
   struct lmn_device *d = D(device);
   struct lmn_query_texture_size q = {0};
   lmn_texture_desc_from_layout(layout, &q.desc);

   uint32_t h = desc_hash(&q.desc) % LMN_SIZE_CACHE;
   pthread_mutex_lock(&d->cache_lock);
   if (!d->size_cache)
      d->size_cache = calloc(LMN_SIZE_CACHE, sizeof(*d->size_cache));
   struct lmn_size_cache *e = d->size_cache ? &d->size_cache[h] : NULL;
   if (e && e->valid && !memcmp(&e->desc, &q.desc, sizeof(q.desc))) {
      if (size_B)
         *size_B = e->size;
      if (align_B)
         *align_B = e->align;
      pthread_mutex_unlock(&d->cache_lock);
      return;
   }
   pthread_mutex_unlock(&d->cache_lock);

   uint64_t size = 0, align = 0;
   struct lmn_slot slot;
   if (lmn_slot_alloc(d, &slot)) {
      q.reply = slot.ref;
      struct lmn_iov iov = {&q, sizeof(q)};
      struct lmn_reply r = lmn_call(d, LMN_OP_QUERY_TEXTURE_SIZE, &iov, 1, &slot);
      lmn_slot_free(d, &slot);
      if (r.status == 0) {
         size = r.values[0];
         align = r.values[1];
         uint64_t seen = atomic_load(&d->max_texture_align);
         while (align > seen && !atomic_compare_exchange_weak(&d->max_texture_align, &seen, align))
            ;
         if (e) {
            pthread_mutex_lock(&d->cache_lock);
            e->valid = true;
            e->desc = q.desc;
            e->size = size;
            e->align = align;
            pthread_mutex_unlock(&d->cache_lock);
         }
      }
   }
   if (size_B)
      *size_B = size;
   if (align_B)
      *align_B = align;
}

uint32_t
mtl_sparse_tile_size_in_bytes(mtl_device *device)
{
   return D(device)->info.sparse_tile_bytes;
}

struct mtl_size
mtl_sparse_tile_size(mtl_device *device, struct kk_image_layout *layout)
{
   LMN_UNIMPLEMENTED();
}

struct mtl_size
mtl_sparse_tile_count(mtl_device *device, struct kk_image_layout *layout,
                      struct mtl_size tile_size)
{
   LMN_UNIMPLEMENTED();
}

mtl_buffer *
mtl_new_buffer_with_bytes_no_copy(mtl_device *device, void *ptr, uint64_t size_B)
{
   LMN_UNIMPLEMENTED();
}

mtl_texture *
mtl_drawable_get_texture(void *drawable_ptr)
{
   LMN_UNIMPLEMENTED();
}
