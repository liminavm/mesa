/*
 * Copyright 2026 Gustavo Noronha Silva
 * SPDX-License-Identifier: MIT
 *
 * Memory and the objects that live in it: heaps (arenas), buffers (ranges of a heap), textures
 * and their views, samplers, sync objects, argument tables and residency sets.
 */
#include "lmn_bridge.h"

#include "kosmickrisp/vulkan/kk_image_layout.h"
#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_formats.h"
#include "vulkan/vulkan_core.h"

#include <drm/virtgpu_drm.h>
#include <errno.h>
#include <inttypes.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "virtgpu_limina_drm.h"

/* -- heaps -- */

static uint32_t
arena_id_alloc(struct lmn_device *dev)
{
   pthread_mutex_lock(&dev->lock);
   uint32_t id = 0;
   for (uint32_t w = 0; w < ARRAY_SIZE_LMN(dev->arena_used) && !id; w++) {
      if (dev->arena_used[w] != ~0ull) {
         uint32_t b = (uint32_t)__builtin_ctzll(~dev->arena_used[w]);
         dev->arena_used[w] |= 1ull << b;
         id = w * 64 + b;
      }
   }
   pthread_mutex_unlock(&dev->lock);
   return id;
}

static void
arena_id_free(struct lmn_device *dev, uint32_t id)
{
   pthread_mutex_lock(&dev->lock);
   dev->arena_used[id / 64] &= ~(1ull << (id % 64));
   pthread_mutex_unlock(&dev->lock);
}

/* An arena: guest memory the host made one Metal heap of, mapped here once. */
static bool
arena_create(struct lmn_device *dev, uint64_t size, struct lmn_arena *out)
{
   uint32_t id = arena_id_alloc(dev);
   if (!id) {
      fprintf(stderr, "[limina-kk] out of arena ids\n");
      return false;
   }
   struct drm_virtgpu_limina_arena_create a = {.arena_id = id, .size = size};
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_ARENA_CREATE, &a) != 0) {
      arena_id_free(dev, id);
      return false;
   }
   void *map = mmap(NULL, a.size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd,
                    (off_t)a.mmap_offset);
   if (map == MAP_FAILED) {
      struct drm_virtgpu_limina_arena_destroy d = {.arena_id = id};
      ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_ARENA_DESTROY, &d);
      arena_id_free(dev, id);
      return false;
   }
   *out = (struct lmn_arena){
      .id = id,
      .size = a.size,
      .gpu_base = a.gpu_base,
      .mmap_offset = a.mmap_offset,
      .map = map,
   };
   atomic_fetch_add(&dev->allocated, a.size);
   return true;
}

/* What the host said a surface GEM object is: its resource and shape. */
struct lmn_gem_surface {
   uint32_t res_handle;
   uint32_t width;
   uint32_t height;
   uint32_t drm_fourcc;
   uint32_t stride;
};

static void
gem_ref(struct lmn_device *dev, uint32_t gem)
{
   pthread_mutex_lock(&dev->pool_lock);
   if (!dev->gems)
      dev->gems = _mesa_hash_table_u64_create(NULL);
   uintptr_t n = (uintptr_t)_mesa_hash_table_u64_search(dev->gems, gem);
   _mesa_hash_table_u64_insert(dev->gems, gem, (void *)(n + 1));
   pthread_mutex_unlock(&dev->pool_lock);
}

static void
gem_unref(struct lmn_device *dev, uint32_t gem)
{
   pthread_mutex_lock(&dev->pool_lock);
   uintptr_t n = (uintptr_t)_mesa_hash_table_u64_search(dev->gems, gem);
   if (n > 1)
      _mesa_hash_table_u64_insert(dev->gems, gem, (void *)(n - 1));
   else
      _mesa_hash_table_u64_remove(dev->gems, gem);
   struct lmn_gem_surface *surface = NULL;
   if (n <= 1 && dev->gem_surfaces) {
      surface = _mesa_hash_table_u64_search(dev->gem_surfaces, gem);
      _mesa_hash_table_u64_remove(dev->gem_surfaces, gem);
   }
   pthread_mutex_unlock(&dev->pool_lock);
   free(surface);
   if (n <= 1) {
      struct drm_gem_close c = {.handle = gem};
      ioctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &c);
   }
}

/* Every mapping of the arena must be gone: the host reuses its range. */
static void
arena_destroy(struct lmn_device *dev, struct lmn_arena *a)
{
   munmap(a->map, a->size);
   struct drm_virtgpu_limina_arena_destroy d = {.arena_id = a->id};
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_ARENA_DESTROY, &d) != 0)
      fprintf(stderr, "[limina-kk] arena %u destroy failed: %s\n", a->id, strerror(errno));
   if (a->gem)
      gem_unref(dev, a->gem);
   atomic_fetch_sub(&dev->allocated, a->size);
   arena_id_free(dev, a->id);
}

/* Maps a shared arena through its GEM object and fills in the rest of `a`. */
static bool
arena_map_gem(struct lmn_device *dev, struct lmn_arena *a)
{
   struct drm_virtgpu_map m = {.handle = a->gem};
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_MAP, &m) != 0)
      return false;
   void *map =
      mmap(NULL, a->size, PROT_READ | PROT_WRITE, MAP_SHARED, dev->fd, (off_t)m.offset);
   if (map == MAP_FAILED)
      return false;
   a->mmap_offset = m.offset;
   a->map = map;
   atomic_fetch_add(&dev->allocated, a->size);
   return true;
}

/* Undoes a shared arena whose mapping failed: the arena id is the host's, the GEM ours. */
static void
arena_unwind(struct lmn_device *dev, struct lmn_arena *a)
{
   struct drm_virtgpu_limina_arena_destroy d = {.arena_id = a->id};
   ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_ARENA_DESTROY, &d);
   if (a->gem)
      gem_unref(dev, a->gem);
   arena_id_free(dev, a->id);
}

/* An arena other processes can share: exported as a GEM object, mapped through it. */
static bool
arena_create_shared(struct lmn_device *dev, uint64_t size, struct lmn_arena *out)
{
   uint32_t id = arena_id_alloc(dev);
   if (!id)
      return false;
   struct drm_virtgpu_limina_arena_create c = {
      .arena_id = id,
      .flags = VIRTGPU_LIMINA_ARENA_SHARED,
      .size = size,
   };
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_ARENA_CREATE, &c) != 0) {
      arena_id_free(dev, id);
      return false;
   }
   *out = (struct lmn_arena){.id = id, .size = c.size, .gpu_base = c.gpu_base};
   struct drm_virtgpu_limina_arena_export e = {.arena_id = id};
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_ARENA_EXPORT, &e) != 0) {
      arena_unwind(dev, out);
      return false;
   }
   out->gem = e.handle;
   gem_ref(dev, out->gem);
   if (!arena_map_gem(dev, out)) {
      arena_unwind(dev, out);
      return false;
   }
   return true;
}

/* The arena a dma-buf another context exported is. */
static bool
arena_import(struct lmn_device *dev, int fd, struct lmn_arena *out)
{
   struct drm_prime_handle p = {.fd = fd};
   if (ioctl(dev->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &p) != 0)
      return false;
   gem_ref(dev, p.handle);
   uint32_t id = arena_id_alloc(dev);
   if (!id) {
      gem_unref(dev, p.handle);
      return false;
   }
   struct drm_virtgpu_limina_arena_import i = {.arena_id = id, .handle = p.handle};
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_ARENA_IMPORT, &i) != 0) {
      gem_unref(dev, p.handle);
      arena_id_free(dev, id);
      return false;
   }
   *out = (struct lmn_arena){
      .id = id, .size = i.size, .gpu_base = i.gpu_base, .gem = p.handle};
   if (!arena_map_gem(dev, out)) {
      arena_unwind(dev, out);
      return false;
   }
   return true;
}

static void
heap_place(struct lmn_heap *h, const struct lmn_arena *a, uint64_t offset, uint64_t size)
{
   h->arena = a->id;
   h->offset = offset;
   h->size = size;
   h->gpu_base = a->gpu_base + offset;
   h->mmap_offset = a->mmap_offset + offset;
   h->map = a->map + offset;
}

static void
heap_destroy(struct lmn_obj *obj)
{
   struct lmn_heap *h = (struct lmn_heap *)obj;
   struct lmn_device *dev = obj->dev;
   struct lmn_pool *p = h->pool;
   if (!p) {
      arena_destroy(dev, &h->own);
      return;
   }
   pthread_mutex_lock(&dev->pool_lock);
   util_vma_heap_free(&p->vma, LMN_POOL_BIAS + h->offset, h->size);
   p->used -= h->size;
   /* An empty pool goes back to the host, unless it is the last one. */
   bool drop = p->used == 0 && dev->pools.next != dev->pools.prev;
   if (drop)
      list_del(&p->link);
   pthread_mutex_unlock(&dev->pool_lock);
   if (drop) {
      util_vma_heap_finish(&p->vma);
      arena_destroy(dev, &p->arena);
      free(p);
   }
}

/* A heap that is a whole arena of its own. */
struct lmn_heap *
lmn_heap_create(struct lmn_device *dev, uint64_t size)
{
   struct lmn_heap *h = lmn_obj_new(dev, LMN_KIND_HEAP, sizeof(*h), heap_destroy);
   if (!h)
      return NULL;
   if (!arena_create(dev, size, &h->own)) {
      free(h);
      return NULL;
   }
   heap_place(h, &h->own, 0, h->own.size);
   return h;
}

/* A heap that is a shared arena of its own: made here (`fd` < 0) or imported from `fd`. */
static struct lmn_heap *
heap_shared(struct lmn_device *dev, uint64_t size, int fd)
{
   struct lmn_heap *h = lmn_obj_new(dev, LMN_KIND_HEAP, sizeof(*h), heap_destroy);
   if (!h)
      return NULL;
   bool ok = fd < 0 ? arena_create_shared(dev, size, &h->own) : arena_import(dev, fd, &h->own);
   if (!ok) {
      free(h);
      return NULL;
   }
   heap_place(h, &h->own, 0, h->own.size);
   return h;
}

mtl_heap *
mtl_new_heap_exportable(mtl_device *device, uint64_t size)
{
   return heap_shared(LMN_CAST(lmn_device, device, LMN_KIND_DEVICE), size, -1);
}

mtl_heap *
mtl_new_heap_from_fd(mtl_device *device, int fd)
{
   if (fd < 0)
      return NULL;
   return heap_shared(LMN_CAST(lmn_device, device, LMN_KIND_DEVICE), 0, fd);
}

int
mtl_heap_export_fd(mtl_heap *heap)
{
   struct lmn_heap *h = LMN_CAST(lmn_heap, heap, LMN_KIND_HEAP);
   if (h->pool || !h->own.gem)
      return -1;
   struct drm_prime_handle p = {.handle = h->own.gem, .flags = DRM_CLOEXEC | DRM_RDWR, .fd = -1};
   if (ioctl(h->base.dev->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &p) != 0)
      return -1;
   return p.fd;
}

static struct lmn_heap *
heap_from_pool(struct lmn_device *dev, uint64_t size)
{
   /* Placed textures need their own alignment; the largest the host has reported covers it. */
   uint64_t align = atomic_load(&dev->max_texture_align);
   if (align < LMN_ARENA_PAGE)
      align = LMN_ARENA_PAGE;
   size = (size + LMN_ARENA_PAGE - 1) & ~(uint64_t)(LMN_ARENA_PAGE - 1);

   struct lmn_heap *h = lmn_obj_new(dev, LMN_KIND_HEAP, sizeof(*h), heap_destroy);
   if (!h)
      return NULL;
   pthread_mutex_lock(&dev->pool_lock);
   list_for_each_entry(struct lmn_pool, p, &dev->pools, link) {
      uint64_t at = util_vma_heap_alloc(&p->vma, size, align);
      if (at) {
         p->used += size;
         h->pool = p;
         heap_place(h, &p->arena, at - LMN_POOL_BIAS, size);
         pthread_mutex_unlock(&dev->pool_lock);
         return h;
      }
   }
   struct lmn_pool *p = calloc(1, sizeof(*p));
   if (!p || !arena_create(dev, LMN_POOL_ARENA, &p->arena)) {
      pthread_mutex_unlock(&dev->pool_lock);
      free(p);
      free(h);
      return NULL;
   }
   util_vma_heap_init(&p->vma, LMN_POOL_BIAS, p->arena.size);
   p->vma.alloc_high = false;
   /* Newest first: the pools that fill up drift to the back. */
   list_add(&p->link, &dev->pools);
   uint64_t at = util_vma_heap_alloc(&p->vma, size, align);
   p->used += size;
   h->pool = p;
   heap_place(h, &p->arena, at - LMN_POOL_BIAS, size);
   pthread_mutex_unlock(&dev->pool_lock);
   return h;
}

void
lmn_pools_finish(struct lmn_device *dev)
{
   list_for_each_entry_safe(struct lmn_pool, p, &dev->pools, link) {
      if (p->used)
         fprintf(stderr, "[limina-kk] %llu bytes of heaps outlived their device\n",
                 (unsigned long long)p->used);
      list_del(&p->link);
      util_vma_heap_finish(&p->vma);
      arena_destroy(dev, &p->arena);
      free(p);
   }
   if (dev->gems) {
      _mesa_hash_table_u64_destroy(dev->gems);
      dev->gems = NULL;
   }
}

mtl_heap *
mtl_new_heap(mtl_device *device, uint64_t size, enum mtl_resource_options resource_options)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   /* Each arena takes one of a fixed number of ids, and KosmicKrisp makes a heap per
    * allocation, so small heaps share pooled arenas; large ones still get their own. */
   if (size > LMN_POOL_MAX_HEAP)
      return lmn_heap_create(dev, size);
   return heap_from_pool(dev, size);
}

void *
mtl_heap_map_at(mtl_heap *heap, void *addr, uint64_t size_B)
{
   struct lmn_heap *h = LMN_CAST(lmn_heap, heap, LMN_KIND_HEAP);
   if (size_B > h->size)
      return NULL;
   void *map = mmap(addr, size_B, PROT_READ | PROT_WRITE, MAP_SHARED | (addr ? MAP_FIXED : 0),
                    h->base.dev->fd, (off_t)h->mmap_offset);
   return map == MAP_FAILED ? NULL : map;
}

uint64_t
mtl_heap_get_size(mtl_heap *heap)
{
   return LMN_CAST(lmn_heap, heap, LMN_KIND_HEAP)->size;
}

void
mtl_heap_set_label(mtl_heap *heap, const char *label)
{
}

void
mtl_resource_set_label(mtl_resource *resource, const char *label)
{
}

void
mtl_limina_set_label(void *obj, const char *label)
{
}

/* -- buffers -- */

static void
buffer_destroy(struct lmn_obj *obj)
{
   mtl_release(((struct lmn_buffer *)obj)->heap);
}

mtl_buffer *
mtl_new_buffer_with_length(mtl_heap *heap, uint64_t size_B, uint64_t offset_B)
{
   struct lmn_heap *h = LMN_CAST(lmn_heap, heap, LMN_KIND_HEAP);
   if (offset_B + size_B > h->size)
      return NULL;
   struct lmn_buffer *b = lmn_obj_new(h->base.dev, LMN_KIND_BUFFER, sizeof(*b), buffer_destroy);
   if (!b)
      return NULL;
   b->heap = mtl_retain(h);
   b->offset = offset_B;
   b->length = size_B;
   return b;
}

uint64_t
mtl_buffer_get_length(mtl_buffer *buffer)
{
   return LMN_CAST(lmn_buffer, buffer, LMN_KIND_BUFFER)->length;
}

uint64_t
mtl_buffer_get_gpu_address(mtl_buffer *buffer)
{
   struct lmn_buffer *b = LMN_CAST(lmn_buffer, buffer, LMN_KIND_BUFFER);
   return b->heap->gpu_base + b->offset;
}

void *
mtl_get_contents(mtl_buffer *buffer)
{
   struct lmn_buffer *b = LMN_CAST(lmn_buffer, buffer, LMN_KIND_BUFFER);
   return b->heap->map + b->offset;
}

/* -- textures -- */

static void
texture_destroy(struct lmn_obj *obj)
{
   struct lmn_texture *t = (struct lmn_texture *)obj;
   lmn_destroy_host_object(obj);
   mtl_release(t->parent);
   /* After the DESTROY: the host's texture keeps the surface, the GEM object the resource. */
   if (t->gem)
      gem_unref(obj->dev, t->gem);
}

static struct lmn_texture *
texture_new(struct lmn_device *dev, struct lmn_obj *parent)
{
   struct lmn_texture *t = lmn_obj_new(dev, LMN_KIND_TEXTURE, sizeof(*t), texture_destroy);
   if (!t)
      return NULL;
   t->base.handle = lmn_new_handle(dev);
   t->parent = mtl_retain(parent);
   return t;
}

/* Sends a create command whose reply carries the new object's resource ID. On failure the
 * object is freed without a DESTROY, since the host never made it. */
static bool
create_with_id(struct lmn_device *dev, uint16_t op, const struct lmn_iov *iov, unsigned n,
               struct lmn_slot *slot, uint64_t *id, const char *what)
{
   struct lmn_reply r = lmn_call(dev, op, iov, n, slot);
   lmn_slot_free(dev, slot);
   if (r.status != 0) {
      fprintf(stderr, "[limina-kk] the host refused a %s (error %u)\n", what, r.status);
      return false;
   }
   if (id)
      *id = r.values[0];
   return true;
}

static void
props_from_desc(struct lmn_texture *t, const struct lmn_texture_desc *d)
{
   t->props = (struct mtl_texture_props){
      .width = d->width,
      .height = d->height,
      .pixel_format = d->pixel_format,
      .texture_type = d->texture_type,
      .sample_count = d->sample_count,
      .mip_levels = d->mip_levels,
      .array_length = d->array_length,
      .usage = d->usage,
   };
}

static mtl_texture *
texture_create(struct lmn_device *dev, struct lmn_obj *parent, uint32_t arena, uint64_t offset,
               uint64_t bytes_per_row, const struct kk_image_layout *layout)
{
   struct lmn_texture *t = texture_new(dev, parent);
   struct lmn_slot slot;
   if (!t)
      return NULL;
   if (!lmn_slot_alloc(dev, &slot)) {
      t->base.handle = 0;
      mtl_release(t);
      return NULL;
   }
   struct lmn_create_texture c = {
      .handle = t->base.handle,
      .arena = arena,
      .offset = offset,
      .bytes_per_row = bytes_per_row,
      .reply = slot.ref,
   };
   lmn_texture_desc_from_layout(layout, &c.desc);
   /* As the Metal bridge does: Metal takes a linear 2D array of one layer only as 2D. */
   if (bytes_per_row && c.desc.texture_type == MTL_TEXTURE_TYPE_2D_ARRAY &&
       c.desc.array_length == 1)
      c.desc.texture_type = MTL_TEXTURE_TYPE_2D;
   props_from_desc(t, &c.desc);
   struct lmn_iov iov = {&c, sizeof(c)};
   if (!create_with_id(dev, LMN_OP_CREATE_TEXTURE, &iov, 1, &slot, &t->resource_id, "texture")) {
      t->base.handle = 0;
      mtl_release(t);
      return NULL;
   }
   return t;
}

mtl_texture *
mtl_new_texture_with_descriptor(mtl_heap *heap, const struct kk_image_layout *layout,
                                uint64_t offset)
{
   struct lmn_heap *h = LMN_CAST(lmn_heap, heap, LMN_KIND_HEAP);
   return texture_create(h->base.dev, &h->base, h->arena, h->offset + offset, 0, layout);
}

mtl_texture *
mtl_new_texture_with_descriptor_linear(mtl_buffer *buffer, const struct kk_image_layout *layout,
                                       uint64_t offset)
{
   struct lmn_buffer *b = LMN_CAST(lmn_buffer, buffer, LMN_KIND_BUFFER);
   return texture_create(b->base.dev, &b->base, b->heap->arena,
                         b->heap->offset + b->offset + offset,
                         layout->linear_stride_B, layout);
}

/* The DRM fourcc a shareable texture's surface is in, or 0 for a format it cannot be in. */
static uint32_t
shareable_fourcc(enum mtl_pixel_format format)
{
   switch (format) {
   case MTL_PIXEL_FORMAT_B8G8R8A8_UNORM:
   case MTL_PIXEL_FORMAT_B8G8R8A8_SRGB:
      return DRM_FORMAT_ARGB8888;
   case MTL_PIXEL_FORMAT_R8G8B8A8_UNORM:
   case MTL_PIXEL_FORMAT_R8G8B8A8_SRGB:
      return DRM_FORMAT_ABGR8888;
   default:
      return 0;
   }
}

bool
mtl_pixel_format_is_shareable(enum mtl_pixel_format format)
{
   return shareable_fourcc(format) != 0;
}

/* Notes what the host said surface GEM object `gem` is, for later imports of it in this file. */
static void
gem_surface_note(struct lmn_device *dev, uint32_t gem, struct lmn_gem_surface s)
{
   struct lmn_gem_surface *copy = malloc(sizeof(*copy));
   if (!copy)
      return;
   *copy = s;
   pthread_mutex_lock(&dev->pool_lock);
   if (!dev->gem_surfaces)
      dev->gem_surfaces = _mesa_hash_table_u64_create(NULL);
   struct lmn_gem_surface *old = _mesa_hash_table_u64_search(dev->gem_surfaces, gem);
   _mesa_hash_table_u64_insert(dev->gem_surfaces, gem, copy);
   pthread_mutex_unlock(&dev->pool_lock);
   free(old);
}

static bool
gem_surface_find(struct lmn_device *dev, uint32_t gem, struct lmn_gem_surface *out)
{
   pthread_mutex_lock(&dev->pool_lock);
   struct lmn_gem_surface *s =
      dev->gem_surfaces ? _mesa_hash_table_u64_search(dev->gem_surfaces, gem) : NULL;
   if (s)
      *out = *s;
   pthread_mutex_unlock(&dev->pool_lock);
   return s != NULL;
}

/* Whether `layout` describes a single 2D image a surface can be the storage of. */
static bool
surface_layout_ok(const struct kk_image_layout *layout)
{
   return shareable_fourcc(layout->format.mtl) && layout->levels == 1 && layout->layers == 1 &&
          layout->sample_count_sa == 1 && layout->depth_px == 1 &&
          layout->linear_stride_B <= UINT32_MAX;
}

/* A texture over surface `res_handle`, holding the reference to `gem` the caller took. */
static mtl_texture *
texture_over_surface(struct lmn_device *dev, uint32_t gem, uint32_t res_handle,
                     const struct kk_image_layout *layout)
{
   struct lmn_texture *t = texture_new(dev, NULL);
   struct lmn_slot slot;
   if (!t) {
      gem_unref(dev, gem);
      return NULL;
   }
   t->gem = gem;
   if (!lmn_slot_alloc(dev, &slot)) {
      t->base.handle = 0;
      mtl_release(t);
      return NULL;
   }
   struct lmn_create_texture_from_surface c = {
      .handle = t->base.handle,
      .surface = res_handle,
      .reply = slot.ref,
   };
   lmn_texture_desc_from_layout(layout, &c.desc);
   /* Metal takes a linear 2D array of one layer only as 2D. */
   if (c.desc.texture_type == MTL_TEXTURE_TYPE_2D_ARRAY)
      c.desc.texture_type = MTL_TEXTURE_TYPE_2D;
   props_from_desc(t, &c.desc);
   struct lmn_iov iov = {&c, sizeof(c)};
   if (!create_with_id(dev, LMN_OP_CREATE_TEXTURE_FROM_SURFACE, &iov, 1, &slot, &t->resource_id,
                       "texture over a surface")) {
      t->base.handle = 0;
      mtl_release(t);
      return NULL;
   }
   return t;
}

mtl_texture *
mtl_new_texture_from_fd(mtl_device *device, int fd, const struct kk_image_layout *layout)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   if (fd < 0 || !surface_layout_ok(layout))
      return NULL;
   struct drm_prime_handle p = {.fd = fd};
   if (ioctl(dev->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &p) != 0)
      return NULL;
   gem_ref(dev, p.handle);
   struct lmn_gem_surface s;
   if (!gem_surface_find(dev, p.handle, &s)) {
      struct drm_virtgpu_limina_surface_import im = {.handle = p.handle};
      if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_SURFACE_IMPORT, &im) != 0) {
         gem_unref(dev, p.handle);
         return NULL;
      }
      s = (struct lmn_gem_surface){
         .res_handle = im.res_handle,
         .width = im.width,
         .height = im.height,
         .drm_fourcc = im.drm_fourcc,
         .stride = im.stride,
      };
      gem_surface_note(dev, p.handle, s);
   }
   /* The texture takes the surface's own rows; an image laid out at another pitch would shear. */
   if (s.width != layout->width_px || s.height != layout->height_px ||
       s.drm_fourcc != shareable_fourcc(layout->format.mtl) ||
       (layout->linear && layout->linear_stride_B != s.stride)) {
      fprintf(stderr,
              "[limina-kk] a %ux%u surface (fourcc 0x%x, pitch %u) is not the %ux%u image "
              "(Metal format %u, pitch %" PRIu64 ") it was imported as\n",
              s.width, s.height, s.drm_fourcc, s.stride, layout->width_px, layout->height_px,
              layout->format.mtl, (uint64_t)(layout->linear ? layout->linear_stride_B : 0));
      gem_unref(dev, p.handle);
      return NULL;
   }
   return texture_over_surface(dev, p.handle, s.res_handle, layout);
}

mtl_texture *
mtl_new_texture_shareable(mtl_device *device, const struct kk_image_layout *layout)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   if (!surface_layout_ok(layout) || !layout->linear)
      return NULL;
   struct drm_virtgpu_limina_surface_create s = {
      .width = layout->width_px,
      .height = layout->height_px,
      .drm_fourcc = shareable_fourcc(layout->format.mtl),
      .stride = (uint32_t)layout->linear_stride_B,
   };
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_SURFACE_CREATE, &s) != 0)
      return NULL;
   gem_ref(dev, s.handle);
   gem_surface_note(dev, s.handle,
                    (struct lmn_gem_surface){
                       .res_handle = s.res_handle,
                       .width = s.width,
                       .height = s.height,
                       .drm_fourcc = s.drm_fourcc,
                       .stride = s.stride,
                    });
   return texture_over_surface(dev, s.handle, s.res_handle, layout);
}

int
mtl_texture_export_fd(mtl_texture *texture)
{
   struct lmn_texture *t = LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE);
   if (!t->gem)
      return -1;
   struct drm_prime_handle p = {.handle = t->gem, .flags = DRM_CLOEXEC | DRM_RDWR, .fd = -1};
   if (ioctl(t->base.dev->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &p) != 0)
      return -1;
   return p.fd;
}

uint64_t
mtl_texture_get_gpu_resource_id(mtl_texture *texture)
{
   return LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE)->resource_id;
}

void
mtl_texture_get_props(mtl_texture *texture, struct mtl_texture_props *props)
{
   *props = LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE)->props;
}

static uint32_t
view_type(VkImageViewType type, uint8_t samples)
{
   switch (type) {
   case VK_IMAGE_VIEW_TYPE_1D:
      return MTL_TEXTURE_TYPE_1D;
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
      return MTL_TEXTURE_TYPE_1D_ARRAY;
   case VK_IMAGE_VIEW_TYPE_2D:
      return samples > 1 ? MTL_TEXTURE_TYPE_2D_MULTISAMPLE : MTL_TEXTURE_TYPE_2D;
   case VK_IMAGE_VIEW_TYPE_CUBE:
      return MTL_TEXTURE_TYPE_CUBE;
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return MTL_TEXTURE_TYPE_CUBE_ARRAY;
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
      return samples > 1 ? MTL_TEXTURE_TYPE_2D_ARRAY_MULTISAMPLE : MTL_TEXTURE_TYPE_2D_ARRAY;
   case VK_IMAGE_VIEW_TYPE_3D:
      return MTL_TEXTURE_TYPE_3D;
   default:
      fprintf(stderr, "[limina-kk] unsupported VkImageViewType %d\n", type);
      abort();
   }
}

static uint32_t
swizzle(uint8_t pipe)
{
   switch (pipe) {
   case PIPE_SWIZZLE_X:
      return MTL_TEXTURE_SWIZZLE_RED;
   case PIPE_SWIZZLE_Y:
      return MTL_TEXTURE_SWIZZLE_GREEN;
   case PIPE_SWIZZLE_Z:
      return MTL_TEXTURE_SWIZZLE_BLUE;
   case PIPE_SWIZZLE_W:
      return MTL_TEXTURE_SWIZZLE_ALPHA;
   case PIPE_SWIZZLE_0:
      return MTL_TEXTURE_SWIZZLE_ZERO;
   default:
      return MTL_TEXTURE_SWIZZLE_ONE;
   }
}

static mtl_texture *
view_create(struct lmn_texture *parent, struct lmn_create_texture_view *c)
{
   struct lmn_device *dev = parent->base.dev;
   struct lmn_texture *v = texture_new(dev, &parent->base);
   struct lmn_slot slot;
   if (!v)
      return NULL;
   if (!lmn_slot_alloc(dev, &slot)) {
      v->base.handle = 0;
      mtl_release(v);
      return NULL;
   }
   c->handle = v->base.handle;
   c->parent = parent->base.handle;
   c->reply = slot.ref;
   v->props = parent->props;
   v->props.pixel_format = c->pixel_format;
   v->props.texture_type = c->texture_type;
   v->props.mip_levels = c->level_count;
   v->props.array_length = c->slice_count;
   struct lmn_iov iov = {c, sizeof(*c)};
   if (!create_with_id(dev, LMN_OP_CREATE_TEXTURE_VIEW, &iov, 1, &slot, &v->resource_id,
                       "texture view")) {
      v->base.handle = 0;
      mtl_release(v);
      return NULL;
   }
   return v;
}

static mtl_texture *
view_with(mtl_texture *texture, const struct kk_view_layout *layout, bool use_swizzle)
{
   struct lmn_texture *t = LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE);
   struct lmn_create_texture_view c = {
      .pixel_format = layout->format.mtl,
      .texture_type = view_type(layout->view_type, layout->sample_count_sa),
      .base_level = layout->base_level,
      .level_count = layout->num_levels,
      .base_slice = layout->base_array_layer,
      .slice_count = layout->array_len,
      .use_swizzle = use_swizzle,
   };
   if (use_swizzle) {
      c.swizzle[0] = swizzle(layout->swizzle.red);
      c.swizzle[1] = swizzle(layout->swizzle.green);
      c.swizzle[2] = swizzle(layout->swizzle.blue);
      c.swizzle[3] = swizzle(layout->swizzle.alpha);
   }
   return view_create(t, &c);
}

mtl_texture *
mtl_new_texture_view_with(mtl_texture *texture, const struct kk_view_layout *layout)
{
   return view_with(texture, layout, true);
}

mtl_texture *
mtl_new_texture_view_with_no_swizzle(mtl_texture *texture, const struct kk_view_layout *layout)
{
   return view_with(texture, layout, false);
}

mtl_texture *
mtl_new_texture_view_with_format(mtl_texture *texture, uint32_t pixel_format)
{
   struct lmn_texture *t = LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE);
   struct lmn_create_texture_view c = {
      .pixel_format = pixel_format,
      .texture_type = t->props.texture_type,
      .base_level = 0,
      .level_count = t->props.mip_levels,
      .base_slice = 0,
      .slice_count = t->props.array_length,
   };
   return view_create(t, &c);
}

/* Moves a texture region between host-process memory and the texture through a staging range
 * the host can reach. */
static void
texture_transfer(struct lmn_texture *t, void *host_ptr, struct mtl_texture_memory_copy *data,
                 bool to_texture)
{
   struct lmn_device *dev = t->base.dev;
   uint64_t len = data->image_size.z > 1
                     ? data->buffer_2d_image_size_B * data->image_size.z
                     : data->buffer_stride_B * data->image_size.y;
   if (len == 0)
      len = data->buffer_2d_image_size_B ? data->buffer_2d_image_size_B : data->buffer_stride_B;

   pthread_mutex_lock(&dev->lock);
   if (!dev->staging || dev->staging->size < len) {
      if (dev->staging)
         mtl_release(dev->staging);
      pthread_mutex_unlock(&dev->lock);
      struct lmn_heap *s = lmn_heap_create(dev, len < (4u << 20) ? (4u << 20) : len);
      pthread_mutex_lock(&dev->lock);
      dev->staging = s;
   }
   struct lmn_heap *s = dev->staging;
   pthread_mutex_unlock(&dev->lock);
   if (!s) {
      fprintf(stderr, "[limina-kk] no staging memory for a %llu-byte texture transfer\n",
              (unsigned long long)len);
      return;
   }

   struct lmn_slot slot;
   if (!lmn_slot_alloc(dev, &slot))
      return;
   if (to_texture)
      memcpy(s->map, host_ptr, len);
   struct lmn_texture_transfer c = {
      .texture = t->base.handle,
      .to_texture = to_texture,
      .level = (uint32_t)data->image_level,
      .slice = (uint32_t)data->image_slice,
      .origin = {(uint32_t)data->image_origin.x, (uint32_t)data->image_origin.y,
                 (uint32_t)data->image_origin.z},
      .size = {(uint32_t)data->image_size.x, (uint32_t)data->image_size.y,
               (uint32_t)data->image_size.z},
      .bytes_per_row = data->buffer_stride_B,
      .bytes_per_image = data->buffer_2d_image_size_B,
      .data = {.arena = s->arena, .offset = 0},
      .reply = slot.ref,
   };
   struct lmn_iov iov = {&c, sizeof(c)};
   struct lmn_reply r = lmn_call(dev, LMN_OP_TEXTURE_TRANSFER, &iov, 1, &slot);
   lmn_slot_free(dev, &slot);
   if (r.status != 0) {
      fprintf(stderr, "[limina-kk] texture transfer failed (error %u)\n", r.status);
      return;
   }
   if (!to_texture)
      memcpy(host_ptr, s->map, len);
}

void
mtl_texture_get_bytes(mtl_texture *texture, void *host_ptr, struct mtl_texture_memory_copy *data)
{
   texture_transfer(LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE), host_ptr, data, false);
}

void
mtl_texture_replace_region(mtl_texture *texture, const void *host_ptr,
                           struct mtl_texture_memory_copy *data)
{
   texture_transfer(LMN_CAST(lmn_texture, texture, LMN_KIND_TEXTURE), (void *)host_ptr, data,
                    true);
}

/* -- samplers -- */

struct lmn_sampler_desc_obj {
   struct lmn_obj base;
   struct lmn_sampler_desc desc;
};

mtl_sampler_descriptor *
mtl_new_sampler_descriptor(void)
{
   struct lmn_sampler_desc_obj *d =
      lmn_obj_new(NULL, LMN_KIND_SAMPLER_DESC, sizeof(*d), NULL);
   if (d) {
      d->desc.max_anisotropy = 1;
      d->desc.lod_max_clamp = 1000.0f;
      d->desc.normalized_coordinates = 1;
   }
   return d;
}

#define SD(d) (&LMN_CAST(lmn_sampler_desc_obj, (d), LMN_KIND_SAMPLER_DESC)->desc)

void
mtl_sampler_descriptor_set_normalized_coordinates(mtl_sampler_descriptor *descriptor,
                                                  bool normalized_coordinates)
{
   SD(descriptor)->normalized_coordinates = normalized_coordinates;
}

void
mtl_sampler_descriptor_set_address_mode(mtl_sampler_descriptor *descriptor,
                                        enum mtl_sampler_address_mode u,
                                        enum mtl_sampler_address_mode v,
                                        enum mtl_sampler_address_mode w)
{
   SD(descriptor)->address_s = u;
   SD(descriptor)->address_t = v;
   SD(descriptor)->address_r = w;
}

void
mtl_sampler_descriptor_set_border_color(mtl_sampler_descriptor *descriptor,
                                        enum mtl_sampler_border_color color)
{
   SD(descriptor)->border_color = color;
}

void
mtl_sampler_descriptor_set_filters(mtl_sampler_descriptor *descriptor,
                                   enum mtl_sampler_min_mag_filter min_filter,
                                   enum mtl_sampler_min_mag_filter mag_filter,
                                   enum mtl_sampler_mip_filter mip_filter)
{
   SD(descriptor)->min_filter = min_filter;
   SD(descriptor)->mag_filter = mag_filter;
   SD(descriptor)->mip_filter = mip_filter;
}

void
mtl_sampler_descriptor_set_lod_clamp(mtl_sampler_descriptor *descriptor, float min, float max)
{
   SD(descriptor)->lod_min_clamp = min;
   SD(descriptor)->lod_max_clamp = max;
}

void
mtl_sampler_descriptor_set_max_anisotropy(mtl_sampler_descriptor *descriptor, uint64_t max)
{
   SD(descriptor)->max_anisotropy = max ? (uint32_t)max : 1u;
}

void
mtl_sampler_descriptor_set_compare_function(mtl_sampler_descriptor *descriptor,
                                            enum mtl_compare_function function)
{
   SD(descriptor)->compare_function = function;
}

void
mtl_sampler_descriptor_set_reduction_mode(mtl_sampler_descriptor *descriptor,
                                          enum mtl_sampler_reduction_mode mode)
{
   SD(descriptor)->reduction_mode = mode;
}

struct lmn_sampler {
   struct lmn_obj base;
   uint64_t resource_id;
};

mtl_sampler *
mtl_new_sampler(mtl_device *device, mtl_sampler_descriptor *descriptor)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   struct lmn_sampler *s =
      lmn_obj_new(dev, LMN_KIND_SAMPLER, sizeof(*s), lmn_destroy_host_object);
   struct lmn_slot slot;
   if (!s)
      return NULL;
   if (!lmn_slot_alloc(dev, &slot)) {
      free(s);
      return NULL;
   }
   s->base.handle = lmn_new_handle(dev);
   struct lmn_create_sampler c = {
      .handle = s->base.handle,
      .desc = *SD(descriptor),
      .reply = slot.ref,
   };
   struct lmn_iov iov = {&c, sizeof(c)};
   if (!create_with_id(dev, LMN_OP_CREATE_SAMPLER, &iov, 1, &slot, &s->resource_id, "sampler")) {
      free(s);
      return NULL;
   }
   return s;
}

uint64_t
mtl_sampler_get_gpu_resource_id(mtl_sampler *sampler)
{
   return LMN_CAST(lmn_sampler, sampler, LMN_KIND_SAMPLER)->resource_id;
}

/* -- sync objects -- */

/* How many queue signals of one event are remembered; a forgotten one only makes a later
 * export wait for a later commit. */
#define LMN_EVENT_SIGNALS 16

struct lmn_shared_event {
   struct lmn_obj base;
   struct lmn_slot mirror;
   /* Queue signals not reached yet, in push order: value and the commit they follow. Under
    * `dev->lock`. */
   struct {
      uint64_t value, seqno;
   } signals[LMN_EVENT_SIGNALS];
   unsigned signal_count;
};

static mtl_fence *
sync_create(mtl_device *device, uint32_t kind, uint32_t sync_kind)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   struct lmn_obj *o = lmn_obj_new(dev, kind, sizeof(*o), lmn_destroy_host_object);
   if (!o)
      return NULL;
   o->handle = lmn_new_handle(dev);
   struct lmn_sync_object c = {.handle = o->handle, .kind = sync_kind};
   lmn_push(dev, LMN_OP_CREATE_SYNC, &c, sizeof(c));
   return o;
}

mtl_fence *
mtl_new_fence(mtl_device *device)
{
   return sync_create(device, LMN_KIND_FENCE, LMN_SYNC_FENCE);
}

mtl_event *
mtl_new_event(mtl_device *device)
{
   return sync_create(device, LMN_KIND_EVENT, LMN_SYNC_EVENT);
}

static void
shared_event_destroy(struct lmn_obj *obj)
{
   struct lmn_shared_event *e = (struct lmn_shared_event *)obj;
   lmn_destroy_host_object(obj);
   /* The host stops raising the mirror when it runs the DESTROY; the slot is reused once a
    * commit made after it has completed, by which time it has. */
   lmn_slot_free_after_next_commit(obj->dev, &e->mirror);
}

mtl_shared_event *
mtl_new_shared_event(mtl_device *device)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   struct lmn_shared_event *e =
      lmn_obj_new(dev, LMN_KIND_SHARED_EVENT, sizeof(*e), shared_event_destroy);
   if (!e)
      return NULL;
   if (!lmn_slot_alloc(dev, &e->mirror)) {
      free(e);
      return NULL;
   }
   e->base.handle = lmn_new_handle(dev);
   struct lmn_sync_object c = {
      .handle = e->base.handle,
      .kind = LMN_SYNC_SHARED_EVENT,
      .initial = 0,
      .mirror = e->mirror.ref,
   };
   lmn_push(dev, LMN_OP_CREATE_SYNC, &c, sizeof(c));
   return e;
}

static uint64_t *
mirror(struct lmn_shared_event *e)
{
   return (uint64_t *)e->mirror.ptr;
}

int
mtl_shared_event_wait_until_signaled_value(mtl_shared_event *event_handle, uint64_t value,
                                           uint64_t timeout_ms)
{
   struct lmn_shared_event *e = LMN_CAST(lmn_shared_event, event_handle, LMN_KIND_SHARED_EVENT);
   struct lmn_device *dev = e->base.dev;
   if (__atomic_load_n(mirror(e), __ATOMIC_ACQUIRE) >= value)
      return 1;
   /* Whatever will signal it must reach the host. */
   pthread_mutex_lock(&dev->lock);
   lmn_flush_locked(dev);
   pthread_mutex_unlock(&dev->lock);

   struct timespec start;
   clock_gettime(CLOCK_MONOTONIC, &start);
   for (unsigned round = 0;; round++) {
      if (__atomic_load_n(mirror(e), __ATOMIC_ACQUIRE) >= value)
         return 1;
      if (lmn_device_lost(dev, NULL, NULL))
         return 0;
      if ((round & 63) == 63) {
         struct timespec now;
         clock_gettime(CLOCK_MONOTONIC, &now);
         uint64_t ms = (uint64_t)(now.tv_sec - start.tv_sec) * 1000 +
                       (uint64_t)((now.tv_nsec - start.tv_nsec) / 1000000);
         if (ms >= timeout_ms)
            return 0;
      }
      lmn_backoff(round);
   }
}

void
lmn_shared_event_note_signal(void *event, uint64_t value, uint64_t seqno)
{
   struct lmn_shared_event *e = event;
   uint64_t reached = __atomic_load_n(mirror(e), __ATOMIC_ACQUIRE);
   unsigned n = 0;
   for (unsigned i = 0; i < e->signal_count; i++) {
      if (e->signals[i].value > reached)
         e->signals[n++] = e->signals[i];
   }
   if (n == LMN_EVENT_SIGNALS) {
      memmove(&e->signals[0], &e->signals[1], sizeof(e->signals[0]) * (n - 1));
      n--;
   }
   e->signals[n].value = value;
   e->signals[n].seqno = seqno;
   e->signal_count = n + 1;
}

int
mtl_shared_event_export_sync_file(mtl_shared_event *event_handle, uint64_t value)
{
   struct lmn_shared_event *e = LMN_CAST(lmn_shared_event, event_handle, LMN_KIND_SHARED_EVENT);
   struct lmn_device *dev = e->base.dev;
   bool known = false;
   uint64_t seqno = 0;
   pthread_mutex_lock(&dev->lock);
   if (__atomic_load_n(mirror(e), __ATOMIC_ACQUIRE) >= value) {
      known = true;
   } else {
      for (unsigned i = 0; i < e->signal_count && !known; i++) {
         if (e->signals[i].value >= value) {
            seqno = e->signals[i].seqno;
            known = true;
         }
      }
   }
   pthread_mutex_unlock(&dev->lock);
   if (!known)
      return -1;
   /* Seqno 0 is complete from the start: an already reached value gets a signaled fence. */
   struct drm_virtgpu_limina_fence f = {.seqno = seqno, .flags = VIRTGPU_LIMINA_FENCE_FD};
   if (ioctl(dev->fd, DRM_IOCTL_VIRTGPU_LIMINA_FENCE, &f) != 0)
      return -1;
   return f.fence_fd;
}

uint64_t
mtl_shared_event_get_signaled_value(mtl_shared_event *event_handle)
{
   struct lmn_shared_event *e = LMN_CAST(lmn_shared_event, event_handle, LMN_KIND_SHARED_EVENT);
   return __atomic_load_n(mirror(e), __ATOMIC_ACQUIRE);
}

void
mtl_shared_event_set_signaled_value(mtl_shared_event *event_handle, uint64_t value)
{
   struct lmn_shared_event *e = LMN_CAST(lmn_shared_event, event_handle, LMN_KIND_SHARED_EVENT);
   struct lmn_device *dev = e->base.dev;
   struct lmn_set_shared_event c = {.handle = e->base.handle, .value = value};
   pthread_mutex_lock(&dev->lock);
   if (lmn_push_locked(dev, LMN_OP_SET_SHARED_EVENT, &(struct lmn_iov){&c, sizeof(c)}, 1))
      lmn_flush_locked(dev);
   pthread_mutex_unlock(&dev->lock);
   /* Metal sets the value synchronously; so does the mirror, ahead of the host. */
   uint64_t cur = __atomic_load_n(mirror(e), __ATOMIC_RELAXED);
   while (cur < value &&
          !__atomic_compare_exchange_n(mirror(e), &cur, value, true, __ATOMIC_RELEASE,
                                       __ATOMIC_RELAXED))
      ;
}

/* -- residency: the host keeps every arena of a context resident -- */

mtl_residency_set *
mtl_new_residency_set(mtl_device *device)
{
   return lmn_obj_new(LMN_CAST(lmn_device, device, LMN_KIND_DEVICE), LMN_KIND_RESIDENCY_SET,
                      sizeof(struct lmn_obj), NULL);
}

void
mtl_residency_set_add_allocation(mtl_residency_set *residency_set, mtl_allocation *allocation)
{
}

void
mtl_residency_set_remove_allocation(mtl_residency_set *residency_set,
                                    mtl_allocation *allocation)
{
}

void
mtl_residency_set_commit(mtl_residency_set *residency_set)
{
}

void
mtl_residency_set_request_residency(mtl_residency_set *residency_set)
{
}

void
mtl_residency_set_end_residency(mtl_residency_set *residency_set)
{
}

/* -- argument tables -- */

struct lmn_argument_table_desc {
   struct lmn_obj base;
   uint32_t max;
};

mtl_argument_table_descriptor *
mtl_new_argument_table_descriptor(void)
{
   return lmn_obj_new(NULL, LMN_KIND_ARGUMENT_TABLE_DESC, sizeof(struct lmn_argument_table_desc),
                      NULL);
}

void
mtl_set_max_buffer_binding_count(mtl_argument_table_descriptor *descriptor, uint32_t count)
{
   LMN_CAST(lmn_argument_table_desc, descriptor, LMN_KIND_ARGUMENT_TABLE_DESC)->max = count;
}

mtl_argument_table *
mtl_new_argument_table(mtl_device *device, mtl_argument_table_descriptor *descriptor)
{
   struct lmn_device *dev = LMN_CAST(lmn_device, device, LMN_KIND_DEVICE);
   uint32_t max =
      LMN_CAST(lmn_argument_table_desc, descriptor, LMN_KIND_ARGUMENT_TABLE_DESC)->max;
   if (max == 0 || max > 31)
      return NULL;
   struct lmn_argument_table *t =
      lmn_obj_new(dev, LMN_KIND_ARGUMENT_TABLE, sizeof(*t), lmn_destroy_host_object);
   if (!t)
      return NULL;
   t->base.handle = lmn_new_handle(dev);
   t->max = max;
   struct lmn_create_argument_table c = {.handle = t->base.handle, .max_buffer_bind_count = max};
   lmn_push(dev, LMN_OP_CREATE_ARGUMENT_TABLE, &c, sizeof(c));
   return t;
}

void
mtl_set_address(mtl_argument_table *table, uint64_t address, uint32_t binding)
{
   struct lmn_argument_table *t = LMN_CAST(lmn_argument_table, table, LMN_KIND_ARGUMENT_TABLE);
   if (binding >= t->max) {
      fprintf(stderr, "[limina-kk] argument table binding %u past its %u\n", binding, t->max);
      abort();
   }
   /* Recorded here and written into a command buffer before the next draw or dispatch that
    * uses the table, which is when Metal reads it. */
   t->addr[binding] = address;
   t->set_mask |= 1u << binding;
   t->dirty |= 1u << binding;
}
