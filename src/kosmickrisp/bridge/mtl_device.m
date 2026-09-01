/*
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "mtl_device.h"

/* limina: per-class allocation census (limina_mtl_note_new). */
#include "mtl_bridge.h"

/* TODO_KOSMICKRISP Remove */
#include "kk_image_layout.h"
#include "kk_private.h"

#include <Metal/MTL4Counters.h>
#include <Metal/MTLCaptureManager.h>
#include <Metal/MTLDevice.h>
#include <IOSurface/IOSurfaceRef.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <stdatomic.h>

/* Device creation */
mtl_device *
mtl_device_create()
{
   mtl_device *device = nil;

   @autoreleasepool {
      NSArray<id<MTLDevice>> *devs = [MTLCopyAllDevices() autorelease];
      uint32_t device_count = [devs count];
      
      for (uint32_t i = 0u; i < device_count; ++i) {
         if ([devs[i] supportsFamily:MTLGPUFamilyMetal4]) {
            device = (mtl_device *)[devs[i] retain];
            break;
         }
      }
   }

   return device;
}

/* Device operations */
void
mtl_start_gpu_capture(mtl_device *mtl_dev_handle, const char *directory)
{
   @autoreleasepool {
      id<MTLDevice> mtl_dev = (id<MTLDevice>)mtl_dev_handle;
      MTLCaptureManager *captureMgr = [MTLCaptureManager sharedCaptureManager];

      MTLCaptureDescriptor *captureDesc = [[MTLCaptureDescriptor new] autorelease];
      captureDesc.captureObject = mtl_dev;
      captureDesc.destination = MTLCaptureDestinationDeveloperTools;

      /* limina: without MTL_CAPTURE_ENABLED=1 (or the Info.plist key) Metal refuses programmatic
       * capture, and a directory we cannot write to falls back to DeveloperTools, which needs
       * Xcode already attached. Both failures otherwise look identical to "no trace appeared". */
      if (directory && ![captureMgr supportsDestination: MTLCaptureDestinationGPUTraceDocument])
         fprintf(stderr, "[LIMINA-KK-CAPTURE] GPUTraceDocument unsupported -- falling back to "
                         "DeveloperTools, which writes no file. Is MTL_CAPTURE_ENABLED=1 set?\n");

      if (directory && [captureMgr supportsDestination: MTLCaptureDestinationGPUTraceDocument]) {
         NSString *dir = [NSString stringWithUTF8String:directory];
         NSString *pname = [[NSProcessInfo processInfo] processName];
         NSString *capture_path = [NSString stringWithFormat:@"%@/%@.gputrace", dir, pname];
         captureDesc.destination = MTLCaptureDestinationGPUTraceDocument;
         captureDesc.outputURL = [NSURL fileURLWithPath: capture_path];
      }

      NSError *err = nil;
      if (![captureMgr startCaptureWithDescriptor:captureDesc error:&err]) {
         fprintf(stderr, "Failed to automatically start GPU capture session (Error code %li) using startCaptureWithDescriptor: %s\n",
                 (long)err.code, err.localizedDescription.UTF8String);
      }
   }
}

void
mtl_stop_gpu_capture()
{
   @autoreleasepool {
      [[MTLCaptureManager sharedCaptureManager] stopCapture];
   }
}

/* Device feature query */
void
mtl_device_get_name(mtl_device *dev, char buffer[256])
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      [device.name getCString:buffer maxLength:(sizeof(char) * 256) encoding:NSUTF8StringEncoding];
   }
}

void
mtl_device_get_architecture_name(mtl_device *dev, char buffer[256])
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      [device.architecture.name getCString:buffer maxLength:(sizeof(char) * 256) encoding:NSUTF8StringEncoding];
   }
}

uint32_t
mtl_device_get_gpu_apple_family(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      uint32_t gpu_family = 0u;
      MTLGPUFamily family = MTLGPUFamilyApple1;
      while([device supportsFamily:family]) {
         family += 1u;
         gpu_family += 1u;
      }
      return gpu_family;
   }
}

uint64_t
mtl_device_get_registry_id(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.registryID;
   }
}

bool
mtl_device_supports_sample_count(mtl_device *dev, uint32_t sample_count)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return [device supportsTextureSampleCount:sample_count];
   }
}

struct mtl_size
mtl_device_max_threads_per_threadgroup(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return (struct mtl_size){.x = device.maxThreadsPerThreadgroup.width,
                               .y = device.maxThreadsPerThreadgroup.height,
                               .z = device.maxThreadsPerThreadgroup.depth};
   }
}

uint32_t
mtl_device_max_threadgroup_memory_length(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.maxThreadgroupMemoryLength;
   }
}

uint64_t
mtl_device_max_buffer_length(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.maxBufferLength;
   }
}

uint64_t
mtl_device_recommended_max_working_set_size(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.recommendedMaxWorkingSetSize;
   }
}

uint64_t
mtl_device_current_allocated_size(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.currentAllocatedSize;
   }
}

uint32_t
mtl_device_max_argument_buffer_sampler_count(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return device.maxArgumentBufferSamplerCount;
   }
}

/* Timestamp query */
uint64_t
mtl_device_get_gpu_timestamp(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      MTLTimestamp cpu_ts, gpu_ts;

      [device sampleTimestamps:&cpu_ts gpuTimestamp:&gpu_ts];

      return (uint64_t)gpu_ts;
   }
}

uint64_t
mtl_device_timestamp_frequency(mtl_device *dev)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      return [device queryTimestampFrequency];
   }
}

mtl_counter_heap *
mtl_new_timestamp_counter_heap(mtl_device *dev, uint32_t count)
{
   @autoreleasepool {
      id<MTLDevice> device = (id<MTLDevice>)dev;
      MTL4CounterHeapDescriptor *desc =
         [[MTL4CounterHeapDescriptor alloc] init];
      desc.type = MTL4CounterHeapTypeTimestamp;
      desc.count = count;

      NSError *error = nil;
      id<MTL4CounterHeap> heap = [device newCounterHeapWithDescriptor:desc
                                                                error:&error];
      [desc release];

      if (heap == nil) {
         fprintf(stderr, "Failed to create timestamp counter heap: %s\n",
                 error ? error.localizedDescription.UTF8String : "unknown");
         return (mtl_counter_heap *)limina_mtl_note_new(NULL);
      }

      return (mtl_counter_heap *)limina_mtl_note_new((mtl_counter_heap *)heap);
   }
}

/* Resource queries */
/* TODO_KOSMICKRISP Return a struct */
void
mtl_heap_buffer_size_and_align_with_length(mtl_device *device, uint64_t *size_B,
                                           uint64_t *align_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLSizeAndAlign size_align = [dev heapBufferSizeAndAlignWithLength:*size_B options:KK_MTL_RESOURCE_OPTIONS];
      *size_B = size_align.size;
      *align_B = size_align.align;
   }
}

/* TODO_KOSMICKRISP Remove */
static MTLTextureDescriptor *
mtl_new_texture_descriptor(const struct kk_image_layout *layout)
{
   @autoreleasepool {
      MTLTextureDescriptor *descriptor = [MTLTextureDescriptor new];
      descriptor.textureType = (MTLTextureType)layout->type;
      descriptor.pixelFormat = layout->format.mtl;
      descriptor.width = layout->width_px;
      descriptor.height = layout->height_px;
      descriptor.depth = layout->depth_px;
      descriptor.mipmapLevelCount = layout->levels;
      descriptor.sampleCount = layout->sample_count_sa;
      descriptor.arrayLength = layout->layers;
      descriptor.allowGPUOptimizedContents = layout->optimized_layout;
      descriptor.usage = (MTLTextureUsage)layout->usage;
      /* We don't set the swizzle because Metal complains when the usage has store or render target with swizzle... */
      
      return (MTLTextureDescriptor *)limina_mtl_note_new(descriptor);
   }
}

/* limina: DEALLOC census for the IOSurface-backed import textures.
 *
 * The 2026-08-07 storm left every scanout IOSurface resident (8.6 G) although both vkr and
 * kk_destroy_bo released their refs. Release CALLS balancing proves nothing about the object
 * dying, so count the death itself: an associated object is released when its host is
 * deallocated, so a sentinel attached here turns "was this texture freed" into a number. If
 * created and deallocated diverge, the holder is inside Metal/KK — and the IOSurface cannot
 * die while a texture over it lives. */
#define LIMINA_KK_CENSUS_BUILD_TAG "kk-sentinel-1"

static _Atomic uint64_t g_limina_kk_tex_create;
static _Atomic uint64_t g_limina_kk_tex_dealloc;
static _Atomic uint64_t g_limina_kk_selftest_dealloc;

@interface LiminaKKSentinel : NSObject
@property(nonatomic) bool selftest;
@end

@implementation LiminaKKSentinel
- (void)dealloc
{
   atomic_fetch_add(_selftest ? &g_limina_kk_selftest_dealloc
                              : &g_limina_kk_tex_dealloc,
                    1);
   [super dealloc];
}
@end

static const char g_limina_kk_sentinel_key; /* address only */

static void
limina_kk_sentinel_attach(id obj, bool selftest)
{
   if (!obj)
      return;
   LiminaKKSentinel *s = [[LiminaKKSentinel alloc] init];
   s.selftest = selftest;
   objc_setAssociatedObject(obj, &g_limina_kk_sentinel_key, s,
                            OBJC_ASSOCIATION_RETAIN_NONATOMIC);
   [s release]; /* the association holds the only ref; it dies with obj */
}

/* An instrument that never fires reads exactly like "nothing leaked". Prove the mechanism on
 * an object whose death we control, and print a build tag so a stale dylib shows in the log. */
static void
limina_kk_sentinel_selftest(void)
{
   uint64_t before = atomic_load(&g_limina_kk_selftest_dealloc);
   @autoreleasepool {
      NSObject *victim = [[NSObject alloc] init];
      limina_kk_sentinel_attach(victim, true);
      [victim release];
   }
   fprintf(stderr,
           "[LIMINA-SENTINEL] kk " LIMINA_KK_CENSUS_BUILD_TAG
           " armed; dealloc sentinel self-test: %s\n",
           atomic_load(&g_limina_kk_selftest_dealloc) > before
              ? "OK"
              : "FAILED (dealloc counts below are meaningless)");
}

static void
limina_kk_sentinel_once(void)
{
   static pthread_once_t once = PTHREAD_ONCE_INIT;
   pthread_once(&once, limina_kk_sentinel_selftest);
}

void
mtl_limina_texture_census(char *buf, unsigned long len)
{
   uint64_t c = atomic_load(&g_limina_kk_tex_create),
            d = atomic_load(&g_limina_kk_tex_dealloc);
   snprintf(buf, len, "kk import-tex created %llu deallocated %llu (alive %lld)",
            (unsigned long long)c, (unsigned long long)d, (long long)(c - d));
}

/* A retain count is a LEAD, not a verdict — the runtime takes transient refs — but "2 where we
 * hold 1" names a second holder at a point in time the sentinels cannot report on. */
long
mtl_limina_retain_count(void *obj)
{
   return obj ? CFGetRetainCount((CFTypeRef)obj) : -1;
}

/* limina: is this external handle an IOSurfaceRef (vs an id<MTLTexture>)?
 * Lives in the bridge so plain-C callers need no CoreFoundation/IOSurface
 * linkage of their own. */
bool
mtl_handle_is_iosurface(void *handle)
{
   return handle && CFGetTypeID((CFTypeRef)handle) == IOSurfaceGetTypeID();
}

/* limina: create a texture whose storage IS an IOSurface
 * (newTextureWithDescriptor:iosurface:plane:). Built from the adopting
 * image's own kk_image_layout so kk_image_plane_bind's verbatim-adoption
 * checks pass by construction. IOSurface-backed textures must be plain 2D.
 *
 * `plane` selects the plane of a planar surface (NV12 decode targets bind
 * luma as R8 and chroma as RG8 from the one surface); 0 for the single-plane
 * scanout and shared-buffer imports. Metal validates the descriptor against
 * that plane's own geometry, so a mismatched layout fails here rather than
 * silently sampling the wrong bytes. */
mtl_texture *
mtl_new_texture_with_descriptor_iosurface(mtl_device *device,
                                          const struct kk_image_layout *layout,
                                          void *iosurface, uint32_t plane)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLTextureDescriptor *descriptor =
         [mtl_new_texture_descriptor(layout) autorelease];
      descriptor.resourceOptions = MTLResourceStorageModeShared;
      if (descriptor.textureType != MTLTextureType2D ||
          descriptor.mipmapLevelCount != 1 || descriptor.arrayLength != 1 ||
          descriptor.sampleCount != 1) {
         fprintf(stderr,
                 "[LIMINA-KK-IOSURF] refusing IOSurface texture: layout not "
                 "plain 2D (type=%u levels=%u layers=%u samples=%u)\n",
                 layout->type, layout->levels, layout->layers,
                 layout->sample_count_sa);
         return (mtl_texture *)limina_mtl_note_new(NULL);
      }
      const size_t plane_count = IOSurfaceGetPlaneCount((IOSurfaceRef)iosurface);
      if (plane > 0 && plane >= plane_count) {
         fprintf(stderr,
                 "[LIMINA-KK-IOSURF] refusing IOSurface texture: plane %u of a "
                 "%zu-plane surface\n",
                 plane, plane_count);
         return (mtl_texture *)limina_mtl_note_new(NULL);
      }
      id<MTLTexture> tex = [dev newTextureWithDescriptor:descriptor
                                              iosurface:(IOSurfaceRef)iosurface
                                                  plane:plane];
      if (tex) {
         limina_kk_sentinel_once();
         limina_kk_sentinel_attach(tex, false);
         atomic_fetch_add(&g_limina_kk_tex_create, 1);
      }
      return (mtl_texture *)limina_mtl_note_new(tex);
   }
}

uint64_t
mtl_minimum_linear_texture_alignment_for_pixel_format(
   mtl_device *device, enum mtl_pixel_format format)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format];
   }
}

void
mtl_heap_texture_size_and_align_with_descriptor(mtl_device *device,
                                                struct kk_image_layout *layout,
                                                uint64_t *size_B,
                                                uint64_t *align_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLTextureDescriptor *descriptor = [mtl_new_texture_descriptor(layout) autorelease];
      descriptor.resourceOptions = KK_MTL_RESOURCE_OPTIONS;

      MTLSizeAndAlign size_align = [dev heapTextureSizeAndAlignWithDescriptor:descriptor];
      if (size_B)
         *size_B = size_align.size;
      if (align_B)
         *align_B = size_align.align;
   }
}

uint32_t
mtl_sparse_tile_size_in_bytes(mtl_device *device) {
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return [dev sparseTileSizeInBytes];
   }
}

struct mtl_size
mtl_sparse_tile_size(mtl_device *device, struct kk_image_layout *layout) {
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLSize tile_size = [dev sparseTileSizeWithTextureType:(MTLTextureType)layout->type
                                                 pixelFormat:layout->format.mtl
                                                 sampleCount:layout->sample_count_sa];
      return (struct mtl_size){tile_size.width, tile_size.height, tile_size.depth};
   }
}

struct mtl_size
mtl_sparse_tile_count(mtl_device *device, struct kk_image_layout *layout,
                      struct mtl_size tile_size) {
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      MTLRegion pixel_region = MTLRegionMake3D(
          0, 0, 0, layout->width_px, layout->height_px, layout->depth_px);
      MTLRegion tile_region;
      [dev convertSparsePixelRegions:&pixel_region
                       toTileRegions:&tile_region
                        withTileSize:MTLSizeMake(tile_size.x, tile_size.y, tile_size.z)
                       alignmentMode:MTLSparseTextureRegionAlignmentModeOutward
                          numRegions:1];

      return (struct mtl_size){tile_region.size.width, tile_region.size.height,
                               tile_region.size.depth};
   }
}

/* Resource creation */
mtl_buffer *
mtl_new_buffer_with_bytes_no_copy(mtl_device *device, void* ptr,
                                  uint64_t size_B)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return (mtl_buffer *)limina_mtl_note_new([dev newBufferWithBytesNoCopy:ptr length:size_B options:KK_MTL_RESOURCE_OPTIONS deallocator:nil]);
   }
}

mtl_command_allocator *
mtl_new_command_allocator(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return (mtl_command_allocator *)limina_mtl_note_new([dev newCommandAllocator]);
   }
}

mtl_command_buffer *
mtl_new_command_buffer(mtl_device *device)
{
   @autoreleasepool {
      id<MTLDevice> dev = (id<MTLDevice>)device;
      return (mtl_command_buffer *)limina_mtl_note_new([dev newCommandBuffer]);
   }
}
