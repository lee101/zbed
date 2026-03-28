const std = @import("std");
const build_options = @import("build_options");

pub const cudaEnabled = Impl.cudaEnabled;
pub const deviceCount = Impl.deviceCount;
pub const isAvailable = Impl.isAvailable;
pub const lastErrorMessage = Impl.lastErrorMessage;
pub const deviceName = Impl.deviceName;
pub const GpuEmbedder = Impl.GpuEmbedder;
pub const GpuSearchIndex = Impl.GpuSearchIndex;

const Impl = if (build_options.have_cuda) Real else Stub;

const Stub = struct {
    const Allocator = std.mem.Allocator;

    pub fn cudaEnabled() bool {
        return false;
    }

    pub fn deviceCount() usize {
        return 0;
    }

    pub fn isAvailable() bool {
        return false;
    }

    pub fn lastErrorMessage() []const u8 {
        return "CUDA support not built in";
    }

    pub fn deviceName(_: Allocator, _: usize) ![]u8 {
        return error.CudaUnavailable;
    }

    pub const GpuEmbedder = struct {
        const Self = @This();

        pub fn init(_: anytype) !Self {
            return error.CudaUnavailable;
        }

        pub fn deinit(_: *Self) void {}

        pub fn embedQuantized(_: *Self, _: anytype, _: []const u8, _: anytype, _: anytype) !usize {
            return error.CudaUnavailable;
        }
    };

    pub const GpuSearchIndex = struct {
        const Self = @This();

        pub fn init(_: Allocator, _: anytype) !Self {
            return error.CudaUnavailable;
        }

        pub fn deinit(_: *Self) void {}

        pub fn search(_: *Self, _: anytype, _: usize, _: f32, _: anytype) !usize {
            return error.CudaUnavailable;
        }
    };
};

const Real = struct {
    const Allocator = std.mem.Allocator;
    const embed_mod = @import("embed.zig");
    const index_mod = @import("index.zig");
    const search_mod = @import("search.zig");

    const c = @cImport({
        @cInclude("zbed_cuda.h");
    });

    pub fn cudaEnabled() bool {
        return true;
    }

    pub fn deviceCount() usize {
        const count = c.zbed_cuda_device_count();
        return if (count <= 0) 0 else @intCast(count);
    }

    pub fn isAvailable() bool {
        return @This().deviceCount() > 0;
    }

    pub fn lastErrorMessage() []const u8 {
        return std.mem.span(c.zbed_cuda_last_error_message());
    }

    pub fn deviceName(allocator: Allocator, device: usize) ![]u8 {
        var buf: [256]u8 = [_]u8{0} ** 256;
        if (c.zbed_cuda_get_device_name(@intCast(device), &buf, buf.len) == 0) {
            return error.CudaDeviceQueryFailed;
        }
        return allocator.dupe(u8, std.mem.sliceTo(&buf, 0));
    }

    pub const GpuEmbedder = struct {
        const Self = @This();

        handle: ?*anyopaque = null,

        pub fn init(model: *const embed_mod.EmbedModel) !Self {
            if (!Real.isAvailable()) return error.CudaUnavailable;
            if (model.embed_dim != 512) return error.UnsupportedDimension;

            const handle = c.zbed_cuda_embedder_create(
                model.weights.ptr,
                model.scales.ptr,
                @intCast(model.vocab_size),
                @intCast(model.embed_dim),
            );
            if (handle == null) return error.CudaInitFailed;
            return .{ .handle = handle };
        }

        pub fn deinit(self: *Self) void {
            if (self.handle) |handle| {
                c.zbed_cuda_embedder_destroy(handle);
                self.handle = null;
            }
        }

        pub fn embedQuantized(self: *Self, model: *const embed_mod.EmbedModel, text: []const u8, scratch: *embed_mod.EmbedScratch, out: *embed_mod.QuantizedEmbedding) !usize {
            const n_tokens = model.tokenizer.tokenize(text, scratch.token_buf[0..]);
            if (n_tokens == 0) {
                @memset(scratch.float_buf[0..model.embed_dim], 0);
                out.scale = embed_mod.quantize(scratch.float_buf[0..model.embed_dim], out.data[0..model.embed_dim]);
                out.norm = search_mod.quantizedNorm(out.data[0..model.embed_dim]);
                out.valid_tokens = 0;
                return 0;
            }

            const ok = c.zbed_cuda_embedder_embed(
                self.handle,
                scratch.token_buf[0..n_tokens].ptr,
                @intCast(n_tokens),
                scratch.float_buf[0..model.embed_dim].ptr,
            );
            if (ok == 0) return error.CudaEmbedFailed;

            out.scale = embed_mod.quantize(scratch.float_buf[0..model.embed_dim], out.data[0..model.embed_dim]);
            out.norm = search_mod.quantizedNorm(out.data[0..model.embed_dim]);
            out.valid_tokens = n_tokens;
            return n_tokens;
        }
    };

    pub const GpuSearchIndex = struct {
        const Self = @This();

        handle: ?*anyopaque = null,
        host_scores: []f32 = &.{},
        allocator: Allocator,
        n_docs: usize,

        pub fn init(allocator: Allocator, index: *const index_mod.Index) !Self {
            if (!Real.isAvailable()) return error.CudaUnavailable;
            if (index.dim != 512) return error.UnsupportedDimension;

            const handle = c.zbed_cuda_search_create(
                index.embeddings.items.ptr,
                index.norms.items.ptr,
                @intCast(index.count()),
                @intCast(index.dim),
            );
            if (handle == null) return error.CudaInitFailed;

            errdefer c.zbed_cuda_search_destroy(handle);

            return .{
                .handle = handle,
                .host_scores = try allocator.alloc(f32, index.count()),
                .allocator = allocator,
                .n_docs = index.count(),
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.handle) |handle| {
                c.zbed_cuda_search_destroy(handle);
                self.handle = null;
            }
            if (self.host_scores.len > 0) {
                self.allocator.free(self.host_scores);
                self.host_scores = &.{};
            }
        }

        pub fn search(self: *Self, query: *const embed_mod.QuantizedEmbedding, top_k: usize, threshold: f32, results_buf: []search_mod.SearchResult) !usize {
            if (self.n_docs == 0) return 0;
            if (query.valid_tokens == 0 or query.norm == 0) return 0;

            const ok = c.zbed_cuda_search_scores(
                self.handle,
                query.data[0..512].ptr,
                query.norm,
                self.host_scores.ptr,
            );
            if (ok == 0) return error.CudaSearchFailed;

            const k = @min(top_k, results_buf.len);
            if (k == 0) return 0;

            var heap_size: usize = 0;
            for (self.host_scores, 0..) |score, doc_idx| {
                if (score < threshold) continue;
                heapPush(results_buf[0..k], &heap_size, .{
                    .doc_idx = doc_idx,
                    .score = score,
                });
            }

            std.sort.insertion(search_mod.SearchResult, results_buf[0..heap_size], {}, struct {
                fn lessThan(_: void, lhs: search_mod.SearchResult, rhs: search_mod.SearchResult) bool {
                    return lhs.score > rhs.score;
                }
            }.lessThan);
            return heap_size;
        }
    };

    fn heapPush(heap: []search_mod.SearchResult, heap_size: *usize, item: search_mod.SearchResult) void {
        if (heap.len == 0) return;

        if (heap_size.* < heap.len) {
            heap[heap_size.*] = item;
            heap_size.* += 1;
            if (heap_size.* == heap.len) heapify(heap);
            return;
        }

        if (item.score > heap[0].score) {
            heap[0] = item;
            siftDown(heap, 0);
        }
    }

    fn heapify(buf: []search_mod.SearchResult) void {
        var i = buf.len / 2;
        while (i > 0) {
            i -= 1;
            siftDown(buf, i);
        }
    }

    fn siftDown(buf: []search_mod.SearchResult, start: usize) void {
        var i = start;
        while (true) {
            var smallest = i;
            const left = 2 * i + 1;
            const right = 2 * i + 2;
            if (left < buf.len and buf[left].score < buf[smallest].score) smallest = left;
            if (right < buf.len and buf[right].score < buf[smallest].score) smallest = right;
            if (smallest == i) break;
            std.mem.swap(search_mod.SearchResult, &buf[i], &buf[smallest]);
            i = smallest;
        }
    }
};
