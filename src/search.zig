const std = @import("std");
const Allocator = std.mem.Allocator;

pub const SearchResult = struct {
    doc_idx: usize,
    score: f32,
};

const SIMD_WIDTH = 8;
const VecF32 = @Vector(SIMD_WIDTH, f32);

pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    const dim = @min(a.len, b.len);
    if (dim == 0) return 0;

    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    var i: usize = 0;
    if (dim >= SIMD_WIDTH) {
        var v_dot: VecF32 = @splat(0);
        var v_na: VecF32 = @splat(0);
        var v_nb: VecF32 = @splat(0);

        while (i + SIMD_WIDTH <= dim) : (i += SIMD_WIDTH) {
            const va: VecF32 = a[i..][0..SIMD_WIDTH].*;
            const vb: VecF32 = b[i..][0..SIMD_WIDTH].*;
            v_dot += va * vb;
            v_na += va * va;
            v_nb += vb * vb;
        }

        dot = @reduce(.Add, v_dot);
        norm_a = @reduce(.Add, v_na);
        norm_b = @reduce(.Add, v_nb);
    }

    while (i < dim) : (i += 1) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0 or norm_b == 0) return 0;
    return dot / (@sqrt(norm_a) * @sqrt(norm_b));
}

pub fn quantizedNorm(vec: []const i8) f32 {
    if (vec.len == 0) return 0;

    var sum: i64 = 0;
    var i: usize = 0;
    while (i + 16 <= vec.len) : (i += 16) {
        inline for (0..16) |j| {
            const v: i32 = vec[i + j];
            sum += @as(i64, v * v);
        }
    }
    while (i < vec.len) : (i += 1) {
        const v: i32 = vec[i];
        sum += @as(i64, v * v);
    }
    return @sqrt(@as(f32, @floatFromInt(sum)));
}

pub fn dotInt8(a: []const i8, b: []const i8) i32 {
    const dim = @min(a.len, b.len);
    var sum: i32 = 0;
    var i: usize = 0;

    while (i + 32 <= dim) : (i += 32) {
        inline for (0..32) |j| {
            sum += @as(i32, a[i + j]) * @as(i32, b[i + j]);
        }
    }
    while (i < dim) : (i += 1) {
        sum += @as(i32, a[i]) * @as(i32, b[i]);
    }
    return sum;
}

pub fn quantizedCosineSimilarity(a: []const i8, a_norm: f32, b: []const i8, b_norm: f32) f32 {
    if (a.len == 0 or b.len == 0 or a_norm == 0 or b_norm == 0) return 0;
    const dot = @as(f32, @floatFromInt(dotInt8(a, b)));
    return dot / (a_norm * b_norm);
}

pub const FlatIndex = struct {
    embeddings: []f32,
    norms: []f32,
    n_docs: usize,
    dim: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, dim: usize) FlatIndex {
        return .{
            .embeddings = &.{},
            .norms = &.{},
            .n_docs = 0,
            .dim = dim,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FlatIndex) void {
        if (self.embeddings.len > 0) self.allocator.free(self.embeddings);
        if (self.norms.len > 0) self.allocator.free(self.norms);
    }

    pub fn build(self: *FlatIndex, embeddings: []f32, n_docs: usize) !void {
        if (self.embeddings.len > 0) self.allocator.free(self.embeddings);
        if (self.norms.len > 0) self.allocator.free(self.norms);

        self.embeddings = embeddings;
        self.n_docs = n_docs;
        self.norms = try self.allocator.alloc(f32, n_docs);

        for (0..n_docs) |d| {
            const row = self.embeddings[d * self.dim .. (d + 1) * self.dim];
            var s: f32 = 0;
            var j: usize = 0;
            while (j + SIMD_WIDTH <= self.dim) : (j += SIMD_WIDTH) {
                const v: VecF32 = row[j..][0..SIMD_WIDTH].*;
                s += @reduce(.Add, v * v);
            }
            while (j < self.dim) : (j += 1) {
                s += row[j] * row[j];
            }
            self.norms[d] = @sqrt(s);
        }
    }

    pub fn search(self: *const FlatIndex, query: []const f32, top_k: usize, threshold: f32, results_buf: []SearchResult) usize {
        if (self.n_docs == 0 or query.len == 0) return 0;

        const dim = @min(query.len, self.dim);
        var q_norm: f32 = 0;
        {
            var j: usize = 0;
            while (j + SIMD_WIDTH <= dim) : (j += SIMD_WIDTH) {
                const v: VecF32 = query[j..][0..SIMD_WIDTH].*;
                q_norm += @reduce(.Add, v * v);
            }
            while (j < dim) : (j += 1) {
                q_norm += query[j] * query[j];
            }
        }

        q_norm = @sqrt(q_norm);
        if (q_norm == 0) return 0;

        const k = @min(top_k, results_buf.len);
        if (k == 0) return 0;
        var heap_size: usize = 0;

        for (0..self.n_docs) |d| {
            const d_norm = self.norms[d];
            if (d_norm == 0) continue;

            const row = self.embeddings[d * self.dim .. d * self.dim + dim];
            var dot: f32 = 0;
            var j: usize = 0;
            while (j + SIMD_WIDTH <= dim) : (j += SIMD_WIDTH) {
                const va: VecF32 = query[j..][0..SIMD_WIDTH].*;
                const vb: VecF32 = row[j..][0..SIMD_WIDTH].*;
                dot += @reduce(.Add, va * vb);
            }
            while (j < dim) : (j += 1) {
                dot += query[j] * row[j];
            }

            const score = dot / (q_norm * d_norm);
            if (score < threshold) continue;
            heapPush(results_buf[0..k], &heap_size, .{ .doc_idx = d, .score = score });
        }

        return finalizeResults(results_buf[0..heap_size]);
    }
};

pub const QuantizedFlatIndex = struct {
    embeddings: []const i8,
    norms: []const f32,
    n_docs: usize,
    dim: usize,

    pub fn init(embeddings: []const i8, norms: []const f32, dim: usize) QuantizedFlatIndex {
        return .{
            .embeddings = embeddings,
            .norms = norms,
            .n_docs = if (dim == 0) 0 else norms.len,
            .dim = dim,
        };
    }

    pub fn search(self: *const QuantizedFlatIndex, query: []const i8, query_norm: f32, top_k: usize, threshold: f32, results_buf: []SearchResult) usize {
        if (self.n_docs == 0 or self.dim == 0 or query.len < self.dim or query_norm == 0) return 0;

        const k = @min(top_k, results_buf.len);
        if (k == 0) return 0;
        var heap_size: usize = 0;

        for (0..self.n_docs) |doc_idx| {
            const doc_norm = self.norms[doc_idx];
            if (doc_norm == 0) continue;

            const row = self.embeddings[doc_idx * self.dim .. (doc_idx + 1) * self.dim];
            const score = quantizedCosineSimilarity(query[0..self.dim], query_norm, row, doc_norm);
            if (score < threshold) continue;
            heapPush(results_buf[0..k], &heap_size, .{ .doc_idx = doc_idx, .score = score });
        }

        return finalizeResults(results_buf[0..heap_size]);
    }
};

fn heapPush(heap: []SearchResult, heap_size: *usize, item: SearchResult) void {
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

fn finalizeResults(results: []SearchResult) usize {
    std.sort.insertion(SearchResult, results, {}, struct {
        fn lessThan(_: void, lhs: SearchResult, rhs: SearchResult) bool {
            return lhs.score > rhs.score;
        }
    }.lessThan);
    return results.len;
}

fn heapify(buf: []SearchResult) void {
    if (buf.len <= 1) return;
    var i = buf.len / 2;
    while (i > 0) {
        i -= 1;
        siftDown(buf, i);
    }
}

fn siftDown(buf: []SearchResult, start: usize) void {
    var i = start;
    while (true) {
        var smallest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        if (left < buf.len and buf[left].score < buf[smallest].score) smallest = left;
        if (right < buf.len and buf[right].score < buf[smallest].score) smallest = right;
        if (smallest == i) break;
        std.mem.swap(SearchResult, &buf[i], &buf[smallest]);
        i = smallest;
    }
}

test "cosine similarity identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const sim = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.001);
}

test "quantized cosine similarity identical vectors" {
    const a = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]i8{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const sim = quantizedCosineSimilarity(&a, quantizedNorm(&a), &b, quantizedNorm(&b));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.001);
}

test "quantized flat index search" {
    const dim = 4;
    const embeddings = [_]i8{
        10, 0, 0, 0,
        0, 10, 0, 0,
        9, 9, 0, 0,
    };
    const norms = [_]f32{
        quantizedNorm(embeddings[0..4]),
        quantizedNorm(embeddings[4..8]),
        quantizedNorm(embeddings[8..12]),
    };
    const query = [_]i8{ 10, 0, 0, 0 };
    const view = QuantizedFlatIndex.init(&embeddings, &norms, dim);

    var results: [3]SearchResult = undefined;
    const n = view.search(&query, quantizedNorm(&query), 3, 0.0, &results);

    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqual(@as(usize, 0), results[0].doc_idx);
    try std.testing.expect(results[1].doc_idx == 2 or results[2].doc_idx == 2);
}
