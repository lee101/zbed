const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

/// SIMD width for f32 operations.
const SIMD_WIDTH = 8;
const VecF32 = @Vector(SIMD_WIDTH, f32);

/// A single search result.
pub const SearchResult = struct {
    /// Index into the document / embedding list.
    doc_idx: usize,
    /// Cosine similarity score in [-1, 1].
    score: f32,
};

/// Compute cosine similarity between two f32 vectors.
/// Uses SIMD @Vector for the inner loop.
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    const dim = @min(a.len, b.len);
    if (dim == 0) return 0;

    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    // SIMD path
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

    // Scalar remainder
    while (i < dim) : (i += 1) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0 or norm_b == 0) return 0;
    return dot / (@sqrt(norm_a) * @sqrt(norm_b));
}

/// Flat brute-force index for cosine similarity search.
pub const FlatIndex = struct {
    /// All embeddings: [n_docs * dim] flat layout.
    embeddings: []f32,
    /// Precomputed L2 norms for each document.
    norms: []f32,
    /// Number of documents.
    n_docs: usize,
    /// Embedding dimension.
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

    /// Build index from a flat array of embeddings [n_docs * dim].
    /// Takes ownership of the data.
    pub fn build(self: *FlatIndex, embeddings: []f32, n_docs: usize) !void {
        if (self.embeddings.len > 0) self.allocator.free(self.embeddings);
        if (self.norms.len > 0) self.allocator.free(self.norms);

        self.embeddings = embeddings;
        self.n_docs = n_docs;
        self.norms = try self.allocator.alloc(f32, n_docs);

        // Precompute norms
        for (0..n_docs) |d| {
            const row = self.embeddings[d * self.dim .. (d + 1) * self.dim];
            var s: f32 = 0;
            var j: usize = 0;
            while (j + SIMD_WIDTH <= self.dim) : (j += SIMD_WIDTH) {
                const v: VecF32 = row[j..][0..SIMD_WIDTH].*;
                const sq = v * v;
                s += @reduce(.Add, sq);
            }
            while (j < self.dim) : (j += 1) {
                s += row[j] * row[j];
            }
            self.norms[d] = @sqrt(s);
        }
    }

    /// Search for top-k most similar documents to query.
    /// Returns results sorted by descending similarity.
    pub fn search(self: *const FlatIndex, query: []const f32, top_k: usize, threshold: f32, results_buf: []SearchResult) usize {
        if (self.n_docs == 0 or query.len == 0) return 0;

        const dim = @min(query.len, self.dim);

        // Compute query norm
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

        // Scan all documents, maintain a min-heap of top-k
        const k = @min(top_k, results_buf.len);
        var heap_size: usize = 0;

        for (0..self.n_docs) |d| {
            const d_norm = self.norms[d];
            if (d_norm == 0) continue;

            const row = self.embeddings[d * self.dim .. d * self.dim + dim];

            // Dot product with SIMD
            var dot: f32 = 0;
            {
                var j: usize = 0;
                while (j + SIMD_WIDTH <= dim) : (j += SIMD_WIDTH) {
                    const va: VecF32 = query[j..][0..SIMD_WIDTH].*;
                    const vb: VecF32 = row[j..][0..SIMD_WIDTH].*;
                    dot += @reduce(.Add, va * vb);
                }
                while (j < dim) : (j += 1) {
                    dot += query[j] * row[j];
                }
            }

            const score = dot / (q_norm * d_norm);
            if (score < threshold) continue;

            if (heap_size < k) {
                results_buf[heap_size] = .{ .doc_idx = d, .score = score };
                heap_size += 1;
                if (heap_size == k) {
                    // Build min-heap
                    heapify(results_buf[0..heap_size]);
                }
            } else if (score > results_buf[0].score) {
                results_buf[0] = .{ .doc_idx = d, .score = score };
                siftDown(results_buf[0..heap_size], 0);
            }
        }

        // Sort results by descending score
        const result_slice = results_buf[0..heap_size];
        std.sort.insertion(SearchResult, result_slice, {}, struct {
            pub fn lessThan(_: void, lhs: SearchResult, rhs: SearchResult) bool {
                return lhs.score > rhs.score; // descending
            }
        }.lessThan);

        return heap_size;
    }
};

/// Build a min-heap from an array.
fn heapify(buf: []SearchResult) void {
    if (buf.len <= 1) return;
    var i = buf.len / 2;
    while (i > 0) {
        i -= 1;
        siftDown(buf, i);
    }
}

/// Sift down element at index i in a min-heap.
fn siftDown(buf: []SearchResult, start: usize) void {
    var i = start;
    while (true) {
        var smallest = i;
        const left = 2 * i + 1;
        const right = 2 * i + 2;
        if (left < buf.len and buf[left].score < buf[smallest].score) smallest = left;
        if (right < buf.len and buf[right].score < buf[smallest].score) smallest = right;
        if (smallest == i) break;
        const tmp = buf[i];
        buf[i] = buf[smallest];
        buf[smallest] = tmp;
        i = smallest;
    }
}

// ─── tests ───────────────────────────────────────────────────────────
test "cosine similarity identical vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const sim = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sim, 0.001);
}

test "cosine similarity orthogonal vectors" {
    const a = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    const sim = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sim, 0.001);
}

test "cosine similarity opposite vectors" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0 };
    const sim = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), sim, 0.001);
}

test "flat index search" {
    const allocator = std.testing.allocator;
    const dim = 8;
    const n_docs = 4;

    var idx = FlatIndex.init(allocator, dim);
    defer idx.deinit();

    // Create 4 embeddings
    const data = try allocator.alloc(f32, n_docs * dim);
    // doc 0: [1,0,0,...] – unit x
    // doc 1: [0,1,0,...] – unit y
    // doc 2: [1,1,0,...] – diagonal xy
    // doc 3: [-1,0,0,...] – negative x
    @memset(data, 0);
    data[0 * dim + 0] = 1.0;
    data[1 * dim + 1] = 1.0;
    data[2 * dim + 0] = 1.0;
    data[2 * dim + 1] = 1.0;
    data[3 * dim + 0] = -1.0;

    try idx.build(data, n_docs);

    // Query: [1,0,0,...] – should match doc0, then doc2
    var query = [_]f32{0} ** dim;
    query[0] = 1.0;

    var results: [4]SearchResult = undefined;
    const n = idx.search(&query, 4, -1.0, &results);
    try std.testing.expect(n >= 2);
    try std.testing.expectEqual(@as(usize, 0), results[0].doc_idx); // best match
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), results[0].score, 0.001);
}
